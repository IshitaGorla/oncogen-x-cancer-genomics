"""Fetch only METABRIC sample metadata and somatic mutations from cBioPortal.

The selected cohort is cross-sectional for this project. Sample identifiers
are not converted to serial timepoints and this dataset is not forecast input.
"""
from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
STUDY = "brca_metabric"
API = "https://www.cbioportal.org/api"
PAGE_SIZE = 10000


def _get_pages(session: requests.Session, path: str) -> list[dict]:
    values = []
    page = 0
    while True:
        response = session.get(f"{API}{path}", params={"projection": "SUMMARY",
                               "pageSize": PAGE_SIZE, "pageNumber": page}, timeout=90)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError(f"Expected a list from cBioPortal {path}.")
        values.extend(batch)
        if len(batch) < PAGE_SIZE:
            return values
        page += 1


def _save_json_gz(path: Path, values: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(values, handle, separators=(",", ":"))


def download(output: Path = ROOT / "data") -> Path:
    raw = output / "raw"
    processed = output / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"Accept": "application/json", "User-Agent": "ONCOGEN-X-research-prototype/1.0"})
        samples = _get_pages(session, f"/studies/{STUDY}/samples")
        sample_to_patient = {s["sampleId"]: s.get("patientId") for s in samples if s.get("sampleId")}
        if not sample_to_patient:
            raise ValueError("cBioPortal returned no sample identifiers for the selected study.")
        sample_ids = list(sample_to_patient)
        mutations = []
        for start in range(0, len(sample_ids), 1000):
            batch_ids = sample_ids[start:start + 1000]
            page = 0
            while True:
                response = session.post(
                    f"{API}/molecular-profiles/{STUDY}_mutations/mutations/fetch",
                    params={"projection": "DETAILED", "pageSize": PAGE_SIZE, "pageNumber": page},
                    json={"sampleIds": batch_ids}, timeout=120)
                response.raise_for_status()
                batch = response.json()
                if not isinstance(batch, list):
                    raise ValueError("cBioPortal returned an unexpected mutation response.")
                mutations.extend(batch)
                if len(batch) < PAGE_SIZE:
                    break
                page += 1

    raw_samples = raw / f"{STUDY}_samples.json.gz"
    raw_mutations = raw / f"{STUDY}_mutations.json.gz"
    _save_json_gz(raw_samples, samples)
    _save_json_gz(raw_mutations, mutations)
    events = _mutation_events(mutations, sample_to_patient)
    target = processed / f"{STUDY}_mutations.csv"
    events.to_csv(target, index=False)
    figures = output.parent / "outputs" / "figures"
    metrics_dir = output.parent / "outputs" / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    gene_counts = events.gene.value_counts().head(20).sort_values()
    gene_figure = px.bar(gene_counts, orientation="h", title="METABRIC: observed mutation event rows by gene (top 20)",
                         labels={"value": "Mutation event rows", "index": "Gene"}, color_discrete_sequence=["#0b8194"])
    gene_figure.update_layout(template="plotly_white", xaxis_title="Observed event rows", yaxis_title="Gene")
    gene_figure.write_html(figures / "metabric_gene_frequency.html", include_plotlyjs="cdn")
    type_counts = events.alteration_type.value_counts().head(20).sort_values()
    type_figure = px.bar(type_counts, orientation="h", title="METABRIC: observed mutation classifications",
                         labels={"value": "Mutation event rows", "index": "Classification"}, color_discrete_sequence=["#168a78"])
    type_figure.update_layout(template="plotly_white", xaxis_title="Observed event rows", yaxis_title="Mutation classification")
    type_figure.write_html(figures / "metabric_mutation_class.html", include_plotlyjs="cdn")
    dataset_summary = {"study_id": STUDY, "sample_list_samples": len(samples),
                       "mutation_records_returned": len(mutations), "normalized_event_rows": len(events),
                       "patients_with_mutation_rows": int(events.patient_id.nunique()),
                       "samples_with_mutation_rows": int(events.sample_id.nunique()),
                       "genes_with_mutation_rows": int(events.gene.nunique()),
                       "temporal_validity": "cross-sectional; not a forecasting cohort"}
    (metrics_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    manifest = {
        "study_id": STUDY,
        "study_name": "Breast Cancer (METABRIC, Nature 2012 & Nat Commun 2016)",
        "publications": [
            {"citation": "Curtis et al., Nature 2012, 486:346-352", "doi": "10.1038/nature10983"},
            {"citation": "Pereira et al., Nature Communications 2016, 7:11479", "doi": "10.1038/ncomms11479"}
        ],
        "source_url": "https://www.cbioportal.org/study/summary?id=brca_metabric",
        "api_documentation": "https://docs.cbioportal.org/web-api-and-clients/",
        "api_base": API,
        "accessed": date.today().isoformat(),
        "raw_files": [str(raw_samples.relative_to(output)), str(raw_mutations.relative_to(output))],
        "processed_file": str(target.relative_to(output)),
        "figure_files": ["outputs/figures/metabric_gene_frequency.html", "outputs/figures/metabric_mutation_class.html"],
        "temporal_validity": "Cross-sectional study. No temporal order is inferred.",
        "sample_count": len(samples), "mutation_record_count": len(mutations),
        "processed_event_count": len(events),
        "unmatched_mutation_samples": int(events.patient_id.isna().sum()) if len(events) else 0,
        "data_use": "Respect cBioPortal and study attribution; consult the study publication and portal terms before redistribution."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def _mutation_events(mutations: list[dict], sample_to_patient: dict[str, str]) -> pd.DataFrame:
    rows = []
    for mutation in mutations:
        gene = mutation.get("gene") or {}
        sample_id = mutation.get("sampleId")
        gene_name = gene.get("hugoGeneSymbol") or mutation.get("geneSymbol") or mutation.get("hugoGeneSymbol")
        if not gene_name:
            continue
        rows.append({
            "patient_id": mutation.get("patientId") or sample_to_patient.get(sample_id),
            "sample_id": sample_id,
            "cancer_type": "Breast cancer",
            "timepoint": "single cross-sectional sample",
            "gene": gene_name,
            "alteration": mutation.get("proteinChange") or mutation.get("mutationStatus") or "Not reported",
            "alteration_type": mutation.get("mutationType") or "Not reported",
            "genomic_status": "Observed alteration",
            "data_source": f"cBioPortal {STUDY} (cross-sectional; not longitudinal)",
        })
    return pd.DataFrame(rows).drop_duplicates()


if __name__ == "__main__":
    print(download())
