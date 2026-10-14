# ONCOGEN-X — Longitudinal Genomic Intelligence

Research prototype for testing whether past-only patient trajectory features improve prediction of alteration status at a patient's next **genuinely observed** sample.

## Research integrity and current data

- The included `data/sample/demo_genomic_data.csv` is fictional synthetic interface-demo data. It is not used as evidence or as a trained model dataset.
- `python -m src.download_cbioportal` downloads the public cBioPortal METABRIC mutation study for real **static descriptive exploration**. METABRIC is not treated as a verified serial cohort by this project. Mutation records alone do not establish negative assay results or temporal order.
- The benchmark accepts a separately curated wide table with one patient/sample/timepoint per row and complete 0/1 gene states. A 0 must mean the gene was assayed and confirmed absent. Do not fill an unmeasured/unknown value with zero.
- No performance result is provided until the benchmark is run on a suitable longitudinal dataset. The repo does not contain real serial training data, saved trained models, or clinical forecasts.

See [dataset documentation](docs/dataset.md), [research report](docs/research_report.md), and the [source-grounded literature review](literature/literature_review.md).

## Setup (Windows PowerShell)

Python 3.10+ is required. Verify that the selected interpreter actually launches (`py -0p` lists registered versions). In this workspace, the Windows Python Launcher pointed to a missing Store runtime, so the old venv could not run. After installing a standard Python from python.org if needed, from the project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the launcher is unavailable or points to a missing executable, install Python from python.org and enable the launcher, or replace `py -3` with the full path to the installed `python.exe`. The current `.venv` was repaired with Python 3.12.14 in this workspace.

## Get and inspect public data

```powershell
python -m src.download_cbioportal
```

The script retrieves only sample metadata and somatic mutation records through cBioPortal's REST API, saves compressed responses in `data/raw/`, normalized alteration rows to `data/processed/brca_metabric_mutations.csv`, and provenance/counts to `data/manifest.json`. It does not assign temporal order. See source links in `data/manifest_template.json` and [dataset notes](docs/dataset.md).

## Prepare a legitimate longitudinal state table

Create a CSV with columns `patient_id,sample_id,timepoint` plus gene columns containing only `0` or `1`. Sortability of `timepoint` must reflect genuine time. Use only genes measured with comparable coverage at every included sample, or curate a separate observed-state indicator before constructing this input. **Do not use the cBioPortal METABRIC mutation-event file as this wide table.** A format-only template is in `data/sample/longitudinal_states_template.csv`; its example rows are deliberately blank.

## Run benchmark

```powershell
python -m src.train path\to\curated_longitudinal_states.csv --output outputs --seed 42
```

This runs logistic regression and random forest using static current-state and temporal feature representations on the same single 25% patient-group holdout. Outputs include per-gene metrics, held-out predictions, run metadata, and saved estimators under `outputs/`. ROC-AUC and PR-AUC are unavailable when the held-out target has only one class. The split has no repeated-CV uncertainty estimate; describe that limitation. No neural model is included because this workspace has no verified serial cohort to justify one.

## Launch dashboard

```powershell
streamlit run dashboard/app.py
```

The app supports schema-validated CSV upload, cohort summaries, patient/sample exploration, and interactive Plotly charts. The shipped demo data are visibly marked synthetic. The application does not invent model output; forecasting and model-comparison pages state when trained results are unavailable.

## Tests

```powershell
python -m pytest
```

## Repository publication

This workspace has a local Git repository initialized, but no commit or GitHub/GitLab remote. The current managed workspace grants read-only access to `.git`, so stage and commit from your normal PowerShell session. After reviewing the files and study data terms, you can publish manually:

```powershell
git add .
git commit -m "Build ONCOGEN-X research prototype"
gh repo create cancer-genomic-forecasting --public --source . --remote origin --push
```

## Five-minute demo

1. Open the overview and explain the research question and the synthetic-data notice.
2. Explore the shipped demo dataset, filters, and data-quality table.
3. Open a patient timeline and describe it as a synthetic interface demonstration.
4. Show the METABRIC downloader and explain why cross-sectional samples do not validate temporal forecasting.
5. Explain the guarded state-table requirements and patient-held-out static-versus-temporal benchmark; show the report's present limitations.

Dashboard capture guidance is in [screenshots/README.md](screenshots/README.md). No screenshots are included yet.

## Limitations and next steps

The most important missing asset is a sufficiently sized, legitimately ordered repeat-sample cohort with per-sample assay coverage and measured negative states. Curate that cohort and document licensing/access before making forecasting claims. Then run the benchmark, inspect leakage, report uncertainty, and only consider LSTM/GRU if independent trajectory counts support them. The project is a research prototype and is not intended for clinical use.
