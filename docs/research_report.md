# ONCOGEN-X: Research Report

- **Project:** Longitudinal Genomic Intelligence
- **Report type:** Research and implementation submission
- **Status:** Descriptive data pipeline implemented; real longitudinal forecasting experiment not performed
- **Report date:** 11 October 2026

## Abstract

ONCOGEN-X is a research prototype designed to ask whether patient-level genomic history can improve prediction of alteration status at the next genuinely observed sample, compared with the current genomic state alone. The repository contains a cBioPortal METABRIC mutation-event downloader for cross-sectional description, synthetic data for dashboard demonstration and software tests, a guarded feature builder, and a patient-grouped benchmark implementation. The local METABRIC event file supports reproducible descriptive counts, but it has no serial sample times and no confirmed negative assay states. The synthetic fixtures verify software behavior only. No eligible curated longitudinal training table was available, so no real longitudinal model fitting, test-set evaluation, or forecasting experiment has been performed. No model metrics or clinical conclusions are reported.

## Research question, motivation, and objectives

### Research question

On an independently verified cohort of serial tumour samples with comparable assay coverage, do past-only trajectory features improve next-observation alteration-state prediction over the current gene-state vector?

### Motivation

Cancer sequencing describes sampled material at a time and place. Repeated samples can potentially reveal persistence and change, but an apparent gain or loss can also arise from spatial heterogeneity, sample composition, panel differences, sequencing depth, or detection limits. A prediction study therefore needs justified collection order and explicit evidence that a negative state was measured. A mutation event list by itself cannot provide those conditions.

### Objectives

1. Provide a reproducible static descriptive path for public METABRIC mutation events.
2. Separate real cohort description from synthetic interface and test fixtures.
3. Build consecutive-observation targets and history features without using future observations as predictors.
4. Compare static and temporal representations with Logistic Regression (LR) and Random Forest (RF) under the same patient-held-out split, once eligible longitudinal data exist.
5. Preserve provenance, report uncertainty and limitations, and avoid clinical interpretation of unvalidated predictions.

These objectives describe the software and the planned experiment. Objective 4 has not been completed on real longitudinal data.

## System architecture

| Component | Role |
|---|---|
| `src/download_cbioportal.py` | Requests the METABRIC sample list and mutation profile from cBioPortal, maps sample IDs to patient IDs, writes compressed source responses and a normalized event table, and generates descriptive summaries/figures. It sets one explicit cross-sectional timepoint label. |
| `src/data_loader.py`, `src/preprocessing.py` | Load, validate, and filter the event-format data used by the dashboard. |
| `dashboard/app.py` | Streamlit interface for overview, genomic records, patient/sample exploration, alteration summaries, a curated-state feature inspector, forecast status, model results, explainability, and roadmap. It does not create forecasts. |
| `src/temporal_features.py` | Validate a separate wide binary state table, construct one row for each consecutive observed-sample transition, build cutoff-only features, and place the next sample's gene states in target columns. |
| `src/train.py`, `src/evaluation.py` | Read a curated wide CSV and run LR/RF for static and temporal representations with a single patient-grouped split; save metrics, predictions, run metadata, feature importances, and fitted estimators. |
| `tests/` | Test data validation, feature/target construction, downloader normalization behavior, split separation, and benchmark execution on synthetic fixtures. Tests are software checks, not biological evidence. |

The event table used for METABRIC description and the wide state matrix needed for forecasting are distinct schemas. The METABRIC event table must not be reshaped into serial observations by sorting sample IDs.

## Data provenance and what was observed

### Real METABRIC data: descriptive analysis only

The downloader targets cBioPortal study `brca_metabric` and retrieves the study sample list plus its somatic mutation profile through the cBioPortal REST API. The local manifest records access date **2026-10-11** and counts. The pipeline joins mutations to patients by sample ID and removes exact duplicate normalized event rows. It does not download a whole-study archive, infer collection dates, impute absent genes, or establish assay coverage.

Reproducible counts from the local normalized file `data/processed/brca_metabric_mutations.csv`:

| Description | Count |
|---|---:|
| Study sample-list records returned | 2,509 |
| Mutation records returned by profile endpoint | 13,183 |
| Normalized event rows | 13,181 |
| Patients represented by mutation events | 2,358 |
| Samples represented by mutation events | 2,358 |
| Genes represented | 173 |
| Mutation event rows without a patient join | 0 |

The 2,358 event-bearing samples are not the full study sample list; samples without a mutation event do not appear as negative rows in this event table. These are **event-row counts**, not patient-level alteration prevalence. More than one event may occur for a gene in a sample.

The ten most frequent genes by event-row count in the current local file are PIK3CA (1,112), TP53 (888), AHNAK2 (537), MUC16 (499), SYNE1 (341), MAP3K1 (337), KMT2C (318), GATA3 (280), AHNAK (246), and CDH1 (243). These values are counts of normalized mutation event rows, not counts of positive patients or samples.

The mutation classification counts are Missense_Mutation (10,196), Nonsense_Mutation (837), Frame_Shift_Del (825), Frame_Shift_Ins (476), Splice_Site (403), In_Frame_Del (250), Splice_Region (123), In_Frame_Ins (61), Nonstop_Mutation (5), and Translation_Start_Site (5). The event table has one constant `timepoint` value, `single cross-sectional sample`, and one `genomic_status` value, `Observed alteration`. It cannot establish temporal order or measured negatives. The gene-frequency and mutation-class figures are descriptive visualizations of these event rows only.

These counts are reproducible from the downloaded, locally ignored raw/processed files by running the downloader and inspecting its normalized CSV and `outputs/metrics/dataset_summary.json`. A new API retrieval may return changed counts. No restricted dataset is included in this submission; raw and processed cohort files remain ignored by Git, while the provenance manifest and aggregate summary are retained.

### Synthetic data: software demonstration only

`data/sample/demo_genomic_data.csv` is fictional interface data and is labeled synthetic. The benchmark test constructs a separate synthetic fixture of 16 patients, three ordered observations per patient, and two binary genes to exercise the four model/representation combinations. The temporal-feature tests also use small fabricated states. None of these fixtures are observations from people, research evidence, or trained models for the METABRIC cohort. Their successful execution establishes only that code paths work on those artificial inputs.

### Real longitudinal experiment: not performed

No valid curated patient-by-sample longitudinal state table is present. The project has not fitted or evaluated a model on real longitudinal data. It therefore reports no experimental model performance, feature-importance findings, statistical comparison, future forecasts, or clinical claims. The implemented benchmark has only been exercised by the synthetic software test.

## Forecast input eligibility and preprocessing

The required wide CSV schema is:

```text
patient_id,sample_id,timepoint,<gene1>,<gene2>,...
```

It must have at least one gene column and no extra metadata columns. Each row represents one serially collected patient/sample. Patient/sample pairs and timepoints within each patient must be unique. Timepoints must be consistently finite numeric values or valid ISO dates/datetimes, and their order must be verified from genuine collection metadata. Labels such as `T1`, `baseline`, or a sample ID do not establish order. The code can validate format and uniqueness but cannot verify chronology provenance.

Every selected gene must be observed as binary at every included row. `1` means detected; `0` means that gene was adequately assayed in that sample and no qualifying alteration was found. Unknown, unsequenced, low-coverage, and unreported values cannot be entered as zero. The current schema has no assay mask, so curators must restrict the table to genes with comparable adequate coverage across all included samples. Missing gene values are rejected, not imputed. The minimum of four patients with transitions is a software guard, not a justified sample-size target.

For each patient, rows are chronologically ordered and each adjacent pair forms one transition. A patient with one sample yields no transition. For transition cutoff sample *t*, each gene's target is its binary state in the immediately next observed sample. This is a next-observation target, not a fixed-duration horizon and not a clinical event target.

## Static and temporal representations

The benchmark trains one binary classifier per target gene and evaluates two predictor sets:

- **Static/current-state representation:** current binary state for every selected gene plus the current alteration count. It uses the cutoff sample's current state, without the earlier history features.
- **Temporal representation:** current-state features plus history-derived values computed through and including the cutoff: number of observations so far; per-gene historical frequency, consecutive persistence, ever-observed status, and steps since last acquisition; acquisition and loss counts since the previous observation; and pairwise historical co-occurrence frequencies. Date-based inputs additionally produce elapsed days between prior observations. The current target observation is excluded from these features.

The feature builder calculates labels in `target__<gene>` columns. `feature_columns` excludes target columns, patient/sample identifiers, and target sample/timepoint metadata. The static representation selects only current-state and current-burden predictors; temporal adds the history features. The same transition rows and target labels are used for both representations.

## Evaluation, leakage controls, and planned metrics

The implemented comparison includes LR and RF with both representations: four configurations total. `GroupShuffleSplit` creates one fixed-seed split holding out 25% of patient IDs; that split is reused for every configuration. All transitions from a held-out patient remain in test. LR standardization is inside its scikit-learn pipeline and LR uses balanced class weights; RF uses balanced subsampling. The default classification threshold is 0.5. A gene is skipped for a model if its training target has only one class.

Leakage controls in the implementation include: (1) patient-level split groups rather than transition-row split; (2) features use only observations at or before the cutoff; (3) next-sample labels and target metadata are excluded from predictors; (4) scaling is fitted within the LR training pipeline; and (5) both representations share the same split and target rows. These controls prevent direct use of future rows and cross-patient train/test overlap. They cannot prevent confounding, leakage already embedded in a curator's source data, or bias caused by nonuniform sampling/assays.

For a future eligible run, the current code writes per-gene accuracy, precision, recall, F1, confusion counts, ROC-AUC, and average precision (PR-AUC) when defined; it also writes gene-macro means and pooled micro metrics. ROC-AUC and PR-AUC are undefined when the held-out labels contain one class. The split metadata and held-out predictions are saved with the metrics. These are planned experimental outputs; no values from a real longitudinal run exist. Row-level transition pooling can give patients with more samples more weight, and the single holdout has no repeated-split or cross-validation uncertainty estimate. A real study should predefine patient-level aggregation, eligibility, model selection, and uncertainty analysis.

## Interpretability approach

The current dashboard can display RF impurity-based feature importances from a completed run. LR coefficients are available in saved estimators but are not currently exported as a dedicated report. Importance is model- and dataset-dependent; it is not evidence of causality, mechanism, clinical utility, or a biological driver. Any eventual interpretation would need stability checks across patient splits and careful handling of correlated history features.

## Results and limitations

The only research data result reported here is the reproducible descriptive count summary above. The synthetic test's execution results are not an experiment. There are no real longitudinal model results. METABRIC's event-only, cross-sectional structure cannot answer the forecast question. The current pipeline also cannot independently establish time provenance or assay coverage from input values alone.

Other limitations include binary states that omit variant identity, clonality, tumour region, depth, purity, treatment, and assay sensitivity; potential sampling-related apparent gains/losses; the single patient holdout; potentially unstable rare-gene estimates; and row-level weighting of patients with many transitions. No LSTM/GRU experiment was run or warranted without a verified serial cohort and enough independent patient sequences. This prototype is not for diagnosis, treatment decisions, or clinical risk prediction.

## Future work

Curate an appropriately licensed repeat-sample cohort with source-backed timestamps/order, comparable per-sample assay coverage, and measured negative states. Document eligibility and missingness before analysis. Then run the locked LR/RF static-versus-temporal comparison, report patient counts and class prevalence, assess uncertainty and patient-level weighting, perform sensitivity analyses for intervals/coverage where supported, and seek external validation. Consider sequence models only if the number and structure of independent trajectories justify them. Update this report only with runs performed on eligible data.

## Reproducibility (Windows PowerShell)

Run commands from the repository root in Windows PowerShell. Python 3.10 or later is required; the checked-in `requirements.txt` lists NumPy, pandas, scikit-learn, joblib, Streamlit, Plotly, requests, and pytest. The workspace's working virtual environment uses Python 3.12.14. Dependency minimums are specified rather than a fully locked environment; record installed versions for any formal run.

```powershell
# Create and activate the environment (skip creation if .venv already works)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Acquire current public METABRIC descriptive data and outputs
python -m src.download_cbioportal

# Reproduce the event-row counts reported above from the normalized CSV
python -c 'import pandas as pd; d=pd.read_csv("data/processed/brca_metabric_mutations.csv"); print(d.gene.value_counts().head(10).to_string()); print(d.alteration_type.value_counts().to_string())'

# Run the existing tests
.\.venv\Scripts\python.exe -m pytest -q

# Optional: run only with an independently curated eligible longitudinal CSV
python -m src.train path\to\curated_longitudinal_states.csv --output outputs --seed 42

# Launch dashboard locally
python -m streamlit run dashboard/app.py
```

The downloader contacts the live cBioPortal API; a rerun can produce different counts. Do not use the METABRIC event CSV or synthetic demo as the training CSV. For an eligible future run, retain the input provenance, access/consent basis, software versions, seed, patient split, and outputs without committing private or restricted source data. Dashboard views are specified in [`screenshots/README.md`](../screenshots/README.md); no screenshots are included in this report.

## Literature and reference verification

The 14 annotated entries in [`literature/literature_review.md`](../literature/literature_review.md) provide context, not evidence that this experiment has been completed or that the question is unprecedented. On 11 October 2026, each listed DOI was queried through the Crossref works API; all 14 returned HTTP 200 and their returned titles, journals, volume/issue, and page or article identifiers were checked against the bibliography. Publisher DOI destinations are linked in that review. One corporate author is corrected there to the Crossref title metadata, **The Cancer Genome Atlas Network**. This is bibliographic verification, not an independent appraisal of every paper's scientific conclusions.
