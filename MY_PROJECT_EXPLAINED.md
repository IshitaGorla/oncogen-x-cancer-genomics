# ONCOGEN-X, explained for learning

This guide describes the project that is actually in this repository. It separates three things that can look similar at first:

1. **Real METABRIC data** used for descriptive mutation-event counts and charts.
2. **Synthetic data** used to make the dashboard and tests run without a real longitudinal cohort.
3. **The real longitudinal forecasting experiment**, which has not yet been performed.

The guide does not report model performance. The toy tables below are explicitly illustrative and are not rows from METABRIC or the synthetic CSV.

---

## 1. The entire project in simple words

Imagine taking a few snapshots of a tumour from the same patient at different times. Each snapshot can show which genes had certain alterations. ONCOGEN-X is a small research application for exploring mutation records and for preparing code that could compare a patient's current gene state with the patient's earlier history.

The question behind the project is: **if we know a patient's earlier genomic observations, can we predict which genes will have an alteration in that patient's next observed sample better than if we only use the current sample?**

A **gene** is a stretch of DNA that carries biological instructions. A **genetic alteration** is a change or difference in DNA or a gene's copy count. For example, an alteration record might say that a protein change occurred in TP53. That does not, by itself, tell us how common it is, whether it caused a cancer, or what will happen to a patient.

Genomic data can be recorded in different shapes. The METABRIC download in this project is an **event table**: a row means that a mutation event was reported for a sample and gene. The training code expects a different shape, a **wide state table**: one row per patient/sample, with one column per gene and a 0 or 1 in every gene cell.

In this project, “forecasting” means predicting a gene's binary alteration state in the next sample that was actually observed. It does not mean predicting a date, treatment response, survival, or a patient's clinical future. The code has a training and evaluation pipeline for a properly curated wide table, but the repository has no appropriate real longitudinal table. Therefore, it has not demonstrated real forecasting.

What ONCOGEN-X can do today:

- Download and summarize real cBioPortal METABRIC mutation events for static description.
- Show synthetic example records in a Streamlit dashboard.
- Validate and turn a separate complete, ordered, binary state table into past-only features and next-observation labels.
- Run four classifier configurations on synthetic test data to check the software path.

What it cannot do today:

- Infer that METABRIC samples are in chronological order.
- Treat an absent mutation-event row as a confirmed negative test.
- Verify that every gene was adequately assayed in every input sample.
- Produce a scientifically supported real longitudinal model result or clinical forecast.

**Studying mutation patterns** counts and visualizes records already in the data. **Predicting future mutation status** requires time-ordered samples, measured positive and negative states, a trained model, and a separate evaluation. A descriptive chart is not a forecast.

## 2. Why this project matters

Cancer cells can change over time, and different parts of one tumour can also differ from each other. Repeated genomic samples might show which detected alterations persist or appear later. That could make a useful research question, but an apparent difference between two samples is not automatically a biological change. The samples might come from different tumour locations, use different sequencing panels, or have different coverage and detection limits.

The research goal in everyday language is:

> For patients whose samples have a real, documented order and comparable testing, does remembering earlier gene states help predict the next sample better than looking only at the current state?

This is a computational research question. A **clinical tool** would need much more: reliable evidence on appropriate patient populations, external validation, clinical review, safety/governance processes, and evidence that using it improves decisions. ONCOGEN-X is not a diagnostic system, is not intended to guide treatment, and has not been clinically validated.

## 3. The technology stack

The table names technologies actually used or declared by this repository. A package being listed in `requirements.txt` does not always mean the project imports it directly.

| Technology | What it is and why this project uses it | Where it appears and a project example |
|---|---|---|
| **Python** | The programming language that connects the downloader, data processing, feature builder, model code, dashboard, and tests. | The files under `src/`, `dashboard/app.py`, and `tests/` are Python. Running `python -m src.download_cbioportal` executes the downloader module. |
| **Pandas** | A library for working with tables, similar to a spreadsheet that can be filtered and transformed in code. | `src/download_cbioportal.py` creates a DataFrame for event rows; `src/temporal_features.py` sorts and checks state tables; the dashboard filters tables. `value_counts()` counts mutation event rows by gene. |
| **NumPy** | A library for numerical arrays and calculations. | NumPy is listed in `requirements.txt`, and scikit-learn depends on numerical array operations. The project source does not directly import NumPy by name. |
| **Streamlit** | A Python framework that turns Python code into an interactive local web application. | `dashboard/app.py` uses controls such as `st.radio`, `st.selectbox`, `st.file_uploader`, tables, and Plotly charts. |
| **Plotly Express** | A charting library that builds interactive charts from table-like data. | `plotly.express` is used by the dashboard and downloader for bars, pies, scatter plots, and heatmaps. The downloader saves two HTML figures. |
| **scikit-learn** | A machine-learning library containing classifiers, preprocessing, data splitting, and evaluation metrics. | `src/evaluation.py` uses `LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `GroupShuffleSplit`, and metric functions. |
| **Requests and the cBioPortal API** | Requests sends HTTP messages; an API is a defined way for software to ask another service for data. | `src/download_cbioportal.py` calls cBioPortal's study sample endpoint and mutation fetch endpoint. It does not use a browser to download the data. |
| **CSV** | A plain-text table format with comma-separated columns. | The processed METABRIC mutation events are saved as `data/processed/brca_metabric_mutations.csv`. Training reads a different, curated wide CSV using `pd.read_csv`. |
| **JSON** | A text format for structured values such as lists and key/value records. | The API responses are JSON objects/lists. The downloader saves them in compressed `.json.gz` files and writes a JSON manifest and dataset summary. |
| **gzip / `.json.gz`** | gzip compresses a file; `.json.gz` means JSON stored in compressed form. | `_save_json_gz` writes the raw sample and mutation responses to `data/raw/`. It saves storage space while retaining API responses. |
| **joblib** | A Python tool that saves and reloads fitted models efficiently. | `src/evaluation.py` writes one `.joblib` estimator file per fitted gene/model/representation under the output `models` folder. No real-run models are currently present. |
| **pytest and unittest** | Testing tools. pytest discovers and runs tests; the tests themselves use Python's `unittest` assertions. | Run `python -m pytest -q`. `tests/test_temporal_features.py` checks validation and feature logic; `tests/test_evaluation.py` executes the benchmark on a synthetic fixture. |
| **Virtual environment (`.venv`)** | A project-local Python installation area for packages, isolated from other projects. | The working environment in this workspace is Python 3.12.14. `.gitignore` excludes `.venv/`; the repo's dependencies are listed in `requirements.txt`. |

## 4. How the entire system works, step by step

The overall idea resembles a small data workshop: download records, put them into a useful table, explore them, and keep the separate forecasting machinery ready for an eligible dataset. The stages are not all automatically connected.

### Step A: Ask cBioPortal for public study records

`src/download_cbioportal.py` sets the study ID to `brca_metabric` and API base to `https://www.cbioportal.org/api`.

- `_get_pages()` requests the study's sample list in pages of up to 10,000 records.
- The downloader builds a lookup from each API `sampleId` to its `patientId`.
- It asks the mutation profile for mutations, sending sample IDs in batches of 1,000 and fetching pages of up to 10,000 events.
- If the API request fails, Requests raises an error. If the API response is not a list, the code raises a `ValueError`.

This is a live network step. A future download can return different counts if the source changes.

### Step B: Store raw API responses

`_save_json_gz()` writes the sample response to `data/raw/brca_metabric_samples.json.gz` and mutation response to `data/raw/brca_metabric_mutations.json.gz`. “Raw” means the response is kept close to what the API supplied, before the project's event-table transformation.

### Step C: Turn mutation objects into event rows

`_mutation_events()` loops through mutation records. It takes the sample ID, gene symbol, protein change, and mutation type when present. It gets a patient ID from the mutation response or sample-to-patient lookup. The output also adds `Breast cancer`, the constant timepoint `single cross-sectional sample`, `Observed alteration`, and a data-source note. Records without a gene symbol are skipped. Exact duplicate normalized rows are dropped.

The output is saved to `data/processed/brca_metabric_mutations.csv`. It has these columns: `patient_id`, `sample_id`, `cancer_type`, `timepoint`, `gene`, `alteration`, `alteration_type`, `genomic_status`, `data_source`.

This processing does **not** turn missing event rows into 0s. It does not infer collection dates. It does not create a longitudinal wide matrix.

### Step D: Make descriptive summaries and figures

Still inside `download()`, Pandas counts events by gene and alteration class. Plotly Express saves `outputs/figures/metabric_gene_frequency.html` and `outputs/figures/metabric_mutation_class.html`. The downloader writes `outputs/metrics/dataset_summary.json` and `data/manifest.json` with source, date, and count information.

These are summaries of mutation event rows. They are not patient prevalence, mutation risk, model results, or temporal trends.

### Step E: Load and explore data in the dashboard

`dashboard/app.py` chooses between a built-in synthetic event table and the METABRIC event CSV (the METABRIC option appears only if that CSV exists). `src/data_loader.py` checks for the required event-table columns, trims whitespace in strings, preserves missing values, and drops exact duplicate rows. `src/preprocessing.py` supplies filtering, text search, and missing-value summaries.

The dashboard can also accept an uploaded long-format event CSV on its Genomic Data page. Separately, the Temporal Features page accepts an uploaded **wide binary state CSV**. These are different uploaders and different schemas.

### Step F: Create temporal features and labels from a curated table

The Temporal Features page and training CLI call `src/temporal_features.py`. It verifies IDs, unique samples/timepoints, timepoint format, and complete binary gene states. It orders each patient's samples and creates a transition from one observation to the next. The historical state becomes input columns; the next state becomes `target__GENE` columns.

The dashboard feature inspector displays/downloads the constructed feature rows. It does not fit a model. The separate CLI calls `src/evaluation.py` to train/evaluate.

### Step G: Fit and evaluate models only through the CLI

`src/train.py` reads the curated state CSV and treats every column except `patient_id`, `sample_id`, and `timepoint` as a gene column. It calls `run_benchmark()` from `src/evaluation.py`. That function constructs the feature frame, splits by patient, fits the classifiers, calculates metrics, and writes outputs.

The dashboard does **not** invoke `run_benchmark()`. Its Model Performance and Explainability pages only read files if the CLI has created them. The Future Forecast page is a status/explanation screen; it does not make a new patient's forecast.

## 5. Understanding my dataset

### What is METABRIC?

METABRIC is a breast cancer molecular profiling study. This project accesses the cBioPortal study `brca_metabric` and downloads the sample list plus its somatic mutation profile. A **somatic mutation** is a DNA change found in tumour-acquired material; “somatic” distinguishes it from a change inherited in every cell. The downloaded profile is used here for static description.

### The important nouns

- **Patient:** the person-level identity used to connect one or more samples. `patient_id` is the grouping key for train/test separation.
- **Sample:** the particular tissue or specimen measurement. `sample_id` identifies a sample. A patient could have repeated samples in a genuinely longitudinal study, but sample IDs alone do not tell us their time order.
- **Gene:** the named gene associated with a mutation record, such as TP53.
- **Mutation record/event:** a reported alteration in a sample. It is one event row, not an entire patient profile.
- **Timepoint:** when a sample was collected, or a justified ordering value. In this METABRIC event file the timepoint is a literal constant label, not a date or sequence.

### What is in the actual download?

The current local manifest says the API returned 2,509 sample-list records and 13,183 mutation records. The normalized event CSV has 13,181 rows, represents 2,358 patients and 2,358 samples with mutation records, and contains 173 genes. The join had zero unmatched mutation rows in this download. The **sample-list count** is not the same as the number of samples that show up in the **event CSV**: a sample with no returned mutation event does not create a negative row in this processed file.

One sample can have events in many genes, and one gene/sample combination can in principle have multiple mutation events. Therefore 13,181 event rows are not 13,181 patients. The current normalized event counts are not enough to say how many patients were tested negative for each gene.

The sample list response in `data/raw/brca_metabric_samples.json.gz` retains API sample records used for the sample-to-patient lookup. The mutation response in `data/raw/brca_metabric_mutations.json.gz` retains the API mutation objects. The processed CSV is a flatter, easier-to-inspect list of normalized event rows.

`data/manifest.json` records the study ID/name, publication DOIs, source/API addresses, retrieval date, file paths, returned counts, processed row count, unmatched joins, cross-sectional caveat, and data-use note. `data/manifest_template.json` describes the manifest fields and expected files. `outputs/metrics/dataset_summary.json` is a smaller count summary.

The raw and processed cohort files are ignored by Git. That keeps downloaded data out of ordinary repository staging. The manifest and aggregate summary are preserved as useful provenance. Always check the applicable cBioPortal and study terms before redistributing data.

### Why this data cannot prove longitudinal forecasting

The processed records carry one repeated label: `single cross-sectional sample`. They contain no genuine per-sample timepoint sequence for this project. Also, the rows represent observed events. An absent row could mean that no event was returned, not necessarily that the gene was tested with adequate coverage and found normal.

### Tiny illustration only - not METABRIC data

The following invented table demonstrates what a curated state table could look like. These values are just an illustration; they are not observations from a person or from this project.

| patient_id | sample_id | timepoint | TP53 | KRAS |
|---|---|---:|---:|---:|
| example-A | example-A-1 | 1 | 0 | 0 |
| example-A | example-A-2 | 2 | 1 | 0 |
| example-A | example-A-3 | 3 | 1 | 1 |

In this toy example, `0` is meant to mean “tested adequately and not detected,” and `1` means “detected.” Those meanings must come from real assay records when making an actual training table. The digits 1, 2, 3 represent a known order in the illustration only.

## 6. Understanding the machine learning

### Start with a row of examples

A **feature** is an input clue given to the model. A **target** (also called a **label**) is the answer the model is asked to predict. One **training example** in this project is a patient's cutoff state/history and the observed gene states at that patient's next sample.

Imagine a teacher showing a student many examples with clues and known answers. A model is the program that adjusts its internal rules to connect clues to answers. It can then score new examples. It does not understand cancer biology just because it learned a pattern.

### Static and temporal inputs

- **Static features** here mean the genes' current 0/1 states at the cutoff sample plus the count of currently altered genes. They do not include the earlier trajectory summaries.
- **Temporal features** add information from the history through the cutoff: how often a gene was positive, how many consecutive observations it has remained positive, whether it was ever positive in the included history, how many observed steps since a detected 0-to-1 change, recent acquisition/loss counts, historical gene co-occurrence, and how many observations have occurred so far.
- **Next-timepoint prediction** in this implementation really means “predict the next observed sample.” It is not a fixed one-month, six-month, or one-year horizon.

Both versions get the same cutoff rows and next-sample labels. Comparing them asks whether the additional historical information helps on held-out patients.

### The two classifiers

**Logistic Regression (LR)** is a simple binary classifier. Despite “regression” in its name, it can classify 0 versus 1. It combines the input values with learned weights, then converts the result to a probability-like score between 0 and 1. In this code, feature values are standardized inside a pipeline, LR is allowed up to 1,000 iterations, and class weights are balanced. The code applies a 0.5 score threshold to create a 0/1 predicted label.

**Random Forest (RF)** is a collection of decision trees. A decision tree asks a sequence of questions about features; a forest combines many trees. Here it uses 300 trees, minimum leaf size 2, balanced subsampling, and a fixed random seed. A forest can represent nonlinear combinations, but its importance chart is not a causal explanation.

The code fits a separate binary model for each gene target. It runs four comparisons: static LR, temporal LR, static RF, and temporal RF. The feature sets and target structure differ, but the same patient split is reused.

### Patient-grouped train/test split

Training data are used to fit the model. Test data are kept aside to estimate how the model works on data it did not fit. If rows from the same person appear on both sides, a model could exploit person-specific patterns and appear better than it generalizes.

`GroupShuffleSplit` treats patient IDs as groups. The code holds out about 25% of patients in a single seeded split. All transitions for one patient go to either training or test, never both. The exact share of transitions need not be 25%, because the split is by patients, not by rows.

### Leakage and the protections in this code

**Data leakage** is when information unavailable at the intended prediction time sneaks into model inputs or evaluation. It is like giving a student the answer key during a test.

The implementation tries to prevent direct leakage by:

- Calculating history features only through the cutoff sample.
- Putting the next sample's gene state in `target__...` columns, then excluding every `target__` field from `feature_columns()`.
- Excluding patient IDs and sample/timepoint identifiers for the cutoff and target from model predictors.
- Splitting by patient before model fitting/evaluation.
- Fitting LR scaling inside the training pipeline, so the scaler learns from training rows rather than test rows.
- Using the same split for both representations and both classifiers.

This cannot verify that the input's dates are genuine, that the assay panel/coverage is comparable, or that upstream data do not already contain leakage. It also cannot remove confounding such as treatment, tumour site, or uneven sample schedules if those are not represented or controlled.

### What the evaluation numbers mean

Suppose an **illustrative** test set has 10 truly positive next-sample labels. A model predicts positive for 6 of them: 4 predictions are correct and 2 are false alarms. It misses 6 true positives.

- **Precision** = correct positive predictions / all positive predictions = 4 / 6 = about 0.67. Of the cases flagged positive, how many were right?
- **Recall** = correct positive predictions / all actual positives = 4 / 10 = 0.40. Of the actual positives, how many did it find?
- **F1-score** combines precision and recall using their harmonic mean. For these illustrative values, F1 is 0.50. It falls when either precision or recall is low.
- **Accuracy** is the fraction of all 0/1 labels classified correctly. With rare positive states, a model can get high accuracy by predicting 0 too often, so accuracy alone can be misleading.
- **ROC-AUC** measures how well the score ranks a randomly selected positive above a randomly selected negative across possible thresholds. It requires both classes in the evaluated labels in this implementation.
- **PR-AUC** here is scikit-learn's average precision. It summarizes precision/recall behavior for the positive class and is often informative when positives are uncommon. Its baseline depends on positive prevalence.

The code writes per-gene metrics, unweighted gene-macro averages, and pooled micro metrics for each model/representation. Macro means average over genes; micro metrics pool gene-transition labels. Because micro metrics pool rows, patients with more transitions can contribute more labels. If a gene's training set has only one class, that gene/model fit is skipped. If test labels have one class, ROC-AUC and PR-AUC are left undefined by this code. A single split does not give a confidence interval or show how much results vary under other splits.

### Feature importance and SHAP

The dashboard's Explainability page uses RF **impurity-based feature importance** if a file from a completed benchmark exists. The metric shows which features reduced tree impurity across the fitted forest. It can be biased toward some feature types and can distribute importance across correlated features. It is not a biological explanation.

**SHAP is not implemented**: there is no SHAP import or calculation in the project. LR coefficients are saved inside the serialized estimator, but the code does not export them as a dedicated report. No actual feature-importance output from a real longitudinal run is present.

### Why the software test can train a model

The test constructs artificial, fully observed binary trajectories for 16 fictional patient IDs with three ordered rows each, and two genes (TP53 and KRAS). That is enough to exercise input checks, create adjacent transitions, split patients, fit the four model/representation combinations, and write temporary predictions and model files.

This shows the code can execute on an input shaped like a curated state table. It does not show that real METABRIC is longitudinal, that the artificial states resemble real biology, or that the model is useful. The test's temporary artifacts are deleted after the test.

## 7. The temporal feature builder, slowly

The main functions are `_ordered_timepoints()`, `validate_states()`, `temporal_features()`, and `feature_columns()` in `src/temporal_features.py`.

### 7.1 `_ordered_timepoints()` decides what can be sorted

The function accepts finite numeric timepoints, numeric strings, or strings formatted as ISO dates/datetimes. Numeric strings are converted to numeric values, so `"2"` sorts before `"10"`. ISO values are parsed as UTC datetimes. Arbitrary labels such as `T1`, `T2`, or `baseline` are rejected because their text alone does not prove time order. Invalid calendar dates such as `2020-02-30` are rejected.

Passing this check does not prove that the value came from collection records. A curator must still verify that time order is real. The code does not infer chronology from sample names or the order of CSV rows.

### 7.2 `validate_states()` checks the input contract

Required metadata names are `patient_id`, `sample_id`, and `timepoint`. There must be at least one unique gene column; the gene columns cannot be metadata columns. Input column names must be unique. Patient, sample, and timepoint values cannot be missing; patient/sample IDs also cannot be blank after trimming. A patient/sample pair must be unique. A patient cannot have two rows at the same normalized timepoint. Every selected gene value must be exactly 0 or 1 (boolean values are accepted too); missing values are not treated as 0.

This check validates data shape. It cannot know whether a 0 was truly tested and negative, whether coverage was adequate, or whether an ordered numeric value represents a real date/sequence. It also cannot determine whether samples are spatial replicates or serial samples from their values alone.

### 7.3 `temporal_features()` sorts and walks through each patient

The function adds an internal normalized sort key, and for ISO date input an internal parsed timestamp. It sorts by patient, normalized time, then sample ID. Since duplicate patient/timepoint pairs have already been rejected, sample ID is not used to decide the order of tied observations.

For each patient it converts the ordered rows into records. At index 0 it uses the first record as the cutoff and the second record as the target. At index 1 it uses the first two records as history and the third as target, and so on. The last sample for each patient cannot become a cutoff because it has no next observed sample.

### 7.4 What each generated value means

For each transition row:

- `patient_id`: used by the evaluator for patient grouping; excluded from predictors.
- `cutoff_sample` and `cutoff_timepoint`: identity/time of the row whose history is available.
- `target_sample` and `target_timepoint`: next observation metadata; excluded from features.
- `current_alteration_count`: number of gene columns equal to 1 in the cutoff sample.
- `new_acquisition_count`: number of genes changing from 0 in the immediately previous sample to 1 now. On the patient's first cutoff this is 0 because there is no previous row.
- `loss_count_since_previous`: number changing from 1 in the immediately previous sample to 0 now. On the first cutoff this is 0.
- `observations_so_far`: number of rows in the history through the cutoff.
- `GENE__current`: that gene's 0/1 state at the cutoff.
- `GENE__historical_frequency`: fraction of included history rows with state 1, including the cutoff.
- `GENE__consecutive_persistence`: number of consecutive trailing history rows with state 1. It is 0 when the current state is 0.
- `GENE__ever_observed`: despite its name, this is `1` if a gene had state 1 at least once in the included history; it does not mean “ever assayed.”
- `GENE__last_acquired_steps`: number of observed row-to-row steps since the latest 0-to-1 change found in history. It is `-1` if no such change is found (for example, the gene starts at 1 in the available history).
- `GENE__cooccur_OTHER`: fraction of history rows where both gene and other gene were 1 at once. This is calculated from history through the cutoff. Each pair is represented in both directions by the code, so `A__cooccur_B` and `B__cooccur_A` contain the same number.
- `target__GENE`: the target gene's state in the next observed sample. `feature_columns()` excludes this.

For ISO dates only, `days_since_previous_sample` is the actual gap between the cutoff date and the previous date, in days. On the first cutoff it is set to `-1.0`, a sentinel meaning “no previous sample.” Numeric timepoints do not produce this days feature, because numeric values could be visit indices rather than days.

### 7.5 Work through the illustrative three-row example

Use the explicitly invented example from Section 5:

| Time | TP53 | KRAS |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |

At the cutoff at time 2, the available history is time 1 and time 2. The **static** inputs are `TP53__current=1`, `KRAS__current=0`, and `current_alteration_count=1`.

Some **temporal** inputs at that cutoff are `observations_so_far=2`; TP53 historical frequency `1/2`; TP53 persistence `1`; TP53 ever-positive `1`; TP53 last-acquired steps `0`; KRAS historical frequency `0`; KRAS persistence `0`; KRAS ever-positive `0`; KRAS last-acquired steps `-1`; `new_acquisition_count=1` (TP53 changed 0 to 1); and `loss_count_since_previous=0`. Historical TP53/KRAS co-occurrence is `0/2=0` because both were never positive together in the first two rows.

The two target labels are **not** the time-2 states. They come from time 3: `target__TP53=1` and `target__KRAS=1`. All numbers here are illustrative only.

## 8. The training and evaluation pipeline

### From CSV to fitted estimators

1. `src/train.py` parses the path, optional `--output`, and optional `--seed` (default 42).
2. `pd.read_csv()` loads the file. The CLI marks only the three metadata columns as metadata; all remaining column names are treated as genes.
3. `run_benchmark()` calls `temporal_features()` and checks that it made transitions for at least four independent patients. This four-patient check is only a software guard. A scientifically adequate study needs a sample-size and power rationale and enough positive/negative transitions.
4. `GroupShuffleSplit` creates one seeded 25% patient holdout. It splits the feature frame by `patient_id` groups.
5. For static and temporal input sets, and for each of LR and RF, the code loops through each gene's `target__GENE` label.
6. A gene/model is skipped if its training labels contain only one class. Otherwise the estimator is fit on training rows, then produces a positive-class probability score for test rows. Scores >= 0.5 become predicted 1; lower scores become 0.
7. It calculates metrics and, for RF, stores impurity-based feature importance values.
8. It writes output files and returns the metrics table to the CLI for printing.

### Files written by a valid run

Under the output directory (default `outputs/`), the evaluator creates:

- `metrics/model_comparison.csv`: per-gene, macro, and where predictions exist, micro results.
- `metrics/random_forest_feature_importance.csv`: RF feature-importance values.
- `metrics/run_metadata.json`: seed, patient split IDs, transition counts, genes, target description, and caveat.
- `predictions/heldout_predictions.csv`: model scores/labels for held-out gene-transition examples.
- `models/*.joblib`: serialized estimator, feature-column names, gene, representation, and seed.

The current repo has no real-run model comparison, prediction, or model artifact. Tests create temporary artifacts only. The generated outputs may contain patient IDs, so do not commit them if they identify private or restricted data. The repository's `.gitignore` excludes the generated model/prediction results.

### Exact conditions for a training CSV

The input must have `patient_id,sample_id,timepoint` plus one or more gene columns. IDs cannot be blank; patient/sample pairs must be unique; each patient must have a unique timepoint. Timepoints must be finite numbers/numeric strings or valid ISO dates/datetimes, consistently parseable across the table, and their order must be validated using source collection records. `T1` labels are not accepted. Gene columns must be unique, complete, and contain only 0/1 or boolean states. No other metadata columns should be included, because `train.py` treats every non-metadata column as a gene. Gene panel/coverage must be comparable and adequate for every included sample; the code has no assay mask and cannot verify this for you. A zero must mean tested and not detected, not “unknown.” At least four independent patients must have at least two ordered observations to pass the current software guard.

Do not give the METABRIC event CSV to `src.train`. It is long-format, its non-metadata columns are not binary gene states, and its `timepoint` is a fixed text label. It is not a training table.

## 9. Understanding the dashboard

The application is `dashboard/app.py`. It sets up the page styling, chooses a data source, tries to load the event-format data, and then displays the selected page. The default source is **Synthetic demo**. The **cBioPortal METABRIC (static)** radio option is added only if `data/processed/brca_metabric_mutations.csv` exists. A session-state upload can override the selected built-in data on the event-data path.

The sidebar contains the ONCOGEN-X name, a data-source radio selector, a page-navigation radio selector, the selected dataset name, a general “Data + evaluation framework” status, and a dataset-specific caveat. Here are the actual pages:

### OVERVIEW

Shows a research-prototype badge, project description, a notice that it is not clinical, and five summary cards: unique patients, unique samples (or timepoint labels if sample IDs are unavailable), unique genes, number of timepoint labels, and event rows. With METABRIC it adds a note that these counts include only samples/patients with mutation records and that missing rows are not confirmed negatives.

The pipeline diagram marks ingestion/exploration as complete and feature/model code as implemented, while cohort validation is still required. The displayed research question explicitly says METABRIC is cross-sectional and the hypothesis is untested. “Models implemented” means software code exists; it does not mean a valid model experiment has been completed.

### GENOMIC DATA

This page supports the **event-table schema**, not the wide training schema. The uploader asks for `patient_id`, `cancer_type`, `timepoint`, `gene`, `alteration`, `alteration_type`, `genomic_status`, and `data_source`. The validator checks presence of these required columns; then the file is loaded and cleaned.

The controls are multiselect filters for patient, cancer type, and timepoint plus a case-insensitive search across columns. The active records table shows rows after filtering. The Data Quality section shows missing counts and percentages per column, required-column validation, how many exact duplicates cleaning removed, and distinct data-source labels. The metrics are descriptive for whatever event data is loaded. An upload is not automatically verified for scientific provenance or valid longitudinal states.

### PATIENT TIMELINE / PATIENT SAMPLES

The page changes its wording based on the data: if there is only one timepoint label and it contains “cross-sectional,” it says **Patient samples** and explains that samples are compared by ID without chronology. Otherwise it says **Patient timeline** and describes observed alterations across the supplied labels.

The patient dropdown selects one patient. Summary cards show patient ID, cancer type, sample-ID count or timepoint-label count, and event-row count. The scatter plot places gene on the y-axis and sample ID (for cross-sectional data) or timepoint (otherwise) on x; color is `genomic_status`, marker symbol is `alteration_type`, and hover text includes alteration details. The heatmap is a cross-tabulation of event-row counts by gene and sample/timepoint. On cross-sectional data an empty cell means no event row in this table, not a confirmed negative. On the synthetic demo, the labels `Acquired`/`Persistent` and `T1`/`T2`/`T3` are part of fictional interface data, not real clinical observations.

### ALTERATION ANALYSIS

Shows charts derived from the currently loaded event rows: top 12 genes by row count; top 12 alteration strings; a pie chart of alteration-type rows; a cancer-type row-count bar chart; and a grouped bar chart by `timepoint` and `genomic_status`. On METABRIC the timepoint has only one cross-sectional category and the status is `Observed alteration`, so that final chart is not a longitudinal trend. Counts should be read as event-row summaries, not patient prevalence.

### TEMPORAL FEATURES

This is a separate uploader for the **wide curated state-table schema**. It treats every column beyond `patient_id`, `sample_id`, and `timepoint` as a gene and calls `temporal_features()`. If there are no adjacent observations it reports that no transitions were found. Otherwise it provides a patient selector, a dataframe of features and target columns for that patient's transitions, a CSV download button, and a note to verify ordering and assay coverage.

This page does not fit a model and does not display predictions. The built-in blank template has headers only. Do not confuse this page with the event-format Genomic Data uploader.

### FUTURE FORECAST

Shows “No verified forecast model,” a conceptual T1-to-T3 pipeline graphic, cards indicating LR/RF code is implemented but data are required, LSTM/GRU are not run, and future forecasts are unavailable. It explicitly says no forecast is displayed and model scores are not clinical risk probabilities. The arrows and T1/T2 labels are an illustration, not a functioning prediction interface.

### MODEL PERFORMANCE

This page reads `outputs/metrics/model_comparison.csv` if that file exists. It displays the metrics table, a macro F1 comparison chart if macro rows exist, a gene/model selector and confusion-matrix heatmap if gene rows exist, a held-out prediction download if the predictions CSV exists, and run metadata JSON if available.

If the metrics file does not exist, it displays “No experimental results” and a message that no valid longitudinal table has been evaluated. If a file does exist, the UI alone does not prove its input was scientifically eligible; inspect its provenance and run metadata before interpreting it.

### EXPLAINABILITY

Reads `outputs/metrics/random_forest_feature_importance.csv` if present. The user can select a target gene and representation, then see a horizontal bar chart of the 20 largest stored RF importance values. If no file exists it says there is no trained RF importance output yet. The chart is not SHAP, not causal biology, and not present for a real longitudinal run at this time.

### RESEARCH ROADMAP

Displays four groups of roadmap cards: implemented items; work blocked on cohort validation; planned work if the data support it; and future directions. It ends with a research-integrity notice that METABRIC supports static exploration only. This page is a project status view, not analysis output.

## 10. Project files and folders

| File or folder | Purpose and connection |
|---|---|
| `README.md` | Setup, download, curated table requirements, training, dashboard, tests, and project caveats. A good first read. |
| `src/download_cbioportal.py` | Retrieves public METABRIC sample/mutation API data, writes raw and processed files, counts, charts, and manifest. Start here to understand real data provenance. |
| `src/data_loader.py` | Required event-table column checks, whitespace trimming, missing-value preservation, exact-duplicate removal, and CSV loading. |
| `src/preprocessing.py` | Dashboard patient/cancer/timepoint/search filtering and missingness summary. |
| `src/temporal_features.py` | Longitudinal input validation, time ordering, feature calculations, next-observation targets, and feature selection. Core file for the research design. |
| `src/train.py` | Command-line entry point for a curated wide state CSV. Infers gene columns and calls the evaluator. |
| `src/evaluation.py` | Patient-held-out split, LR/RF fitting, test predictions, metrics, saved estimators, metadata, and RF importance. |
| `src/__init__.py` | Marks `src` as an importable Python package, enabling `python -m src...`. |
| `dashboard/app.py` | The Streamlit user interface and the nine navigation pages described above. |
| `tests/test_temporal_features.py` | Tests missing/duplicate input states, ordering, date gaps, target construction, demo labeling, and downloader event normalization. |
| `tests/test_evaluation.py` | Runs all four benchmark configurations on a generated synthetic fixture and checks patient separation and saved artifacts. |
| `requirements.txt` | Minimum package versions needed by this project. It is not a lockfile with exact versions. |
| `.venv/` | Local Python environment; ignored by Git and not part of the source project. |
| `.gitignore` | Excludes virtual environments, secrets, caches, raw/processed cohort data, and generated model/prediction files; keeps provenance files. |
| `data/manifest.json` | Provenance for the current downloaded METABRIC data, including retrieval date, counts, and cross-sectional caveat. |
| `data/manifest_template.json` | Template describing expected provenance fields and paths. |
| `data/raw/*.json.gz` | Compressed API response files if present locally. These are ignored by Git. |
| `data/processed/brca_metabric_mutations.csv` | Normalized long event table if present locally; it is not the training CSV and is ignored by Git. |
| `data/sample/demo_genomic_data.csv` | Fictional event-format records for dashboard demonstration, labeled synthetic. |
| `data/sample/longitudinal_states_template.csv` | Blank header-only wide state-table template; it is not data. |
| `outputs/metrics/dataset_summary.json` | Aggregate counts and cross-sectional note generated by the downloader. |
| `outputs/figures/*.html` | Interactive gene-frequency and mutation-class charts generated from event rows, if present locally. |
| `outputs/metrics/model_comparison.csv`, `outputs/predictions/`, `outputs/models/` | Expected outputs from a benchmark run. No real experiment files currently exist; generated/sensitive outputs are ignored. |
| `docs/dataset.md` | Data provenance and exact curated table requirements. |
| `docs/research_report.md` | Research question, design, data results, limitations, and reproducibility report. |
| `literature/literature_review.md` | Annotated background references with DOI links. |
| `screenshots/README.md` | Recommended dashboard captures and cautions. No screenshots are present. |
| `output/pdf/ONCOGEN-X_research_report.pdf` | Generated research-report PDF from the prior submission task; separate from this learning guide. |

Suggested first files to open: `README.md`, this guide's dataset section, `src/download_cbioportal.py`, `src/data_loader.py`, `dashboard/app.py`, `src/temporal_features.py`, `src/evaluation.py`, and then the tests.

## 11. What happens when I run the commands?

Run these in Windows PowerShell from `C:\Users\Ishita\Desktop\cancer-genomic-forecasting`.

### `python -m src.download_cbioportal`

Python runs the `__main__` block in `src/download_cbioportal.py`, which calls `download()` and prints the processed CSV path. It needs a working network connection to cBioPortal and the dependencies in `requirements.txt`. It creates or overwrites the two compressed raw API files, the processed event CSV, two interactive HTML figures, `outputs/metrics/dataset_summary.json`, and `data/manifest.json`.

It downloads **cross-sectional descriptive data**, not a longitudinal training cohort. Re-running it contacts the live service and may change the local files/counts.

### `python -m src.train ...`

Example command:

```powershell
python -m src.train path\to\curated_longitudinal_states.csv --output outputs --seed 42
```

The command reads the CSV path supplied as its first argument. Optional `--output` sets where results go; optional `--seed` fixes the random split/model seeds, defaulting to 42. It requires a properly curated wide table with real ordered repeated observations, one row per patient/sample, and complete comparable binary gene states. It creates the model, metric, metadata, and prediction output folders/files if the run gets through validation. A bad input raises a validation error or produces no eligible transitions; four patients is only a minimum code check, not evidence of a sufficient scientific sample.

Do not pass `data/processed/brca_metabric_mutations.csv` or the demo event CSV. They are not wide training state tables.

### `streamlit run dashboard/app.py`

This launches a local Streamlit server and opens the UI in a browser. It reads the synthetic demo by default. The cBioPortal METABRIC radio option appears only if its processed CSV exists. The app can display descriptive tables/charts, accept the event-table upload, and separately show features created from a curated-state upload. It does not call `run_benchmark()` or train models. Model/importance pages only read output files created by a separate CLI run.

The module form also works and is explicit about the active interpreter:

```powershell
python -m streamlit run dashboard/app.py
```

### `python -m pytest -q`

pytest finds and runs tests under `tests/`. The tests exercise validation and execution using synthetic fixtures. The benchmark test fits models on its generated temporary states and checks output/separation behavior. The suite does not download data, run the dashboard in a browser, validate a real assay panel, or perform a real longitudinal experiment.

For the virtual environment used in this workspace:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 12. What has genuinely been achieved and what remains

### Implemented and verified as software

- METABRIC API download and event normalization code exists; the local manifest records a successful download and zero unmatched mutation joins for that retrieval.
- Event-format data validation/cleaning and dashboard exploration are implemented.
- The state-table validator, temporal feature construction, and next-observation target logic have focused tests.
- Patient-grouped LR/RF code is implemented for static and temporal representations.
- The synthetic benchmark test verifies that all four configurations can execute, that training/test patient IDs are disjoint, and that prediction/model artifacts are produced in a temporary directory.
- The last recorded test run in this workspace passed 12 tests using the project's virtual environment. It was not rerun while writing this guide, because the benchmark test itself fits models on synthetic fixture data.

### Supported by real-data descriptive analysis

The downloaded METABRIC records support the event-row/sample/gene counts and the gene/class summaries in Section 5. Those are descriptions of the current downloaded file only. They are not mutation prevalence in all study participants, negative test rates, temporal trends, or forecasts.

### Tested only with synthetic data

The shipped dashboard demo is fictional. The unit-test state rows and the benchmark's 16-patient fixture are generated synthetic data. They test application behavior and model-fit code paths, not cancer biology or generalization.

### Not yet demonstrated

- A real longitudinal training dataset with valid source-backed timing and assay coverage.
- A real longitudinal LR/RF benchmark or any real model metrics.
- A real next-observation forecast or clinical utility.
- LSTM/GRU models, SHAP explanations, confidence intervals, repeated validation, or external validation.

### What would be required before making a forecasting claim

Obtain an appropriately licensed serial-sample cohort. Verify patient identity and actual collection order from source records, not sample labels. For every included gene/sample, establish whether it was adequately assayed; use a measured 0 only when the assay could have detected the alteration and did not. Harmonize panel/coverage and define eligibility before looking at test outcomes. Then split by patient, predefine metrics and model selection, run the experiment, inspect errors and uncertainty, and validate on independent data. A predictive result would still not make this a clinical decision system.

## 13. One worked example from beginning to end

Everything in this section is **illustrative**. Patient IDs, gene states, scores, and outcomes below are made up for teaching and are not project or METABRIC results.

### Step 1: Two illustrative patients, three samples each

Assume a curator has verified the numeric time order and measured both genes at all six observations.

| Illustrative patient | Time | TP53 | KRAS |
|---|---:|---:|---:|
| A | 1 | 0 | 0 |
| A | 2 | 1 | 0 |
| A | 3 | 1 | 1 |
| B | 1 | 0 | 1 |
| B | 2 | 0 | 1 |
| B | 3 | 1 | 1 |

Each patient produces two transitions: time 1 to 2 and time 2 to 3. For patient A at cutoff time 2, current-state features are TP53=1, KRAS=0, burden=1. The labels for that row come from time 3: TP53=1 and KRAS=1. The earlier history adds values like TP53 frequency 0.5 and one recent acquisition.

### Step 2: Training examples are cutoff rows, not raw mutation events

Across many patients, every row with a next observed sample becomes one supervised example. The input is the current-only vector for the static model or current-plus-history vector for the temporal model. The target for a TP53 model is only the next TP53 value; the target for a KRAS model is only the next KRAS value.

### Step 3: Keep patients apart

The group split assigns all of patient A's transitions to one side and all of patient B's transitions to the other side. In a real analysis there need to be many more independent patients; the two-patient table here is too small even to pass the code's four-patient minimum.

### Step 4: Fit and score (illustrative only)

Suppose, purely for illustration, a trained KRAS classifier gives the held-out time-2 row score `0.62` and the observed next KRAS state is `1`. At the code's 0.5 threshold it predicts 1 and that row is a true positive. This example score was invented for explanation; ONCOGEN-X has not produced it. Real evaluation would repeat across held-out transition rows and report the defined metrics, while keeping test patients out of model fitting.

## 14. Glossary

| Term | Plain-language meaning |
|---|---|
| **Genomic alteration** | A difference in DNA sequence or gene copy state. It is a description of a measured change, not automatically a cause or diagnosis. |
| **Mutation** | A DNA sequence change. The project download focuses on somatic mutation records. |
| **Gene** | A DNA region associated with biological instructions; in the event table it is represented by a symbol such as TP53. |
| **Patient** | The person-level grouping ID. It keeps that person's transitions together during evaluation. |
| **Sample** | A specimen/measurement from a patient. A sample ID is not a timestamp. |
| **Timepoint** | The recorded date or order of sample collection. The METABRIC event file's timepoint is only a fixed label. |
| **Longitudinal** | Repeated observations of the same subject ordered over time. |
| **Cross-sectional** | Observations representing a cohort snapshot rather than a verified within-patient time sequence. |
| **Feature** | A predictor/input value supplied to a model. |
| **Target / label** | The known output the model is trained to predict. Here it is a gene's next-observation 0/1 state. |
| **Classifier** | A model that predicts a category, here 0 or 1. |
| **Training** | Fitting a model's internal parameters using examples with known labels. |
| **Testing** | Evaluating predictions on held-out examples that were not used to fit the model. |
| **Overfitting** | Learning quirks of the training examples so closely that performance drops on new patients. |
| **Data leakage** | Letting information unavailable at prediction time influence model inputs or evaluation. |
| **Precision** | Among predicted positives, the fraction that were actually positive. |
| **Recall** | Among actual positives, the fraction the model correctly found. |
| **F1-score** | A single score combining precision and recall; high only when both are reasonably high. |
| **ROC-AUC** | A ranking measure of positive vs negative scores over possible thresholds; this code needs both test classes to calculate it. |
| **PR-AUC / average precision** | A precision-recall summary focused on positive labels; useful to inspect for rare positives, with prevalence affecting its baseline. |
| **API** | A defined software interface through which one program requests data/services from another. |
| **Preprocessing** | Organizing/cleaning data before analysis. Here it includes trimming strings, dropping exact event duplicates, validating states, and creating features. |
| **Assay coverage** | How much of the relevant genomic material was measured and how reliably a test could detect an alteration. A missing/uncovered gene is unknown, not a negative. |
| **Event table** | A long table where one row records one reported alteration event. |
| **Wide state table** | A table with one patient/sample per row and one binary column per gene. |
| **Patient-grouped split** | A train/test division that assigns whole patients to one side to avoid sharing a person's rows across both sets. |

## 15. A learning path for this repository

### 1. Start with the data story

Read `README.md`, `docs/dataset.md`, then `data/manifest.json` and `outputs/metrics/dataset_summary.json`.

Ask: **Why are 13,181 event rows not 13,181 tested samples?**  
Check: Because rows represent reported events, and one sample can have multiple events while samples with no event may not appear in the event table.

Ask: **Why doesn't a missing event row mean a confirmed negative?**  
Check: The event table doesn't record that a gene was adequately assayed and found absent for every sample.

### 2. Learn the long event table before machine learning

Open a few rows of `data/sample/demo_genomic_data.csv`, then inspect `src/data_loader.py` and `src/preprocessing.py`. The demo is synthetic. Notice that the dashboard's event schema has columns for gene, alteration, and status, while the training schema is a wide matrix.

Ask: **Does `validate_genomic_data()` verify biological correctness?**  
Check: No. It checks only whether required columns are present. It does not verify study provenance or assay coverage.

### 3. Follow one API record

Read `download()` and `_mutation_events()` in `src/download_cbioportal.py`. Follow how `sampleId` maps to `patientId`, and where the fixed cross-sectional timepoint is assigned.

Ask: **Where does a collection date get assigned?**  
Check: Nowhere. The downloader explicitly uses a constant cross-sectional label.

### 4. Learn the temporal builder using the toy rows

Read `_ordered_timepoints()`, then `validate_states()`, then `temporal_features()`, then `feature_columns()` in that order. Compare their behavior with `tests/test_temporal_features.py`.

Ask: **Which row supplies `target__TP53` for a cutoff at time 2?**  
Check: The next ordered sample, time 3. It is excluded from the input features.

Ask: **What does `GENE__ever_observed` mean in this code?**  
Check: The gene was positive at least once in the included history, not that the gene was assayed.

### 5. Understand the split before the algorithms

Trace `run_benchmark()` in `src/evaluation.py`. Find the `GroupShuffleSplit`, the feature selection, the classifier `.fit()`, and where predictions and metrics are written. Then read `tests/test_evaluation.py` to see what the synthetic fixture actually establishes.

Ask: **Can the same patient appear in both train and test?**  
Check: Not with this group split; patient IDs are split as groups.

Ask: **Does the test prove model performance on cancer data?**  
Check: No. It proves execution on constructed binary states only.

### 6. Explore the dashboard page by page

Run `python -m streamlit run dashboard/app.py`. First keep the Synthetic demo selection and inspect Overview, Genomic Data, Patient Samples/Timeline, Alteration Analysis, Future Forecast, Model Performance, Explainability, and Research Roadmap. The cBioPortal option appears after the processed file exists. A separate uploader under Temporal Features accepts the wide state-table format.

Ask: **Does clicking Future Forecast run a prediction?**  
Check: No. It displays a status explanation and conceptual diagram.

Ask: **Does the Model Performance page train models?**  
Check: No. It reads result files created by the separate training command.

### 7. Use safe experiments

- Run `.\.venv\Scripts\python.exe -m pytest -q`. The test uses synthetic records and writes benchmark outputs only to a temporary directory.
- Inspect the blank `data/sample/longitudinal_states_template.csv` to see the required columns, but do not treat its header as data.
- Do not run the training command on the METABRIC event CSV or on fictional demo events. Do not convert missing event rows to zeros.
- If experimenting with changes to features, use a clearly synthetic fixture in a temporary location and label any outputs as software demonstrations. Do not describe those results as research findings.
- Before a real run, document where the sample order and assay coverage came from and ensure all required states are measured.

Ask: **What evidence would I need before writing “the model forecasts future alterations”?**  
Check: A valid longitudinal cohort, verified order and assay coverage, patient-separated evaluation, reported uncertainty, and ideally independent validation. A passing software test is not enough.
