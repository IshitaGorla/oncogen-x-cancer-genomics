# Dashboard screenshots to capture

No screenshots have been captured or are included yet. This folder is reserved for reviewed dashboard captures; do not add screenshots that imply an experiment or real longitudinal result that has not occurred.

Recommended captures after launching `python -m streamlit run dashboard/app.py`:

1. **Overview** — capture the research question, prototype notice, pipeline status, and the statement that METABRIC is cross-sectional and the hypothesis remains untested. Use the Synthetic demo source only if its fictional status is visible in the sidebar.
2. **Genomic Data — METABRIC** — after running the downloader, capture the selected `cBioPortal METABRIC (static)` source, active event records, and data-quality/provenance context. Keep the caveat that absent event rows are not negatives visible.
3. **Patient Samples — METABRIC** — show sample IDs as categories and the cross-sectional explanation. Do not caption samples as a timeline or interpret an empty matrix cell as a measured negative.
4. **Alteration Analysis — METABRIC** — capture gene event-row counts and mutation-class summaries. Caption them as descriptive event counts, not patient prevalence or forecasts.
5. **Future Forecast** — capture the “No verified forecast model” status and explanation that no eligible cohort or trained model is available.
6. **Model Performance** — capture the no-results state while no valid longitudinal run exists. If a later run populates this page, capture only after its input provenance and eligibility have been reviewed; label the cohort and split clearly.
7. **Explainability** — capture the empty state while no trained RF importance output exists. Only capture feature-importance charts after a reviewed eligible model run, with an explicit noncausal interpretation note.
8. **Research Roadmap** — capture implemented components, cohort-dependent work, and future work.

The Temporal Feature Inspector accepts a wide state table. Do not create or upload an invented trajectory to make this page look populated. If demonstrating it with a software fixture in a future presentation, label the screenshot prominently as synthetic software testing and keep it out of scientific results.

Suggested filenames: `overview.png`, `genomic-data-metabric.png`, `patient-samples-metabric.png`, `alteration-analysis-metabric.png`, `future-forecast-status.png`, `model-performance-empty.png`, `explainability-empty.png`, and `research-roadmap.png`. Store any captures here only after verifying what dataset and page state each image shows.
