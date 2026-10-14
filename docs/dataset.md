# Dataset provenance and preparation

## Public cohort included in the download pipeline

- **Study:** Breast Cancer (METABRIC, Nature 2012 & Nat Commun 2016), cBioPortal study identifier `brca_metabric`.
- **Landing page:** https://www.cbioportal.org/study/summary?id=brca_metabric
- **Study archive:** https://datahub.assets.cbioportal.org/brca_metabric.tar.gz (not used by this downloader; it retrieves selected API resources only)
- **Publication details:** Curtis et al., *Nature* 2012, 486:346–352, [doi:10.1038/nature10983](https://doi.org/10.1038/nature10983); Pereira et al., *Nature Communications* 2016, 7:11479, [doi:10.1038/ncomms11479](https://doi.org/10.1038/ncomms11479).
- **General download and API documentation:** https://docs.cbioportal.org/downloads/ and https://docs.cbioportal.org/web-api-and-clients/
- **Access date:** recorded in `data/manifest.json` when download succeeds.
- **Interpretation in this project:** mutation events and clinical sample/patient identifiers support static descriptive exploration. The script explicitly labels all rows as a single cross-sectional sample cohort. It does not assert that multiple sample IDs per patient are serial timepoints.

The downloader uses the documented cBioPortal REST API and fetches only the study's sample list and somatic mutation profile, avoiding a whole-study archive of unrelated assays. It stores compressed responses as `data/raw/brca_metabric_samples.json.gz` and `data/raw/brca_metabric_mutations.json.gz`, then writes normalized event rows to `data/processed/brca_metabric_mutations.csv`. It maps API `gene.hugoGeneSymbol` to `gene`, `proteinChange` (falling back to `mutationStatus` or `Not reported`) to `alteration`, and `mutationType` to `alteration_type`. Raw and processed cohort files are excluded from Git; the manifest and source documentation are small and can be tracked. The project should be reviewed against the cited publication and data terms before redistribution. No controlled-access data or credentials are included.

The same command creates `outputs/metrics/dataset_summary.json` and two interactive Plotly figures in `outputs/figures/`. These are descriptive event-row summaries, not longitudinal results.

## Processed event dictionary

| Column | Meaning |
|---|---|
| `patient_id` | cBioPortal patient identifier joined from the study sample list |
| `sample_id` | Sample identifier attached to the mutation record |
| `cancer_type` | Broad label set by the preparation script: breast cancer |
| `timepoint` | Constant explicit cross-sectional label, not a date or ordering |
| `gene` | Gene symbol from the cBioPortal API gene object |
| `alteration` | Protein change when available, with API fallback fields |
| `alteration_type` | Mutation classification from the API when available |
| `genomic_status` | `Observed alteration`; absence of a row is not a confirmed negative |
| `data_source` | Study identifier and cross-sectional caveat |

The study sample list contains 2,509 samples; the downloaded mutation profile returned 13,183 events, with 13,181 unique normalized event rows and no unmatched patient joins at access time. The event CSV includes samples represented by mutation records only (2,358 patients and 2,358 sample IDs), so these are not cohort-wide patient/sample counts. Duplicate event rows are dropped. The script does not impute negative states or collection dates.

## Forecast input requirements

Forecasting is implemented only for a separately curated complete state matrix. Its exact CSV schema is `patient_id,sample_id,timepoint,<gene1>,...`; it must contain at least one gene column and no other metadata columns. Every row is one serially collected patient/sample. Patient/sample pairs and within-patient timepoints must be unique. Timepoints must be consistently represented as finite numeric values (their numeric order must reflect collection order) or valid ISO dates/datetimes. Arbitrary labels such as `T1`, `baseline`, or sample names are not accepted as evidence of chronology. The curator must independently verify that the ordering represents genuine collection time; syntax alone cannot establish provenance.

Every gene cell must be an observed binary value. A `0` means the assay could detect that gene in that sample and found no qualifying alteration; a `1` means an alteration was detected. Unknown, unsequenced, low-coverage, and not-reported states must never be coded as 0. This implementation has no assay mask and cannot independently verify assay coverage or the meaning of a zero. Curators must restrict the input to genes with comparable, adequate coverage across every included sample (or first design a different mask-aware method). Missing cells are rejected, not imputed. At least four independent patients with at least two observations are needed to run the grouped benchmark; this is only a software minimum, not a sample-size justification.

The cBioPortal METABRIC download described above does not meet this schema or these requirements: it is a long event table, contains observed mutation rows only, and has a constant cross-sectional timepoint label. It cannot be used to form trajectories or confirmed negative states.

The included `data/sample/longitudinal_states_template.csv` has headers only and contains no patient data. `data/sample/demo_genomic_data.csv` is synthetic and exists only for interface demonstration.
