# Literature review: longitudinal cancer genomics and alteration forecasting

**Scope.** This focused narrative review supports the experimental design; it is not a systematic review or evidence that the proposed question is unprecedented. The 14 DOI records below were queried through the Crossref works API on 11 October 2026; returned bibliographic metadata were checked against the linked publisher DOI destinations. Research papers reconstruct tumour evolution or describe serial sampling, but those aims differ from prospective next-observation prediction.

## Cancer genomics and data resources

The Cancer Genome Atlas (TCGA) established a large multi-platform characterization program across tumour types. TCGA's broad molecular profiles are valuable for cross-sectional genomic discovery, but a large number of patients does not imply repeated sampling of each patient's cancer. The NCI Genomic Data Commons (GDC) standardizes access to genomic and clinical data, with controlled-access constraints for some materials. cBioPortal offers interactive exploration, download files, and a REST API; the portal's availability of a mutation profile alone does not establish sample chronology or assay coverage.

## Heterogeneity, clonal evolution, and longitudinal evidence

Nowell's clonal evolution model describes selection and expansion of heritable tumour-cell variants. Modern sequencing has refined this view: a biopsy samples only a portion of a spatially heterogeneous tumour, and detection depends on assay sensitivity and purity. Yates et al. used multi-region sequencing to show subclonal diversification in primary breast cancers; different regions are spatial samples and should not automatically be treated as sequential timepoints. TRACERx prospectively studies NSCLC evolution and intratumour heterogeneity, while TRACERx ctDNA work used tumour-informed phylogenetic markers to track early-stage lung cancer. These studies supply biological rationale and careful sampling designs; they do not provide a plug-and-play, broadly sized patient-by-timepoint binary matrix for this prototype.

Gerstung et al. reconstructed evolutionary histories from whole-genome data for 2,658 cancers across 38 types. Their work provides evidence that mutation timing and driver evolution can be inferred from genomic patterns. It is a retrospective evolutionary reconstruction, not a direct benchmark of static versus temporal machine-learning features on serial patient samples. The distinction between inferential reconstruction and prospective forecasting is central to ONCOGEN-X.

## Machine learning, interpretation, and the research gap

Machine learning in cancer genomics spans molecular subtype prediction, outcome prediction, driver discovery, and evolutionary inference. A credible prediction study must define its prediction time, distinguish assay non-observation from a measured negative, keep patients separated across training and evaluation, and report class imbalance and uncertainty. Feature attribution can describe what a fitted model used; it does not establish biological causality.

The focused sources reviewed here demonstrate rich approaches to evolutionary reconstruction and serial monitoring, but they do not answer the exact educational benchmark proposed here: on verified, patient-separated next-sample data with known assay coverage, do prespecified persistence, acquisition-history, and co-occurrence features improve over current genomic state alone? This is a narrow project question, not a claim of a field-wide or unprecedented gap. The included METABRIC cBioPortal cohort is cross-sectional for this project and cannot test it.

## Research questions and hypotheses

**RQ1:** On a cohort with genuinely ordered repeat samples and comparable, documented assay coverage, does adding past-only trajectory features improve next-observation alteration-state prediction over current binary state alone?

**H1:** Temporal features may improve performance, measured with patient-held-out per-gene PR-AUC and macro F1; the null outcome (no consistent improvement) is scientifically valid.

**RQ2:** How does performance vary across genes and, only if enough serial observations exist, across elapsed prediction intervals?

**H2:** Performance and uncertainty will vary with prevalence, sequencing interval, and sample availability. This is descriptive and does not imply clinical risk.

## Proposed methodology and limitations

Construct transitions only from samples with an independently justified temporal order. Require an explicit, per-sample indication that each target gene was assayed before treating a 0 as a confirmed negative. Exclude unknown states rather than imputing them to absence. Compute temporal covariates using observations at or before the cutoff. Compare logistic regression and random forest with the same grouped patient split and features available to each representation; report per-gene confusion matrices, accuracy, precision, recall, F1, ROC-AUC where both test classes exist, and PR-AUC. Report independent patient counts and avoid a deep model unless sufficient independent trajectories exist.

The current pipeline implements this guarded interface for **complete binary state tables supplied by a user**. It does not establish that a public cross-sectional mutation table qualifies. Dates, treatment, assay panels, tumour sites, and sequencing depth can all affect observed changes. A measured status change can reflect sampling or assay sensitivity rather than clonal acquisition or loss. A small cohort also produces unstable estimates. No clinical risk or causal interpretation is intended.

## Annotated references

1. **Nowell PC. “The clonal evolution of tumor cell populations.” Science. 1976;194(4260):23–28.** DOI: [10.1126/science.959840](https://doi.org/10.1126/science.959840). Foundational model of clonal selection in tumour populations. It is conceptual and predates sequencing; relevance is the evolutionary framing, not a forecasting method.

2. **The Cancer Genome Atlas Network. “Comprehensive molecular portraits of human breast tumours.” Nature. 2012;490(7418):61–70.** DOI: [10.1038/nature11412](https://doi.org/10.1038/nature11412). Integrated molecular characterization of breast tumours. It is primarily a cross-sectional molecular portrait and illustrates why large molecular cohorts do not automatically provide serial timelines.

3. **Cerami E, Gao J, Dogrusoz U, et al. “The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data.” Cancer Discovery. 2012;2(5):401–404.** DOI: [10.1158/2159-8290.CD-12-0095](https://doi.org/10.1158/2159-8290.CD-12-0095). Introduces cBioPortal's exploration platform. It is a data access and analysis resource, not a temporal cohort validation study.

4. **Gao J, Aksoy BA, Dogrusoz U, et al. “Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal.” Science Signaling. 2013;6(269):pl1.** DOI: [10.1126/scisignal.2004088](https://doi.org/10.1126/scisignal.2004088). Describes integrated portal analyses; supports reproducible access, while chronology still depends on the underlying study metadata.

5. **Yates LR, Gerstung M, Knappskog S, et al. “Subclonal diversification of primary breast cancer revealed by multiregion sequencing.” Nature Medicine. 2015;21:751–759.** DOI: [10.1038/nm.3886](https://doi.org/10.1038/nm.3886). Uses spatially separated tumour regions to characterize subclonal diversity. Its spatial design is biologically informative but must not be mistaken for a serial timeline.

6. **Jamal-Hanjani M, Wilson GA, McGranahan N, et al. “Tracking the Evolution of Non-Small-Cell Lung Cancer.” New England Journal of Medicine. 2017;376:2109–2121.** DOI: [10.1056/NEJMoa1616288](https://doi.org/10.1056/NEJMoa1616288). TRACERx prospectively studies intratumour heterogeneity and NSCLC evolution. Rich multiregion sampling motivates temporal research; its design and biological objectives differ from a generic next-sample classifier.

7. **Abbosh C, Birkbak NJ, Wilson GA, et al. “Phylogenetic ctDNA analysis depicts early-stage lung cancer evolution.” Nature. 2017;545:446–451.** DOI: [10.1038/nature22364](https://doi.org/10.1038/nature22364). Applies tumour-informed phylogenetic ctDNA analysis in TRACERx participants. It demonstrates longitudinal monitoring with specialized assays, not a directly comparable ML benchmark.

8. **Gerstung M, Jolly C, Leshchiner I, et al. “The evolutionary history of 2,658 cancers.” Nature. 2020;578:122–128.** DOI: [10.1038/s41586-019-1907-7](https://doi.org/10.1038/s41586-019-1907-7). Reconstructs mutational processes and driver timing across 38 cancer types using WGS. This evolutionary reconstruction gives strong context but does not validate next-timepoint prediction from serial samples.

9. **Grossman RL, Heath AP, Ferretti V, et al. “Toward a Shared Vision for Cancer Genomic Data.” New England Journal of Medicine. 2016;375:1109–1112.** DOI: [10.1056/NEJMp1607591](https://doi.org/10.1056/NEJMp1607591). Presents the GDC vision for harmonized cancer genomic data access. Data harmonization helps reproducibility; access rules and metadata still constrain analysis.

10. **Jamal-Hanjani M, Hackshaw A, Ngai Y, et al. “Tracking genomic cancer evolution for precision medicine: the lung TRACERx study.” PLoS Biology. 2014;12(7):e1001906.** DOI: [10.1371/journal.pbio.1001906](https://doi.org/10.1371/journal.pbio.1001906). Describes the prospective TRACERx study rationale and design. It emphasizes prospective evolution-aware collection, while not reporting the project’s proposed classifier comparison.

11. **McGranahan N, Swanton C. “Clonal heterogeneity and tumor evolution: past, present, and the future.” Cell. 2017;168(4):613–628.** DOI: [10.1016/j.cell.2017.01.018](https://doi.org/10.1016/j.cell.2017.01.018). Review of tumour heterogeneity and evolutionary methods. Review-level context; it highlights sampling limitations relevant to interpreting observed gene-state changes.

12. **TRACERx Consortium. “The evolution of lung cancer and impact of subclonal selection in TRACERx.” Nature. 2023;616:525–533.** DOI: [10.1038/s41586-023-05783-5](https://doi.org/10.1038/s41586-023-05783-5). Reports evolutionary and subclonal selection findings from TRACERx. It underscores complex evolution; it is not an externally validated general-purpose forecast model.

13. **Curtis C, Shah SP, Chin SF, et al. “The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups.” Nature. 2012;486:346–352.** DOI: [10.1038/nature10983](https://doi.org/10.1038/nature10983). METABRIC molecular profiling established breast cancer subgroups in a large cohort; its value here is contextual, cross-sectional analysis rather than a serial forecast cohort.

14. **Pereira B, Chin SF, Rueda OM, et al. “The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes.” Nature Communications. 2016;7:11479.** DOI: [10.1038/ncomms11479](https://doi.org/10.1038/ncomms11479). Integrates somatic mutation profiles with breast tumour molecular/clinical characteristics. This is relevant to the selected METABRIC mutation study but does not establish time-ordered repeat observations for our intended prediction task.

## Verified data documentation

- cBioPortal downloads: [official documentation](https://docs.cbioportal.org/downloads/) and [REST API documentation](https://docs.cbioportal.org/web-api-and-clients/).
- METABRIC study page: [cBioPortal BRCA METABRIC](https://www.cbioportal.org/study/summary?id=brca_metabric). Its mutation file is used only for static exploration in this repository.
- NCI GDC: [portal](https://portal.gdc.cancer.gov/) and [documentation](https://docs.gdc.cancer.gov/).
