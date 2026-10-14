# ONCOGEN-X — Complete Source Code

Copy-ready snapshot of the project code and setup files. Each section title names the destination file; copy the complete fenced block into that file.

This document includes all Python application, library, and test files, plus `requirements.txt` and `.gitignore`. Dataset CSVs, downloaded METABRIC records, generated metrics, model artifacts, caches, and virtual environments are intentionally excluded; they are data or generated files, not source code.

## Contents

1. `.gitignore`
2. `requirements.txt`
3. `dashboard/app.py`
4. `src/__init__.py`
5. `src/data_loader.py`
6. `src/download_cbioportal.py`
7. `src/evaluation.py`
8. `src/molecular_drift.py`
9. `src/preprocessing.py`
10. `src/temporal_features.py`
11. `src/train.py`
12. `tests/test_evaluation.py`
13. `tests/test_molecular_drift.py`
14. `tests/test_temporal_features.py`

## Part 1 — `.gitignore`

```text
.venv/
venv/
env/
.env
.env.*
!.env.example
.streamlit/secrets.toml
*.pem
*.key
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.log
outputs/figures/*
!outputs/figures/.gitkeep
outputs/models/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
outputs/predictions/*
!outputs/predictions/.gitkeep
outputs/metrics/*
!outputs/metrics/.gitkeep
!outputs/metrics/dataset_summary.json
```

## Part 2 — `requirements.txt`

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
streamlit>=1.32
plotly>=5.18
requests>=2.31
pytest>=8.0
```

## Part 3 — `dashboard/app.py`

```python
from pathlib import Path
import sys
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_genomic_data, validate_genomic_data
from src.preprocessing import filter_genomic_data, missing_value_summary
from src.temporal_features import temporal_features
from src.molecular_drift import analyze_patient, compare_demo_rules, make_demo_data

st.set_page_config(page_title="ONCOGEN-X | Longitudinal Genomics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Manrope',sans-serif;}
.stApp {background:#f6f8fb; color:#14233a;}
section[data-testid="stSidebar"] {background:#091a30; border-right:1px solid #173451;}
section[data-testid="stSidebar"] * {color:#eaf2fb !important;}
.brand {font-size:1.75rem;font-weight:800;letter-spacing:.06em;color:#0a2946;margin:0;}
.eyebrow {font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.13em;color:#16879c;font-weight:600;text-transform:uppercase;}
.page-title {font-size:2rem;font-weight:800;line-height:1.18;color:#0a2946;margin:.25rem 0 .65rem;}
.subtle {color:#52657c;font-size:1rem;line-height:1.65;max-width:850px;}
.metric-card {background:#fff;border:1px solid #dce6ef;border-radius:12px;padding:18px 20px;box-shadow:0 3px 12px rgba(20,46,77,.05);min-height:105px;}
.metric-label {font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.08em;color:#607289;text-transform:uppercase;}.metric-value{font-size:1.9rem;font-weight:800;color:#0c3153;margin-top:5px;}
.notice {background:#eaf7f8;border-left:4px solid #1492a6;padding:14px 18px;border-radius:0 8px 8px 0;color:#23485c;}
.status {display:inline-block;background:#e5f5ed;color:#147146;border:1px solid #bce6cf;border-radius:20px;padding:4px 10px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.06em;font-weight:600;}
.planned {display:inline-block;background:#fff4db;color:#8a5c00;border:1px solid #f1d89b;border-radius:20px;padding:4px 10px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.06em;font-weight:600;}
.pipeline {display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:25px 0;}.pipe-step{background:#fff;border:1px solid #dbe5ee;border-radius:8px;padding:13px 15px;font-size:.72rem;font-family:'DM Mono',monospace;font-weight:600;color:#63758c;}.pipe-step.active{background:#0c3557;color:#fff;border-color:#0c3557;}.arrow{color:#16a1b4;font-size:1.3rem;}
.road-card{background:#fff;border:1px solid #dde6ee;border-radius:10px;padding:16px 20px;margin:10px 0;}.road-card.current{border-left:5px solid #1295a5;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

DEFAULT_DATA = ROOT / "data" / "sample" / "demo_genomic_data.csv"
PUBLIC_DATA = ROOT / "data" / "processed" / "brca_metabric_mutations.csv"
SYNTHETIC_BENCHMARK = ROOT / "outputs" / "synthetic_demo"
DEFAULT_BENCHMARK = ROOT / "outputs"

def available_benchmark_outputs():
    """Return separately selectable saved runs; never infer provenance from METABRIC."""
    sources = {}
    for label, folder in [
        ("Synthetic demonstration run · outputs/synthetic_demo", SYNTHETIC_BENCHMARK),
        ("Existing outputs · dataset source not recorded", DEFAULT_BENCHMARK),
    ]:
        has_metrics = (folder / "metrics" / "model_comparison.csv").is_file()
        has_importance = (folder / "metrics" / "random_forest_feature_importance.csv").is_file()
        if has_metrics or has_importance:
            sources[label] = folder
    return sources

def select_benchmark_output():
    sources = available_benchmark_outputs()
    if not sources:
        return None, None
    label = st.selectbox("Saved benchmark output (independent of the exploration dataset)", list(sources), key="benchmark_output_choice")
    folder = sources[label]
    synthetic = folder == SYNTHETIC_BENCHMARK
    if dataset.startswith("cBioPortal"):
        st.warning("Benchmark results are kept separate from METABRIC static records. Switch Data source to Synthetic demo before viewing any benchmark output.")
        return None, "synthetic" if synthetic else "unverified"
    if synthetic:
        st.warning("SYNTHETIC DEMONSTRATION DATA — saved benchmark outputs from the synthetic longitudinal demo. These are software demonstration results, not real-patient evidence or validated forecasts.")
        st.caption("Read-only dashboard display from outputs/synthetic_demo; opening this page does not train or update models. The saved run metadata records the seed and patient split, but not the input filename.")
        return folder, "synthetic"
    st.warning("This saved output does not identify its input dataset in the run metadata. Its provenance is unverified; do not interpret it as real-patient results without checking the source data.")
    return folder, "unverified"

@st.cache_data(show_spinner=False)
def read_default(): return load_genomic_data(DEFAULT_DATA)

@st.cache_data(show_spinner=False)
def read_public(): return load_genomic_data(PUBLIC_DATA)

def load_active_data():
    upload = st.session_state.get("uploaded_file")
    if upload is not None:
        upload.seek(0)
        return load_genomic_data(upload)
    if st.session_state.get("dataset_choice", "Synthetic demo").startswith("cBioPortal") and PUBLIC_DATA.exists():
        return read_public()
    return read_default()

def cards(items):
    columns = st.columns(len(items))
    for col, (label, value) in zip(columns, items):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

def heading(label, title, text=""):
    st.markdown(f'<div class="eyebrow">{label}</div><div class="page-title">{title}</div>', unsafe_allow_html=True)
    if text: st.markdown(f'<div class="subtle">{text}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand" style="color:white">ONCOGEN-X</div><div class="eyebrow" style="color:#77d8e0">Longitudinal Genomic Intelligence</div>', unsafe_allow_html=True)
    st.divider()
    dataset_options = ["Synthetic demo"] + (["cBioPortal METABRIC (static)"] if PUBLIC_DATA.exists() else [])
    dataset = st.radio("Data source", dataset_options, key="dataset_choice", label_visibility="collapsed")
    page = st.radio("Navigate", ["OVERVIEW", "GENOMIC DATA", "PATIENT TIMELINE", "ALTERATION ANALYSIS", "TEMPORAL FEATURES", "FUTURE FORECAST", "MODEL PERFORMANCE", "EXPLAINABILITY", "RESEARCH ROADMAP", "MOLECULAR DRIFT RESEARCH LAB"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Dataset**  \n{dataset}")
    st.markdown("**Status**  \nData + evaluation framework")
    st.caption("Synthetic records are fictional demo data." if dataset == "Synthetic demo" else "Public mutation records support static exploration only; no longitudinal order is inferred.")

try:
    df = load_active_data()
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}"); st.stop()

if page == "OVERVIEW":
    st.markdown('<div class="status">RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)
    heading("ONCOGEN-X", "Forecasting Future Cancer-Associated Genetic Alterations", "ONCOGEN-X is a research prototype designed to investigate whether temporal patterns in cancer-associated genomic alterations can be represented and eventually used for future alteration forecasting.")
    st.markdown('<div class="notice"><b>Research prototype — computational research only.</b> Not a clinical diagnostic or treatment decision system.</div>', unsafe_allow_html=True)
    st.write("")
    cards([("Patients represented", df.patient_id.nunique()), ("Samples represented", df.sample_id.nunique() if "sample_id" in df else df.timepoint.nunique()), ("Genes", df.gene.nunique()), ("Timepoint labels", df.timepoint.nunique()), ("Mutation rows", len(df))])
    if dataset.startswith("cBioPortal"):
        st.caption("Counts above reflect samples and patients with mutation records. The study sample list contains additional samples; an absent mutation row is not a confirmed negative.")
    st.markdown("### Research pipeline")
    st.markdown('<div class="pipeline"><span class="pipe-step active">DATA INGESTION<br>✓ COMPLETED</span><span class="arrow">↓</span><span class="pipe-step active">EXPLORATION<br>✓ COMPLETED</span><span class="arrow">↓</span><span class="pipe-step active">TEMPORAL FEATURES<br>✓ IMPLEMENTED</span><span class="arrow">↓</span><span class="pipe-step active">STATIC / TEMPORAL MODELS<br>✓ IMPLEMENTED</span><span class="arrow">↓</span><span class="pipe-step">COHORT VALIDATION<br>REQUIRED</span></div>', unsafe_allow_html=True)
    st.markdown("### Research question")
    st.markdown('<div class="notice">Does representing patient-level genomic trajectories improve next-observation alteration-state prediction over current genomic state alone? The included METABRIC study is cross-sectional for this project, so the hypothesis remains untested.</div>', unsafe_allow_html=True)

elif page == "GENOMIC DATA":
    heading("PHASE 1 / FUNCTIONAL MODULE", "Genomic data ingestion & exploration", "Upload a schema-compatible CSV or explore the supplied clearly labelled synthetic demonstration dataset.")
    uploaded = st.file_uploader("Upload genomic CSV", type="csv", help="Required: patient_id, cancer_type, timepoint, gene, alteration, alteration_type, genomic_status, data_source")
    if uploaded is not None:
        try:
            uploaded.seek(0)
            candidate = pd.read_csv(uploaded); missing = validate_genomic_data(candidate)
            if missing: st.error("Upload rejected — missing: " + ", ".join(missing))
            else:
                uploaded.seek(0)
                st.session_state.uploaded_file = uploaded
                df = load_active_data()
                st.success("Dataset validated and loaded.")
        except Exception as exc: st.error(f"Could not read CSV: {exc}")
    a,b,c,d = st.columns(4)
    patients = a.multiselect("Patient", sorted(df.patient_id.unique()))
    cancers = b.multiselect("Cancer type", sorted(df.cancer_type.unique()))
    times = c.multiselect("Timepoint", sorted(df.timepoint.unique()))
    query = d.text_input("Search", placeholder="Gene, alteration, status...")
    active = filter_genomic_data(df, patients, cancers, times, query)
    sample_label = "Samples represented" if "sample_id" in active.columns else "Timepoint labels"
    sample_count = active.sample_id.nunique() if "sample_id" in active.columns else active.timepoint.nunique()
    cards([("Mutation rows", len(active)), ("Patients represented", active.patient_id.nunique()), ("Genes", active.gene.nunique()), (sample_label, sample_count), ("Missing values", int(active.isna().sum().sum()))])
    st.markdown("### Active genomic records")
    st.dataframe(active, use_container_width=True, hide_index=True, height=340)
    st.markdown("### Data quality")
    q1,q2=st.columns([1,1]); q1.dataframe(missing_value_summary(active), use_container_width=True, hide_index=True)
    q2.markdown(f"**Schema validation:** ✓ Required genomic fields present  \n\n**Duplicates removed during preprocessing:** {len(df) - len(df.drop_duplicates())}  \n\n**Data source:** {', '.join(active.data_source.dropna().unique()) if len(active) else 'No records'}")

elif page == "PATIENT TIMELINE":
    cross_sectional = df.timepoint.nunique() == 1 and "cross-sectional" in str(df.timepoint.iloc[0]).lower()
    heading("SAMPLE VIEW" if cross_sectional else "LONGITUDINAL VIEW", "Patient samples" if cross_sectional else "Patient timeline", "This cohort is cross-sectional. Samples are compared by identifier without implying chronology." if cross_sectional else "Observed alterations are shown across supplied timepoints; this is descriptive exploration, not a forecast.")
    patient = st.selectbox("Select patient", sorted(df.patient_id.unique()))
    p = df[df.patient_id == patient].copy(); cancer = p.cancer_type.iloc[0]
    cards([("Patient ID", patient), ("Cancer type", cancer), ("Sample IDs" if cross_sectional else "Timepoint labels", p.sample_id.nunique() if cross_sectional and "sample_id" in p.columns else p.timepoint.nunique()), ("Mutation rows", len(p))])
    p["y"] = p.gene
    x_column = "sample_id" if cross_sectional and "sample_id" in p.columns else "timepoint"
    fig = px.scatter(p, x=x_column, y="y", color="genomic_status", symbol="alteration_type", hover_data=["alteration", "alteration_type"], category_orders={x_column: sorted(p[x_column].unique())}, title="Observed alterations by sample" if cross_sectional else "Alteration emergence and persistence across time")
    fig.update_traces(marker=dict(size=14)); fig.update_layout(template="plotly_white", height=390, legend_title_text="Genomic status", xaxis_title="Sample ID" if cross_sectional else "Observation timepoint", yaxis_title="Gene")
    st.plotly_chart(fig, use_container_width=True)
    matrix = pd.crosstab(p.gene, p[x_column]).reindex(columns=sorted(p[x_column].unique()), fill_value=0)
    heat_title = "Mutation event rows by gene and sample; empty does not mean confirmed negative" if cross_sectional else "Gene × timepoint alteration presence"
    heat = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0,"#f2f6f9"],[1,"#0c7790"]], aspect="auto", title=heat_title)
    heat.update_layout(template="plotly_white", height=350, xaxis_title="Sample ID" if cross_sectional else "Timepoint", yaxis_title="Gene", coloraxis_colorbar_title="Event rows")
    st.plotly_chart(heat, use_container_width=True)

elif page == "ALTERATION ANALYSIS":
    heading("INTERACTIVE EXPLORATION", "Alteration analysis", "Frequency summaries are calculated from the currently loaded dataset.")
    left,right=st.columns(2)
    gene_counts=df.gene.value_counts().head(12).sort_values()
    left.plotly_chart(px.bar(gene_counts, orientation="h", title="Most frequent altered genes", labels={"value":"Records","index":"Gene"}, color=gene_counts.values, color_continuous_scale=["#a5dce2","#0c5e7a"]).update_layout(template="plotly_white", coloraxis_showscale=False), use_container_width=True)
    alt_counts=df.alteration.value_counts().head(12).sort_values()
    right.plotly_chart(px.bar(alt_counts, orientation="h", title="Alteration frequency", labels={"value":"Records","index":"Alteration"}, color=alt_counts.values, color_continuous_scale=["#b9e1db","#197a69"]).update_layout(template="plotly_white", coloraxis_showscale=False), use_container_width=True)
    left,right=st.columns(2)
    left.plotly_chart(px.pie(df, names="alteration_type", hole=.55, title="Alteration type distribution", color_discrete_sequence=px.colors.sequential.Tealgrn), use_container_width=True)
    right.plotly_chart(px.bar(df.cancer_type.value_counts(), title="Cancer type distribution", labels={"value":"Records","index":"Cancer type"}, color_discrete_sequence=["#0d7890"]).update_layout(template="plotly_white"), use_container_width=True)
    trend=df.groupby(["timepoint","genomic_status"]).size().reset_index(name="records")
    st.plotly_chart(px.bar(trend, x="timepoint", y="records", color="genomic_status", barmode="group", title="Alterations across timepoints", category_orders={"timepoint":sorted(df.timepoint.unique())}, color_discrete_map={"Persistent":"#1b8a9c","Acquired":"#e09b32"}).update_layout(template="plotly_white"), use_container_width=True)

elif page == "TEMPORAL FEATURES":
    heading("FEATURE INSPECTOR", "Temporal feature explorer", "Upload a curated wide state table with one row per ordered patient/sample and fully observed binary gene columns. Unknown states are rejected, never treated as absence.")
    state_upload = st.file_uploader("Curated state-table CSV", type="csv", key="state_table")
    if state_upload is not None:
        try:
            states = pd.read_csv(state_upload)
            genes = [c for c in states.columns if c not in {"patient_id", "sample_id", "timepoint"}]
            features = temporal_features(states, genes)
            if features.empty:
                st.info("No consecutive observed samples were found.")
            else:
                patient = st.selectbox("Patient", sorted(features.patient_id.unique()), key="feature_patient")
                selected = features[features.patient_id == patient]
                st.dataframe(selected, use_container_width=True, hide_index=True)
                st.download_button("Download extracted features", selected.to_csv(index=False), "temporal_features.csv", "text/csv")
                st.caption("Targets identify the next observed state and are never included as model features. Verify sample ordering and assay coverage independently.")
        except Exception as exc:
            st.error(f"State table rejected: {exc}")

elif page == "FUTURE FORECAST":
    st.markdown('<div class="planned">NO VERIFIED FORECAST MODEL</div>', unsafe_allow_html=True)
    heading("FORECASTING LAB", "Next-sample forecasts", "A synthetic benchmark is available for software demonstration, but no suitable real longitudinal cohort has been evaluated here. Held-out evaluation predictions are not future forecasts.")
    st.markdown('<div class="pipeline"><span class="pipe-step active">T1</span><span class="arrow">→</span><span class="pipe-step active">T2</span><span class="arrow">→</span><span class="pipe-step">TEMPORAL FEATURES</span><span class="arrow">→</span><span class="pipe-step">ML MODEL</span><span class="arrow">→</span><span class="pipe-step">T3 FORECAST</span></div>', unsafe_allow_html=True)
    cols=st.columns(3)
    for col,title,content,status in zip(cols,["Baseline","Sequential","Future"],["Logistic Regression<br>Random Forest","LSTM / GRU","Future sample predictions"],["IMPLEMENTED · DATA REQUIRED","NOT RUN · COHORT REQUIRED","UNAVAILABLE"]): col.markdown(f'<div class="metric-card"><div class="metric-label">{title}</div><div style="font-weight:700;color:#163a5c;margin-top:10px">{content}</div><div class="planned" style="margin-top:12px">{status}</div></div>', unsafe_allow_html=True)
    st.info("No real-patient future forecasts are displayed. Synthetic benchmark results, available on Model Performance, are not validated forecasts. The model pipeline does not produce clinical risk probabilities.")

elif page == "MODEL PERFORMANCE":
    heading("MODEL VALIDATION", "Static vs temporal comparison", "Saved evaluation outputs are displayed read-only. The benchmark output selector is separate from the exploration data source; METABRIC static records are never used as longitudinal benchmark results.")
    benchmark_dir, benchmark_kind = select_benchmark_output()
    metric_file = benchmark_dir / "metrics" / "model_comparison.csv" if benchmark_dir else None
    if metric_file and metric_file.is_file():
        metrics = pd.read_csv(metric_file)
        if benchmark_kind == "synthetic":
            st.subheader("SYNTHETIC DEMONSTRATION DATA — Evaluation metrics")
            display_metrics = metrics.copy()
            display_metrics.insert(0, "data_label", "SYNTHETIC DEMONSTRATION DATA")
        else:
            st.subheader("Saved evaluation metrics — source not verified")
            display_metrics = metrics
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
        macro = metrics[metrics.get("aggregation_level", pd.Series(index=metrics.index, dtype=str)) == "macro"]
        if len(macro):
            chart_title = ("SYNTHETIC DEMONSTRATION DATA — " if benchmark_kind == "synthetic" else "Source not verified — ") + "Macro F1: static vs temporal features"
            st.plotly_chart(px.bar(macro, x="model", y="f1", color="representation", barmode="group", title=chart_title).update_layout(template="plotly_white"), use_container_width=True)
        gene_metrics = metrics[(metrics.get("aggregation_level", pd.Series(index=metrics.index, dtype=str)) == "gene") & (~metrics.gene.astype(str).str.startswith("__"))]
        if len(gene_metrics):
            label = "SYNTHETIC DEMONSTRATION DATA — Gene confusion matrix" if benchmark_kind == "synthetic" else "Gene confusion matrix (source not verified)"
            selected_gene = st.selectbox(label, sorted(gene_metrics.gene.unique()))
            selected_model = st.selectbox("Model", sorted(gene_metrics.model.unique()), key="cm_model")
            row = gene_metrics[(gene_metrics.gene == selected_gene) & (gene_metrics.model == selected_model)].iloc[0]
            matrix = [[row.tn, row.fp], [row.fn, row.tp]]
            cm_title = ("SYNTHETIC DEMONSTRATION DATA — " if benchmark_kind == "synthetic" else "Source not verified — ") + f"{selected_model} · {selected_gene}"
            st.plotly_chart(px.imshow(matrix, text_auto=True, x=["Predicted 0", "Predicted 1"], y=["Observed 0", "Observed 1"], color_continuous_scale="Blues", title=cm_title).update_layout(template="plotly_white"), use_container_width=True)
        pred_file = benchmark_dir / "predictions" / "heldout_predictions.csv"
        if pred_file.is_file():
            pred = pd.read_csv(pred_file)
            download_label = "Download synthetic demonstration held-out predictions" if benchmark_kind == "synthetic" else "Download held-out predictions (source not verified)"
            download_name = "synthetic_demo_heldout_predictions.csv" if benchmark_kind == "synthetic" else "heldout_predictions.csv"
            st.download_button(download_label, pred.to_csv(index=False), download_name, "text/csv")
        metadata = benchmark_dir / "metrics" / "run_metadata.json"
        if metadata.is_file():
            st.subheader("SYNTHETIC DEMONSTRATION DATA — Run metadata" if benchmark_kind == "synthetic" else "Run metadata (source not verified)")
            st.json(json.loads(metadata.read_text(encoding="utf-8")))
    elif benchmark_dir:
        st.info("The selected saved run has no model_comparison.csv. No training is triggered by the dashboard.")
    else:
        st.markdown('<div class="planned">NO EXPERIMENTAL RESULTS</div>', unsafe_allow_html=True)
        st.markdown('<div class="notice">No saved benchmark outputs were found. The dashboard does not train models automatically.</div>', unsafe_allow_html=True)

elif page == "EXPLAINABILITY":
    heading("MODEL INSPECTION", "Random Forest feature importance", "Importance is descriptive of a fitted model and does not establish biological causation.")
    benchmark_dir, benchmark_kind = select_benchmark_output()
    importance_file = benchmark_dir / "metrics" / "random_forest_feature_importance.csv" if benchmark_dir else None
    if importance_file and importance_file.is_file():
        importance = pd.read_csv(importance_file)
        if benchmark_kind == "synthetic":
            st.subheader("SYNTHETIC DEMONSTRATION DATA — Random Forest feature importance")
            st.caption("Importance values are read from the saved synthetic run. They are model-specific descriptions, not biological evidence or clinical explanations.")
        else:
            st.subheader("Feature importance — output source not verified")
        gene_label = "SYNTHETIC DEMONSTRATION DATA — Target gene" if benchmark_kind == "synthetic" else "Target gene"
        gene = st.selectbox(gene_label, sorted(importance.gene.unique()))
        representation = st.selectbox("Feature representation", sorted(importance.representation.unique()))
        subset = importance[(importance.gene == gene) & (importance.representation == representation)].nlargest(20, "importance")
        chart_title = ("SYNTHETIC DEMONSTRATION DATA — " if benchmark_kind == "synthetic" else "Source not verified — ") + f"Top Random Forest features for {gene} ({representation})"
        st.plotly_chart(px.bar(subset.sort_values("importance"), x="importance", y="feature", orientation="h", title=chart_title).update_layout(template="plotly_white"), use_container_width=True)
    elif benchmark_dir:
        st.info("The selected saved run has no Random Forest feature-importance CSV. No training is triggered by the dashboard.")
    else:
        st.info("No saved Random Forest importance output exists. The dashboard does not train models automatically.")

elif page == "MOLECULAR DRIFT RESEARCH LAB":
    demo_data, demo_truth = make_demo_data(seed=42)
    heading("RESEARCH DEMONSTRATION · FICTIONAL DATA", "Molecular Drift Research Lab", "A configurable, transparent illustration of personal-baseline deviations and persistence across repeated fictional measurements.")
    st.warning("RESEARCH DEMONSTRATION USING FICTIONAL SYNTHETIC MEASUREMENTS. Molecular change does not establish cancer. This rule is not a diagnosis, validated risk score, or proof of cancer prediction.")

    controls, explanation = st.columns([1, 2])
    with controls:
        patient_id = st.selectbox("Fictional demo patient", sorted(demo_data.patient_id.unique()))
        threshold = st.slider("Illustrative deviation threshold (arbitrary units)", 0.25, 4.0, 1.0, 0.25, help="Configurable software-demo value only; not medically calibrated.")
        persistence = st.slider("Required consecutive observations", 2, 4, 2, 1, help="A deviation must exceed the threshold in this many adjacent observed measurements and remain in the same direction.")
        baseline_n = st.slider("Valid observations used for personal baseline", 2, 4, 2, 1)
    result = analyze_patient(demo_data, patient_id, threshold, persistence, baseline_n)

    with explanation:
        st.markdown("**Illustrative rule**  ")
        st.markdown(f"Baseline = mean of the first **{baseline_n} valid measurements**. A deviation qualifies when its absolute distance from baseline is at least **{threshold:g} arbitrary units**. The persistent state requires **{persistence} consecutive non-missing qualifying deviations in the same direction**.")
        st.caption("Values are generated deterministically from a fictional scenario. The thresholds are configurable demonstration settings, not medically meaningful cutoffs.")

    chart_data = result["observations"]
    fig = px.line(chart_data, x="observation", y="signal_units", markers=True,
                  title="SYNTHETIC DEMONSTRATION DATA — Repeated fictional measurements and personal baseline",
                  labels={"observation": "Ordered fictional observation", "signal_units": "Fictional signal (arbitrary units)"})
    if pd.notna(result["baseline"]):
        fig.add_hline(y=result["baseline"], line_dash="dash", line_color="#dc8d27",
                      annotation_text=f"Personal baseline: {result['baseline']:.2f} (mean of first {baseline_n} valid measurements)",
                      annotation_position="bottom right")
    fig.update_layout(template="plotly_white", height=390, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    deviation_chart = px.bar(chart_data, x="observation", y="deviation", color="beyond_threshold",
                             title="SYNTHETIC DEMONSTRATION DATA — Deviation from personal baseline",
                             labels={"observation": "Ordered fictional observation", "deviation": "Measurement − personal baseline (arbitrary units)", "beyond_threshold": "At / beyond threshold"},
                             color_discrete_map={True: "#d98732", False: "#16879c"})
    deviation_chart.add_hline(y=threshold, line_dash="dot", line_color="#b65345")
    deviation_chart.add_hline(y=-threshold, line_dash="dot", line_color="#b65345")
    deviation_chart.update_layout(template="plotly_white", height=310)
    st.plotly_chart(deviation_chart, use_container_width=True)

    st.subheader("SYNTHETIC DEMONSTRATION DATA — Data quality and rule output")
    available = result["valid_count"]
    sufficiency_text = "Sufficient for this illustrative rule" if result["sufficient"] else "Insufficient for this illustrative rule"
    cards([("Available measurements", f"{available} / {len(chart_data)}"), ("Missing measurements", result["missing_count"]), ("Current consecutive qualifying deviations", f"{result['current_persistence']} / {persistence}"), ("Data sufficiency", sufficiency_text)])
    if result["sufficient"]:
        st.markdown(f"### Rule state: **{result['state']}**")
    else:
        st.markdown(f"### Rule state: **{result['state']}**")
    baseline_values = chart_data.loc[chart_data.baseline_observation, ["observation", "signal_units"]]
    st.markdown(f"**Why:** Baseline measurements: {baseline_values.to_dict(orient='records') if len(baseline_values) else 'none'}. Baseline = {result['baseline']:.3f} arbitrary units." if pd.notna(result["baseline"]) else "**Why:** There are too few valid measurements to calculate the requested personal baseline.")
    st.markdown(f"Threshold: |deviation| ≥ {threshold:g}. Latest observation: {chart_data.observation.iloc[-1]:g}; latest deviation: {chart_data.deviation.iloc[-1]:.3f} arbitrary units." if pd.notna(chart_data.deviation.iloc[-1]) else "Latest ordered measurement is missing, so the state is insufficient data.")
    if result["state"] == "persistent change for research review":
        evidence = chart_data.loc[chart_data.consecutive_deviations >= persistence, ["observation", "signal_units", "deviation", "consecutive_deviations"]]
        st.write(f"The trailing sequence reached {result['current_persistence']} consecutive qualifying observations. Supporting observations: {evidence.to_dict(orient='records')}.")
    elif result["state"] == "isolated change":
        candidates = chart_data.loc[(~chart_data.baseline_observation) & chart_data.beyond_threshold, ["observation", "signal_units", "deviation"]]
        st.write(f"The required consecutive sequence was not present at the latest observation. Threshold-crossing observations: {candidates.to_dict(orient='records') if len(candidates) else 'none'}. This three-state demo groups stable/non-persistent trajectories under ‘isolated change’; when there are no crossings, it does not mean an isolated event occurred.")
    else:
        st.write(f"Available valid measurements: {available}; the rule requires at least {baseline_n + persistence}, plus a non-missing latest measurement.")

    st.markdown("### Rule workflow")
    st.markdown('<div class="pipeline"><span class="pipe-step active">FICTIONAL REPEATED MEASUREMENTS</span><span class="arrow">→</span><span class="pipe-step active">FIRST VALID OBSERVATIONS DEFINE BASELINE</span><span class="arrow">→</span><span class="pipe-step active">BASELINE DEVIATIONS</span><span class="arrow">→</span><span class="pipe-step active">MISSINGNESS + CONSECUTIVE PERSISTENCE</span><span class="arrow">→</span><span class="pipe-step">ILLUSTRATIVE THREE-STATE OUTPUT</span></div>', unsafe_allow_html=True)

    st.markdown("### Simple rule vs trajectory rule on the synthetic simulation")
    st.caption("Synthetic scenario labels indicate whether the generator planted a persistent shift. They are simulation labels, not clinical outcomes. The single-measurement rule flags any post-baseline threshold crossing; the trajectory rule requires the configured trailing same-direction persistence and sufficient data.")
    comparison, _ = compare_demo_rules(demo_data, demo_truth, threshold, persistence, baseline_n)
    comparison_display = comparison.copy()
    comparison_display.insert(0, "data_label", "SYNTHETIC DEMONSTRATION DATA")
    st.dataframe(comparison_display, use_container_width=True, hide_index=True)
    st.caption("Accuracy = (TP + TN) / all fictional patients; precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = harmonic mean of precision and recall. Counts are shown. This 12-patient deterministic toy simulation is small, constructed, threshold-dependent, and not an estimate of clinical performance.")
    with st.expander("Inspect fictional measurements and missingness"):
        st.dataframe(chart_data, use_container_width=True, hide_index=True)

else:
    heading("RESEARCH PLAN", "Roadmap to longitudinal genomic forecasting", "The current prototype implements transparent data preparation and exploration only.")
    stages=[("IMPLEMENTED","✓ CSV validation &nbsp;&nbsp; ✓ Synthetic demo explorer &nbsp;&nbsp; ✓ cBioPortal METABRIC static downloader &nbsp;&nbsp; ✓ Guarded temporal feature builder &nbsp;&nbsp; ✓ Patient-grouped benchmark code"),("BLOCKED ON COHORT","○ Curate genuine ordered repeat samples with verified assay coverage &nbsp;&nbsp; ○ Run and report actual evaluation"),("PLANNED IF DATA SUPPORTS","○ Horizon analysis &nbsp;&nbsp; ○ LSTM / GRU only with adequate independent patient sequences"),("FUTURE","○ Feature importance and uncertainty reporting &nbsp;&nbsp; ○ External validation")]
    for index,(title,body) in enumerate(stages): st.markdown(f'<div class="road-card {"current" if index==0 else ""}"><div class="eyebrow">{title}</div><div style="margin-top:8px;color:#304d67">{body}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice">Research integrity: included METABRIC data support static exploration only. No temporal forecast result exists without a verified serial cohort.</div>', unsafe_allow_html=True)
```

## Part 4 — `src/__init__.py`

```python
"""ONCOGEN-X data utilities."""
```

## Part 5 — `src/data_loader.py`

```python
"""Validated data ingestion for the ONCOGEN-X research prototype."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "patient_id", "cancer_type", "timepoint", "gene", "alteration",
    "alteration_type", "genomic_status", "data_source",
]


def validate_genomic_data(df: pd.DataFrame) -> list[str]:
    """Return a list of required columns absent from a candidate dataset."""
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def clean_genomic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise strings and handle basic missing values without inventing data."""
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column].dtype) or cleaned[column].dtype == "object":
            # Normalize text while preserving nulls for honest missingness reporting.
            cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return cleaned.drop_duplicates().reset_index(drop=True)


def load_genomic_data(source: str | Path | object) -> pd.DataFrame:
    """Read CSV data, validate its schema, and return a clean DataFrame."""
    data = pd.read_csv(source)
    missing = validate_genomic_data(data)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return clean_genomic_data(data)
```

## Part 6 — `src/download_cbioportal.py`

```python
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
```

## Part 7 — `src/evaluation.py`

```python
"""Patient-held-out static/temporal benchmark for fully observed state tables."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from src.temporal_features import temporal_features, feature_columns


def run_benchmark(states: pd.DataFrame, gene_columns: list[str], output: str | Path,
                  seed: int = 42) -> pd.DataFrame:
    """Benchmark four configurations with one patient-grouped split.

    Only genes explicitly represented as complete observed 0/1 states are valid.
    No metric is computed for a target with one class in the test set.
    """
    out = Path(output)
    for folder in ("metrics", "predictions", "models"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    frame = temporal_features(states, gene_columns)
    if frame.empty or frame.patient_id.nunique() < 4:
        raise ValueError("At least four independent patients with consecutive states are required.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_i, test_i = next(split.split(frame, groups=frame.patient_id))
    train, test = frame.iloc[train_i], frame.iloc[test_i]
    summaries, predictions, importances = [], [], []
    for representation, temporal in (("static", False), ("temporal", True)):
        cols = feature_columns(frame, temporal)
        X_train, X_test = train[cols], test[cols]
        for model_name in ("logistic_regression", "random_forest"):
            for gene in gene_columns:
                target = f"target__{gene}"
                y_train, y_test = train[target].astype(int), test[target].astype(int)
                if y_train.nunique() < 2:
                    summaries.append(_metric_row(representation, model_name, gene, y_test, None, None,
                                                  frame, train, test, "training target has one class"))
                    continue
                estimator = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                             class_weight="balanced", random_state=seed)) if model_name == "logistic_regression"
                             else RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced_subsample", random_state=seed, n_jobs=-1))
                estimator.fit(X_train, y_train)
                prob = estimator.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                if model_name == "random_forest":
                    for feature, importance in zip(cols, estimator.feature_importances_):
                        importances.append({"representation": representation, "gene": gene,
                                            "feature": feature, "importance": float(importance)})
                summaries.append(_metric_row(representation, model_name, gene, y_test, pred, prob,
                                              frame, train, test, ""))
                for j, (_, row) in enumerate(test.iterrows()):
                    predictions.append({"representation": representation, "model": model_name,
                        "gene": gene, "patient_id": row.patient_id, "cutoff_sample": row.cutoff_sample,
                        "target_sample": row.target_sample, "observed": int(y_test.iloc[j]),
                        "predicted": int(pred[j]), "score": float(prob[j])})
                joblib.dump({"estimator": estimator, "feature_columns": cols, "gene": gene,
                             "representation": representation, "seed": seed},
                            out / "models" / f"{representation}_{model_name}_{gene}.joblib")
    metrics = pd.DataFrame(summaries)
    metric_rows = metrics.to_dict("records")
    for record in metric_rows:
        record["aggregation_level"] = "gene"
    for representation in ("static", "temporal"):
        for model_name in ("logistic_regression", "random_forest"):
            gene_rows = [r for r in metric_rows if r["representation"] == representation and
                         r["model"] == model_name and r.get("aggregation_level") == "gene"]
            if not gene_rows:
                continue
            usable = [r for r in gene_rows if r["f1"] is not None and not pd.isna(r["f1"])]
            macro = {"representation": representation, "model": model_name, "gene": "__macro__",
                     "aggregation_level": "macro", "patients_total": frame.patient_id.nunique(),
                     "train_patients": train.patient_id.nunique(), "test_patients": test.patient_id.nunique(),
                     "test_transitions": len(test), "test_positive": sum(r["test_positive"] for r in gene_rows),
                     "note": "Unweighted mean across evaluable genes."}
            for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                values = [r[name] for r in gene_rows if r[name] is not None and not pd.isna(r[name])]
                macro[name] = sum(values) / len(values) if values else None
            for name in ("tn", "fp", "fn", "tp"):
                macro[name] = None
            metric_rows.append(macro)

            pred_rows = [r for r in predictions if r["representation"] == representation and r["model"] == model_name]
            if pred_rows:
                y_true = [r["observed"] for r in pred_rows]
                y_pred = [r["predicted"] for r in pred_rows]
                y_score = [r["score"] for r in pred_rows]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
                micro = {"representation": representation, "model": model_name, "gene": "__micro__",
                         "aggregation_level": "micro", "patients_total": frame.patient_id.nunique(),
                         "train_patients": train.patient_id.nunique(), "test_patients": test.patient_id.nunique(),
                         "test_transitions": len(test), "test_positive": int(sum(y_true)),
                         "accuracy": accuracy_score(y_true, y_pred),
                         "precision": precision_score(y_true, y_pred, zero_division=0),
                         "recall": recall_score(y_true, y_pred, zero_division=0),
                         "f1": f1_score(y_true, y_pred, zero_division=0), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                         "note": "Micro metrics pool held-out gene-transition labels."}
                micro["roc_auc"] = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None
                micro["pr_auc"] = average_precision_score(y_true, y_score) if len(set(y_true)) == 2 else None
                metric_rows.append(micro)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics" / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions" / "heldout_predictions.csv", index=False)
    pd.DataFrame(importances).to_csv(out / "metrics" / "random_forest_feature_importance.csv", index=False)
    metadata = {"seed": seed, "split": "single GroupShuffleSplit, 25% held-out patients",
                "train_patients": sorted(train.patient_id.unique().tolist()),
                "test_patients": sorted(test.patient_id.unique().tolist()),
                "train_transitions": len(train), "test_transitions": len(test),
                "genes": gene_columns, "target": "gene binary status at next observed sample",
                "caveat": "Interpret as alteration-state forecasting only when assay coverage and time order are verified."}
    (out / "metrics" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def _metric_row(rep, model, gene, truth, pred, probability, frame, train, test, note):
    auc = ap = None
    if pred is None:
        acc = precision = recall = f1 = None
        tn = fp = fn = tp = None
    else:
        acc = accuracy_score(truth, pred)
        precision = precision_score(truth, pred, zero_division=0)
        recall = recall_score(truth, pred, zero_division=0)
        f1 = f1_score(truth, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel().tolist()
        if truth.nunique() > 1:
            auc = roc_auc_score(truth, probability)
            ap = average_precision_score(truth, probability)
    return {"representation": rep, "model": model, "gene": gene, "aggregation_level": "gene",
            "patients_total": frame.patient_id.nunique(), "train_patients": train.patient_id.nunique(),
            "test_patients": test.patient_id.nunique(), "test_transitions": len(test),
            "test_positive": int(truth.sum()), "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": auc, "pr_auc": ap,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "note": note or
            ("ROC-AUC/PR-AUC undefined: test target has one class." if truth.nunique() < 2 else "")}
```

## Part 8 — `src/molecular_drift.py`

```python
"""Fictional, deterministic data and transparent rules for a drift demo page."""

import numpy as np
import pandas as pd


def make_demo_data(seed=42):
    """Create fictional measurements and planted simulation labels (not clinical data)."""
    rng = np.random.default_rng(seed)
    rows = []
    truth = {}
    for patient_number in range(1, 13):
        patient_id = f"DRIFT-DEMO-{patient_number:02d}"
        scenario = "persistent" if patient_number <= 6 else "isolated" if patient_number <= 9 else "stable"
        truth[patient_id] = int(scenario == "persistent")
        personal_level = 8.0 + patient_number * 0.35
        for observation in range(1, 9):
            value = personal_level + rng.normal(0, 0.12)
            if scenario == "persistent" and observation >= 4:
                value += 2.2 + rng.normal(0, 0.12)
            elif scenario == "isolated" and observation == 5:
                value += 2.2
            if (patient_id, observation) in {
                ("DRIFT-DEMO-02", 6), ("DRIFT-DEMO-08", 7), ("DRIFT-DEMO-11", 3)
            }:
                value = np.nan
            rows.append({
                "patient_id": patient_id,
                "observation": observation,
                "signal_units": round(float(value), 3) if not np.isnan(value) else np.nan,
            })
    return pd.DataFrame(rows), truth


def analyze_patient(data, patient_id, threshold=1.0, persistence=2, baseline_n=2):
    """Calculate baseline-relative deviations and a trailing same-direction rule state."""
    required = {"patient_id", "observation", "signal_units"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a finite positive number")
    if not isinstance(persistence, int) or persistence < 1:
        raise ValueError("persistence must be a positive integer")
    if not isinstance(baseline_n, int) or baseline_n < 1:
        raise ValueError("baseline_n must be a positive integer")

    patient = data.loc[data.patient_id == patient_id].copy()
    if patient.empty:
        raise ValueError(f"Unknown patient_id: {patient_id}")
    raw_observation = patient.observation.copy()
    raw_signal = patient.signal_units.copy()
    patient["observation"] = pd.to_numeric(raw_observation, errors="coerce")
    patient["signal_units"] = pd.to_numeric(raw_signal, errors="coerce")
    if (raw_observation.notna() & patient.observation.isna()).any():
        raise ValueError("observation values must be numeric")
    if (raw_signal.notna() & patient.signal_units.isna()).any():
        raise ValueError("signal_units must contain numeric values or explicit missing values")
    if patient.observation.isna().any() or patient.observation.duplicated().any():
        raise ValueError("observation values must be numeric and unique within patient")
    if patient.signal_units.dropna().map(np.isfinite).eq(False).any():
        raise ValueError("signal_units must contain finite numbers or missing values")
    patient = patient.sort_values("observation").reset_index(drop=True)
    patient["deviation"] = np.nan
    patient["baseline_observation"] = False
    valid_indices = patient.index[patient.signal_units.notna()].tolist()
    enough_baseline = len(valid_indices) >= baseline_n
    baseline_indices = valid_indices[:baseline_n] if enough_baseline else valid_indices
    patient.loc[baseline_indices, "baseline_observation"] = True
    baseline = float(patient.loc[baseline_indices, "signal_units"].mean()) if enough_baseline else np.nan
    if enough_baseline:
        patient["deviation"] = patient.signal_units - baseline

    patient["beyond_threshold"] = patient.deviation.abs() >= threshold
    patient["consecutive_deviations"] = 0
    streak = 0
    prior_direction = 0
    for index, row in patient.iterrows():
        if row.baseline_observation:
            continue
        if pd.isna(row.signal_units) or not row.beyond_threshold:
            streak, prior_direction = 0, 0
        else:
            direction = 1 if row.deviation > 0 else -1
            streak = streak + 1 if direction == prior_direction else 1
            prior_direction = direction
        patient.at[index, "consecutive_deviations"] = streak

    missing_count = int(patient.signal_units.isna().sum())
    sufficient = enough_baseline and len(valid_indices) >= baseline_n + persistence and pd.notna(patient.signal_units.iloc[-1])
    final_streak = int(patient.consecutive_deviations.iloc[-1]) if len(patient) else 0
    if not sufficient:
        state = "insufficient data"
    elif final_streak >= persistence:
        state = "persistent change for research review"
    else:
        state = "isolated change"

    return {
        "observations": patient,
        "baseline": baseline,
        "baseline_observations": patient.loc[patient.baseline_observation, "observation"].tolist(),
        "missing_count": missing_count,
        "valid_count": len(valid_indices),
        "sufficient": bool(sufficient),
        "current_persistence": final_streak,
        "state": state,
        "threshold": float(threshold),
        "persistence_required": persistence,
        "baseline_n": baseline_n,
    }


def compare_demo_rules(data, truth, threshold=1.0, persistence=2, baseline_n=2):
    """Compare two toy rules with planted synthetic patient labels; return counts and metrics."""
    rows = []
    for patient_id, expected in truth.items():
        result = analyze_patient(data, patient_id, threshold, persistence, baseline_n)
        post_baseline = result["observations"].loc[~result["observations"].baseline_observation]
        single_measurement = bool((post_baseline.deviation.abs() >= threshold).any())
        trajectory_rule = result["state"] == "persistent change for research review"
        for rule, predicted in [("single measurement", single_measurement), ("trajectory rule", trajectory_rule)]:
            rows.append({"patient_id": patient_id, "rule": rule, "truth": int(expected), "predicted": int(predicted)})
    outcomes = pd.DataFrame(rows)
    metrics = []
    for rule, subset in outcomes.groupby("rule", sort=False):
        tp = int(((subset.truth == 1) & (subset.predicted == 1)).sum())
        fp = int(((subset.truth == 0) & (subset.predicted == 1)).sum())
        fn = int(((subset.truth == 1) & (subset.predicted == 0)).sum())
        tn = int(((subset.truth == 0) & (subset.predicted == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics.append({
            "rule": rule, "patients": len(subset), "accuracy": (tp + tn) / len(subset),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return pd.DataFrame(metrics), outcomes
```

## Part 9 — `src/preprocessing.py`

```python
"""Small, transparent preprocessing helpers for exploration."""
import pandas as pd


def filter_genomic_data(df: pd.DataFrame, patients=None, cancer_types=None, timepoints=None, query=""):
    """Apply exploration filters and a case-insensitive search across displayed fields."""
    result = df.copy()
    if patients:
        result = result[result["patient_id"].isin(patients)]
    if cancer_types:
        result = result[result["cancer_type"].isin(cancer_types)]
    if timepoints:
        result = result[result["timepoint"].isin(timepoints)]
    if query:
        searchable = result.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
        result = result[searchable.any(axis=1)]
    return result.reset_index(drop=True)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missingness for the active dataset."""
    counts = df.isna().sum()
    return pd.DataFrame({"Column": counts.index, "Missing values": counts.values,
                         "Missing (%)": (counts.values / max(len(df), 1) * 100).round(1)})
```

## Part 10 — `src/temporal_features.py`

```python
"""Leakage-safe temporal features for explicitly observed patient states.

Input states are wide tables with one row per patient/sample and binary gene
columns. Only genes measured at each sample may be used; callers must exclude
unmeasured states rather than treating missing values as zero.
"""
from __future__ import annotations

import pandas as pd

META = ["patient_id", "sample_id", "timepoint"]


def _ordered_timepoints(values: pd.Series) -> tuple[pd.Series, bool]:
    """Normalize supported timepoints; reject labels without temporal semantics."""
    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
        order = pd.to_numeric(values, errors="coerce")
        if not order.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
            raise ValueError("Numeric timepoints must be finite.")
        return order, False
    strings = values.astype(str)
    numeric = pd.to_numeric(strings, errors="coerce")
    if numeric.notna().all():
        if not numeric.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
            raise ValueError("Numeric timepoints must be finite.")
        return numeric, False
    # Require ISO calendar dates/datetimes. Arbitrary labels (including T1/T2)
    # cannot establish chronology and must not be sorted into transitions.
    if not strings.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+Zz-]+)?$").all():
        raise ValueError("Timepoints must be finite numeric values or valid ISO dates/datetimes.")
    parsed = pd.to_datetime(strings, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("Timepoints contain an invalid ISO date/datetime.")
    return parsed, True


def validate_states(states: pd.DataFrame, gene_columns: list[str]) -> None:
    missing = [c for c in META + gene_columns if c not in states.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if not gene_columns or len(set(gene_columns)) != len(gene_columns) or set(gene_columns) & set(META):
        raise ValueError("Gene columns must be non-empty, unique, and separate from metadata.")
    if not states.columns.is_unique:
        raise ValueError("Input column names must be unique.")
    if states[META].isna().any().any():
        raise ValueError("Patient, sample and timepoint must be present.")
    if states[["patient_id", "sample_id"]].astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Patient and sample identifiers must be non-empty.")
    if states.duplicated(["patient_id", "sample_id"]).any():
        raise ValueError("Each patient/sample pair must be unique.")
    time_order, _ = _ordered_timepoints(states["timepoint"])
    temporal_keys = pd.DataFrame({"patient_id": states["patient_id"], "_time_order": time_order})
    if temporal_keys.duplicated(["patient_id", "_time_order"]).any():
        raise ValueError("Each patient must have a unique timepoint; co-temporal samples cannot define transitions.")
    values = states[gene_columns]
    if values.isna().any().any() or not values.isin([0, 1, False, True]).all().all():
        raise ValueError("Gene states must be observed binary values; missing is not negative.")


def temporal_features(states: pd.DataFrame, gene_columns: list[str]) -> pd.DataFrame:
    """Return one cutoff row per consecutive observed sample transition.

    `timepoint` must be sortable in true temporal order (dates or ordered values).
    History features only use samples at or before the cutoff; next state is
    intentionally not returned as a feature.
    """
    validate_states(states, gene_columns)
    time_order, is_iso_date = _ordered_timepoints(states["timepoint"])
    ordered = states.assign(_time_order=time_order)
    if is_iso_date:
        ordered = ordered.assign(_parsed_time=time_order)
    ordered = ordered.sort_values(["patient_id", "_time_order", "sample_id"], kind="stable")
    rows = []
    for patient, group in ordered.groupby("patient_id", sort=False):
        records = list(group.to_dict("records"))
        for idx in range(len(records) - 1):
            history = records[: idx + 1]
            current, future = history[-1], records[idx + 1]
            row = {"patient_id": patient, "cutoff_sample": current["sample_id"],
                   "target_sample": future["sample_id"], "cutoff_timepoint": current["timepoint"],
                   "target_timepoint": future["timepoint"],
                   "current_alteration_count": int(sum(int(current[g]) for g in gene_columns)),
                   "new_acquisition_count": int(sum(int(current[g]) and not int(history[-2][g])
                                                     for g in gene_columns)) if len(history) > 1 else 0,
                   "loss_count_since_previous": int(sum(not int(current[g]) and int(history[-2][g])
                                                         for g in gene_columns)) if len(history) > 1 else 0}
            if is_iso_date:
                row["days_since_previous_sample"] = (
                    float((current["_parsed_time"] - history[-2]["_parsed_time"]).total_seconds() / 86400)
                    if len(history) > 1 else -1.0)
            row["observations_so_far"] = len(history)
            for gene in gene_columns:
                vals = [int(r[gene]) for r in history]
                row[f"{gene}__current"] = vals[-1]
                row[f"{gene}__historical_frequency"] = sum(vals) / len(vals)
                row[f"{gene}__consecutive_persistence"] = next(
                    (n - 1 for n, value in enumerate(reversed(vals), start=1) if not value), len(vals))
                row[f"{gene}__ever_observed"] = int(any(vals))
                row[f"{gene}__last_acquired_steps"] = next(
                    (len(vals) - 1 - j for j in range(len(vals) - 1, 0, -1)
                     if vals[j] and not vals[j - 1]), -1)
                # Historical co-occurrence with each other gene, excluding current/future labels.
                for other in gene_columns:
                    if other != gene:
                        row[f"{gene}__cooccur_{other}"] = sum(int(r[gene]) * int(r[other]) for r in history) / len(history)
                row[f"target__{gene}"] = int(future[gene])
            rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, temporal: bool) -> list[str]:
    excluded = {"patient_id", "cutoff_sample", "target_sample", "cutoff_timepoint", "target_timepoint"}
    excluded.update(c for c in frame.columns if c.startswith("target__"))
    if not temporal:
        return [c for c in frame.columns if c not in excluded and
                (c == "current_alteration_count" or c.endswith("__current"))]
    return [c for c in frame.columns if c not in excluded]
```

## Part 11 — `src/train.py`

```python
"""Run the patient-held-out static vs temporal benchmark."""
import argparse
from pathlib import Path
import pandas as pd
from src.evaluation import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", help="CSV: patient_id,sample_id,timepoint plus complete binary gene columns")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    states = pd.read_csv(args.states)
    meta = {"patient_id", "sample_id", "timepoint"}
    genes = [c for c in states.columns if c not in meta]
    if not genes:
        parser.error("No gene-state columns found.")
    results = run_benchmark(states, genes, Path(args.output), args.seed)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
```

## Part 12 — `tests/test_evaluation.py`

```python
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation import run_benchmark


class GroupedBenchmarkTests(unittest.TestCase):
    def test_metrics_and_predictions_are_written_with_patient_separation(self):
        rows = []
        for patient in range(16):
            for timepoint in range(3):
                rows.append({"patient_id": f"p{patient}", "sample_id": f"s{patient}_{timepoint}",
                             "timepoint": timepoint,
                             "TP53": int((patient // 2 + timepoint) % 4 >= 2),
                             "KRAS": int((patient + timepoint) % 3 == 0)})
        states = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory(dir="outputs") as directory:
            metrics = run_benchmark(states, ["TP53", "KRAS"], directory, seed=42)
            metadata = json.loads((Path(directory) / "metrics" / "run_metadata.json").read_text())
            self.assertTrue((Path(directory) / "predictions" / "heldout_predictions.csv").exists())
            self.assertTrue((Path(directory) / "metrics" / "random_forest_feature_importance.csv").exists())
            self.assertTrue(set(metadata["train_patients"]).isdisjoint(metadata["test_patients"]))
            self.assertEqual(set(metrics["representation"]), {"static", "temporal"})
            fitted = metrics[metrics["aggregation_level"] == "gene"]
            self.assertEqual(set(zip(fitted["representation"], fitted["model"])), {
                ("static", "logistic_regression"), ("static", "random_forest"),
                ("temporal", "logistic_regression"), ("temporal", "random_forest")})
            predictions = pd.read_csv(Path(directory) / "predictions" / "heldout_predictions.csv")
            self.assertEqual(set(zip(predictions["representation"], predictions["model"])), {
                ("static", "logistic_regression"), ("static", "random_forest"),
                ("temporal", "logistic_regression"), ("temporal", "random_forest")})
            self.assertEqual(len(list((Path(directory) / "models").glob("*.joblib"))), 8)


if __name__ == "__main__":
    unittest.main()
```

## Part 13 — `tests/test_molecular_drift.py`

```python
import unittest

import pandas as pd

from src.molecular_drift import analyze_patient, compare_demo_rules, make_demo_data


def patient_frame(values):
    return pd.DataFrame({
        "patient_id": "fictional-p1",
        "observation": range(1, len(values) + 1),
        "signal_units": values,
    })


class MolecularDriftTests(unittest.TestCase):
    def test_baseline_uses_first_valid_measurements_and_computes_deviation(self):
        result = analyze_patient(patient_frame([10.0, 12.0, 14.0, 14.0]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["baseline"], 11.0)
        self.assertEqual(result["baseline_observations"], [1, 2])
        self.assertEqual(result["observations"].deviation.tolist(), [-1.0, 1.0, 3.0, 3.0])

    def test_persistence_requires_consecutive_same_direction_observations(self):
        result = analyze_patient(patient_frame([10, 10, 12, 12]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["current_persistence"], 2)
        self.assertEqual(result["state"], "persistent change for research review")

    def test_missing_observation_breaks_persistence_and_is_counted(self):
        result = analyze_patient(patient_frame([10, 10, 12, None, 12]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["current_persistence"], 1)
        self.assertEqual(result["state"], "isolated change")

    def test_latest_missing_measurement_returns_insufficient_data(self):
        result = analyze_patient(patient_frame([10, 10, 12, 12, None]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["state"], "insufficient data")

    def test_insufficient_valid_observations_returns_insufficient_data(self):
        result = analyze_patient(patient_frame([10, 10, 12]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["state"], "insufficient data")

    def test_nonpersistent_adequate_trajectory_returns_isolated_state(self):
        result = analyze_patient(patient_frame([10, 10, 12, 10, 10]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["state"], "isolated change")
        self.assertEqual(result["current_persistence"], 0)

    def test_invalid_threshold_and_unknown_patient_are_rejected(self):
        data = patient_frame([10, 10, 12, 12])
        with self.assertRaisesRegex(ValueError, "threshold"):
            analyze_patient(data, "fictional-p1", threshold=0)
        with self.assertRaisesRegex(ValueError, "Unknown patient"):
            analyze_patient(data, "unknown")

    def test_nonnumeric_measurements_are_not_silently_treated_as_missing(self):
        data = patient_frame([10, 10, 12, 12])
        data["signal_units"] = data.signal_units.astype(object)
        data.loc[2, "signal_units"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "numeric values"):
            analyze_patient(data, "fictional-p1")

    def test_demo_is_deterministic_and_comparison_uses_planted_synthetic_labels(self):
        data_a, truth_a = make_demo_data()
        data_b, truth_b = make_demo_data()
        pd.testing.assert_frame_equal(data_a, data_b)
        self.assertEqual(truth_a, truth_b)
        metrics, outcomes = compare_demo_rules(data_a, truth_a)
        self.assertEqual(set(metrics.rule), {"single measurement", "trajectory rule"})
        self.assertTrue((metrics.patients == len(truth_a)).all())
        self.assertEqual(len(outcomes), 2 * len(truth_a))


if __name__ == "__main__":
    unittest.main()
```

## Part 14 — `tests/test_temporal_features.py`

```python
import pandas as pd
import unittest

from src.temporal_features import temporal_features, validate_states
from src.data_loader import load_genomic_data, validate_genomic_data


def fixture_states():
    return pd.DataFrame([
        {"patient_id": "p1", "sample_id": "s1", "timepoint": 1, "TP53": 0, "KRAS": 0},
        {"patient_id": "p1", "sample_id": "s2", "timepoint": 2, "TP53": 1, "KRAS": 0},
        {"patient_id": "p1", "sample_id": "s3", "timepoint": 3, "TP53": 1, "KRAS": 1},
        {"patient_id": "p2", "sample_id": "s1", "timepoint": 1, "TP53": 0, "KRAS": 1},
        {"patient_id": "p2", "sample_id": "s2", "timepoint": 2, "TP53": 0, "KRAS": 1},
])


class TemporalFeatureTests(unittest.TestCase):
    def test_temporal_features_only_include_past_and_targets_are_next_state(self):
        result = temporal_features(fixture_states(), ["TP53", "KRAS"])
        p1_second = result[(result.patient_id == "p1") & (result.cutoff_sample == "s2")].iloc[0]
        self.assertEqual(p1_second["TP53__current"], 1)
        self.assertEqual(p1_second["TP53__historical_frequency"], 0.5)
        self.assertEqual(p1_second["TP53__last_acquired_steps"], 0)
        self.assertEqual(p1_second["target__KRAS"], 1)
        from src.temporal_features import feature_columns
        self.assertFalse(any(c.startswith("target__") for c in feature_columns(result, True)))


    def test_missing_state_is_not_assumed_negative(self):
        states = fixture_states()
        states.loc[0, "TP53"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing is not negative"):
            validate_states(states, ["TP53", "KRAS"])


    def test_duplicate_patient_sample_is_rejected(self):
        states = fixture_states()
        states.loc[1, "sample_id"] = "s1"
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_states(states, ["TP53", "KRAS"])

    def test_blank_identifiers_are_rejected(self):
        states = fixture_states()
        states.loc[0, "patient_id"] = "  "
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_states(states, ["TP53", "KRAS"])

    def test_co_temporal_samples_are_rejected(self):
        states = fixture_states()
        states.loc[1, "timepoint"] = 1
        with self.assertRaisesRegex(ValueError, "unique timepoint"):
            temporal_features(states, ["TP53", "KRAS"])

    def test_unordered_labels_and_invalid_dates_are_rejected(self):
        states = fixture_states().query("patient_id == 'p1'").copy()
        states["timepoint"] = ["T1", "T10", "T2"]
        with self.assertRaisesRegex(ValueError, "numeric values or valid ISO"):
            temporal_features(states, ["TP53", "KRAS"])
        states["timepoint"] = ["2020-01-01", "2020-02-30", "2020-03-01"]
        with self.assertRaisesRegex(ValueError, "invalid ISO"):
            temporal_features(states, ["TP53", "KRAS"])

    def test_numeric_string_timepoints_sort_numerically(self):
        states = fixture_states().query("patient_id == 'p1'").copy()
        states["timepoint"] = ["10", "2", "1"]
        result = temporal_features(states, ["TP53", "KRAS"])
        self.assertEqual(result.iloc[0]["cutoff_sample"], "s3")
        self.assertEqual(result.iloc[0]["target_sample"], "s2")

    def test_real_iso_dates_produce_observed_gap_feature(self):
        states = fixture_states().query("patient_id == 'p1'").copy()
        states["timepoint"] = ["2020-01-01", "2020-01-08", "2020-01-18"]
        result = temporal_features(states, ["TP53", "KRAS"])
        self.assertEqual(result.iloc[0]["days_since_previous_sample"], -1.0)
        self.assertEqual(result.iloc[1]["days_since_previous_sample"], 7.0)


    def test_provided_synthetic_demo_is_validated_and_labeled(self):
        data = load_genomic_data("data/sample/demo_genomic_data.csv")
        self.assertFalse(validate_genomic_data(data))
        self.assertTrue(data.data_source.str.contains("synthetic", case=False).all())

    def test_cleaning_preserves_missingness(self):
        from src.data_loader import clean_genomic_data
        frame = pd.DataFrame({"patient_id": [" p1 ", None]})
        cleaned = clean_genomic_data(frame)
        self.assertEqual(cleaned.patient_id.iloc[0], "p1")
        self.assertTrue(pd.isna(cleaned.patient_id.iloc[1]))

    def test_cbioportal_mutations_join_sample_patient_without_time_inference(self):
        from src.download_cbioportal import _mutation_events
        events = _mutation_events([{"sampleId": "S1", "gene": {"hugoGeneSymbol": "TP53"},
                                    "proteinChange": "p.R1C", "mutationType": "Missense"}], {"S1": "P1"})
        self.assertEqual(events.loc[0, "patient_id"], "P1")
        self.assertEqual(events.loc[0, "timepoint"], "single cross-sectional sample")
        self.assertEqual(events.loc[0, "gene"], "TP53")


if __name__ == "__main__":
    unittest.main()
```

