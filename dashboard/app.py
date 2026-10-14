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
