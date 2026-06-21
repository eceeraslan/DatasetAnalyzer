import os
import json
import uuid
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from tools.io_utils import smart_read_csv

st.set_page_config(
    page_title="InsightForge",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

DATA_DIR = f"data/{st.session_state.session_id}"

# ---------- Dark mode via JS parent-window injection ----------
# Streamlit uses CSS-in-JS (styled-components) so regular st.markdown CSS gets
# overridden after render. We inject into the parent document via an iframe and
# use a MutationObserver to re-append our style after every Streamlit re-render.

_DARK_CSS = """
.stApp                                       { background-color: #0D1117 !important; }
section[data-testid="stSidebar"]             { background-color: #161B22 !important; }
[data-testid="stHeader"]                     { background-color: #0D1117 !important; }
.stApp p, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stApp li,
.stApp span, .stApp label,
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"] *        { color: #E6EDF3 !important; }
[data-testid="stMetricValue"]               { color: #79C0FF !important; }
[data-testid="stMetricLabel"]               { color: #8B949E !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #161B22 !important;
    border-color:     #30363D !important;
}
.stTextInput input, .stTextArea textarea    {
    background-color: #21262D !important;
    color:            #E6EDF3 !important;
    border-color:     #30363D !important;
}
[data-testid="stExpander"]                  { background-color: #161B22 !important; }
[data-testid="stExpander"] *               { color: #E6EDF3 !important; }
[data-testid="stFileUploaderDropzone"]      {
    background-color: #21262D !important;
    border-color:     #30363D !important;
}
[data-testid="stFileUploaderDropzone"] *   { color: #E6EDF3 !important; }
.stTabs [role="tab"]                        { color: #8B949E !important; }
.stTabs [role="tab"][aria-selected="true"]  { color: #79C0FF !important; }
hr                                          { border-color: #30363D !important; }
[data-testid="stAlert"] *                  { color: #E6EDF3 !important; }
"""

def _inject_dark():
    escaped = _DARK_CSS.replace("`", r"\`")
    components.html(f"""
    <script>
    (function() {{
        const CSS = `{escaped}`;
        const doc = window.parent.document;

        function applyDark() {{
            let el = doc.getElementById('__insightforge_dark');
            if (!el) {{
                el = doc.createElement('style');
                el.id = '__insightforge_dark';
            }}
            el.textContent = CSS;
            doc.head.appendChild(el);   // always re-append to stay last
        }}

        applyDark();

        // Re-apply after every Streamlit re-render (styled-components injection)
        new MutationObserver(function(mutations) {{
            for (var m of mutations) {{
                for (var n of m.addedNodes) {{
                    if (n.id !== '__insightforge_dark') {{ applyDark(); return; }}
                }}
            }}
        }}).observe(doc.head, {{ childList: true }});
    }})();
    </script>
    """, height=0)

def _remove_dark():
    components.html("""
    <script>
    (function() {
        var el = window.parent.document.getElementById('__insightforge_dark');
        if (el) el.remove();
    })();
    </script>
    """, height=0)

if st.session_state.dark_mode:
    _inject_dark()
else:
    _remove_dark()

# ---------- Step bar ----------
_STEPS = ["Planner", "Data Preparation", "Modeling", "Evaluation"]

def _render_steps(active: int):
    cols = st.columns(len(_STEPS))
    for i, (col, step) in enumerate(zip(cols, _STEPS)):
        done    = i < active
        current = i == active
        if done:
            col.markdown(
                f'<div style="background:#1B4FCC;color:#fff;text-align:center;'
                f'padding:0.45rem 0.3rem;border-radius:5px;font-size:0.8rem;font-weight:600;">'
                f'{step}</div>', unsafe_allow_html=True)
        elif current:
            col.markdown(
                f'<div style="border:2px solid #1B4FCC;color:#1B4FCC;text-align:center;'
                f'padding:0.45rem 0.3rem;border-radius:5px;font-size:0.8rem;font-weight:700;">'
                f'{step}</div>', unsafe_allow_html=True)
        else:
            col.markdown(
                f'<div style="border:1px solid rgba(128,128,128,0.3);text-align:center;'
                f'padding:0.45rem 0.3rem;border-radius:5px;font-size:0.8rem;opacity:0.4;">'
                f'{step}</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configuration")

    uploaded_file = st.file_uploader("Dataset (CSV)", type=["csv"])

    target_column = st.text_input(
        "Target Column",
        placeholder="e.g. Survived, SalePrice, Churn",
        help="Leave empty to run clustering automatically.",
    )

    st.divider()

    with st.expander("Dataset Description (optional)"):
        st.caption(
            "Provide documentation about your dataset — column meanings, "
            "units, or special missing-value codes (e.g. -200)."
        )
        description_file = st.file_uploader("PDF or TXT file", type=["pdf", "txt"])
        description_text = st.text_area(
            "Or paste description here",
            placeholder="e.g. Missing values are encoded as -200.",
            height=100,
        )

    st.divider()

    run_btn = st.button(
        "Run Analysis",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    st.divider()
    st.toggle("Dark mode", key="dark_mode")

# ---------- Header ----------
st.title("InsightForge")
st.caption("Multi-agent AI system for automated data analysis and machine learning")
st.divider()

# ---------- Landing page ----------
if not uploaded_file:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("**Step 1 — Upload**")
            st.caption(
                "Upload any tabular CSV dataset. Optionally provide a description "
                "document to help the planner understand column conventions and "
                "missing-value encodings."
            )
    with col2:
        with st.container(border=True):
            st.markdown("**Step 2 — Analyze**")
            st.caption(
                "Four specialized agents collaborate: the Planner decides the problem "
                "type, the Data Preparation agent cleans the data, the Modeling agent "
                "trains and compares multiple models, and the Evaluator scores the pipeline."
            )
    with col3:
        with st.container(border=True):
            st.markdown("**Step 3 — Export**")
            st.caption(
                "Download the cleaned dataset, the trained model (.pkl ready for "
                "inference), and a structured evaluation report with quantitative metrics."
            )

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.info("**Classification**  \nPredict a discrete label  \ne.g. Survived, Churn, Category")
    col_b.info("**Regression**  \nPredict a continuous value  \ne.g. SalePrice, Salary, Score")
    col_c.info("**Clustering**  \nDiscover natural groups  \nNo target column needed")

# ---------- Dataset preview ----------
if uploaded_file:
    df_preview = smart_read_csv(uploaded_file)
    uploaded_file.seek(0)

    st.subheader("Dataset Preview")
    st.dataframe(df_preview.head(10), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_preview.shape[0]:,}")
    c2.metric("Columns", df_preview.shape[1])
    c3.metric("Missing Values", f"{int(df_preview.isnull().sum().sum()):,}")
    c4.metric("Numeric Columns", len(df_preview.select_dtypes(include="number").columns))
    st.divider()

# ---------- Pipeline ----------
if run_btn and uploaded_file:
    os.makedirs(DATA_DIR, exist_ok=True)

    for stale_file in (
        "feature_importance.png", "correlation_heatmap.png",
        "confusion_matrix.png", "model_results.json",
        "description.txt", "description.pdf", "model.pkl", "cleaned.csv",
    ):
        stale = os.path.join(DATA_DIR, stale_file)
        if os.path.exists(stale):
            os.remove(stale)

    save_path = os.path.join(DATA_DIR, "uploaded.csv")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    document_path = ""
    if description_file is not None:
        ext = ".pdf" if description_file.name.lower().endswith(".pdf") else ".txt"
        document_path = os.path.join(DATA_DIR, f"description{ext}")
        with open(document_path, "wb") as f:
            f.write(description_file.read())
    elif description_text.strip():
        document_path = os.path.join(DATA_DIR, "description.txt")
        with open(document_path, "w", encoding="utf-8") as f:
            f.write(description_text.strip())

    step_ph = st.empty()

    def _on_step(idx: int):
        with step_ph.container():
            _render_steps(idx)

    _on_step(0)

    with st.status("Running multi-agent analysis…", expanded=True) as status:
        try:
            from pipeline import run_pipeline
            outputs = run_pipeline(save_path, target_column.strip(), document_path, DATA_DIR, _on_step)
            status.update(label="Analysis complete.", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Pipeline failed.", state="error")
            st.error(f"Error: {exc}")
            st.stop()

    with step_ph.container():
        _render_steps(len(_STEPS))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Planning", "Data Preparation", "Modeling & Insights", "Evaluation Report"]
    )

    with tab1:
        st.markdown("##### Planner Agent Report")
        with st.container(border=True):
            st.markdown(outputs["planner"] or "_No output returned._")

    with tab2:
        st.markdown("##### Data Preparation Agent Report")
        with st.container(border=True):
            st.markdown(outputs["data_prep"] or "_No output returned._")

        cleaned_path = os.path.join(DATA_DIR, "cleaned.csv")
        if os.path.exists(cleaned_path):
            st.divider()
            st.markdown("##### Cleaned Dataset")
            cleaned_df = smart_read_csv(cleaned_path)

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Rows", f"{cleaned_df.shape[0]:,}")
            cc2.metric("Columns", cleaned_df.shape[1])
            cc3.metric("Remaining Missing Values", int(cleaned_df.isnull().sum().sum()))

            st.dataframe(cleaned_df.head(10), use_container_width=True)
            st.download_button(
                "Download Cleaned Dataset",
                cleaned_df.to_csv(index=False).encode(),
                file_name="cleaned.csv",
                mime="text/csv",
            )

    with tab3:
        results_path = os.path.join(DATA_DIR, "model_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)

            pt = results.get("problem_type", "")

            if pt in ("classification", "regression"):
                st.markdown("##### Model Performance")
                metric_keys = [
                    "Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)",
                    "R² Score", "RMSE", "MAE",
                ]
                available = [(k, results[k]) for k in metric_keys if k in results]
                cols = st.columns(len(available))
                for i, (k, v) in enumerate(available):
                    cols[i].metric(k, v)

                if "model_leaderboard_cv" in results:
                    st.divider()
                    st.markdown("##### Model Comparison")
                    n_folds = results.get("cv_folds", 5)
                    lb_df = pd.DataFrame(
                        sorted(results["model_leaderboard_cv"].items(), key=lambda x: x[1], reverse=True),
                        columns=["Model", f"CV Score ({n_folds}-fold)"],
                    )
                    st.dataframe(lb_df, use_container_width=True, hide_index=True)

            elif pt == "clustering":
                st.markdown("##### Clustering Results")
                ck1, ck2 = st.columns(2)
                ck1.metric("Optimal Clusters (k)", results.get("optimal_k", "—"))
                ck2.metric("Silhouette Score", results.get("silhouette_score", "—"))

            st.divider()

        st.markdown("##### Modeling Agent Report")
        with st.container(border=True):
            st.markdown(outputs["modeling"] or "_No output returned._")

        fi_path = os.path.join(DATA_DIR, "feature_importance.png")
        hm_path = os.path.join(DATA_DIR, "correlation_heatmap.png")
        cm_path = os.path.join(DATA_DIR, "confusion_matrix.png")
        has_fi, has_hm, has_cm = os.path.exists(fi_path), os.path.exists(hm_path), os.path.exists(cm_path)

        if has_fi or has_hm or has_cm:
            st.divider()
            st.markdown("##### Visualizations")
            if has_fi or has_hm:
                col_l, col_r = st.columns(2)
                if has_fi:
                    col_l.image(fi_path, caption="Feature Importance", use_container_width=True)
                if has_hm:
                    col_r.image(hm_path, caption="Feature Correlation Heatmap", use_container_width=True)
            if has_cm:
                _, cm_col, _ = st.columns([2, 3, 2])
                cm_col.image(cm_path, caption="Confusion Matrix", use_container_width=True)

        model_path = os.path.join(DATA_DIR, "model.pkl")
        if os.path.exists(model_path):
            st.divider()
            with open(model_path, "rb") as f:
                st.download_button(
                    "Download Trained Model (.pkl)",
                    f,
                    file_name="model.pkl",
                    mime="application/octet-stream",
                )

    with tab4:
        st.markdown("##### Evaluation Report")
        with st.container(border=True):
            st.markdown(outputs["evaluator"] or "_No output returned._")