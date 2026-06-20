import os
import streamlit as st
import pandas as pd
from tools.io_utils import smart_read_csv

st.set_page_config(page_title="InsightForge", layout="wide")

st.title("InsightForge")
st.caption("Multi-Agent AI System for Automated Business Data Analysis & Decision Support")
st.divider()

with st.sidebar:
    st.header("Dataset Settings")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    target_column = st.text_input(
        "Target Column",
        placeholder="e.g. Survived, Price, Churn (leave empty for clustering)",
    )

    st.info(
        "Leave **Target Column** empty if you want the agents to perform clustering "
        "automatically."
    )

    st.divider()
    st.subheader("Dataset Description (optional)")
    st.caption(
        "For better analysis, you can provide documentation about your dataset — "
        "e.g. what missing values look like (such as -200), units, or column meanings. "
        "The Planner agent will read it before deciding."
    )
    description_file = st.file_uploader(
        "Upload a description file (PDF or TXT)", type=["pdf", "txt"]
    )
    description_text = st.text_area(
        "...or paste the description here",
        placeholder="e.g. Missing values are tagged with -200.",
        height=100,
    )

    run_btn = st.button(
        "Run Analysis",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )


if uploaded_file:
    df_preview = smart_read_csv(uploaded_file)
    uploaded_file.seek(0)  # reset so we can save it again below

    st.subheader("Dataset Preview")
    st.dataframe(df_preview.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_preview.shape[0])
    c2.metric("Columns", df_preview.shape[1])
    c3.metric("Missing Values", int(df_preview.isnull().sum().sum()))
    st.divider()


if run_btn and uploaded_file:
    os.makedirs("data", exist_ok=True)

    # Remove stale outputs from a previous run so old plots aren't shown
    for stale in ("data/feature_importance.png", "data/correlation_heatmap.png",
                  "data/confusion_matrix.png", "data/model_results.json",
                  "data/description.txt", "data/description.pdf"):
        if os.path.exists(stale):
            os.remove(stale)

    save_path = "data/uploaded.csv"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    
    document_path = ""
    if description_file is not None:
        ext = ".pdf" if description_file.name.lower().endswith(".pdf") else ".txt"
        document_path = f"data/description{ext}"
        with open(document_path, "wb") as f:
            f.write(description_file.read())
    elif description_text.strip():
        document_path = "data/description.txt"
        with open(document_path, "w", encoding="utf-8") as f:
            f.write(description_text.strip())

    with st.status("Running multi-agent analysis…", expanded=True) as status:
        st.write("Planner agent: inspecting dataset…")
        if document_path:
            st.write("Planner agent: reading dataset description…")
        try:
            from pipeline import run_pipeline
            outputs = run_pipeline(save_path, target_column.strip(), document_path)
            status.update(label="Analysis complete!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Error: {exc}")
            st.stop()

    st.success("All agents finished. See results below.")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Planning", "Data Preparation", "Modeling & Insights", "Evaluation Report"]
    )

    with tab1:
        st.subheader("Planner Agent")
        st.write(outputs["planner"] or "_No output returned._")

    with tab2:
        st.subheader("Data Preparation Agent")
        st.write(outputs["data_prep"] or "_No output returned._")
        cleaned_path = "data/cleaned.csv"
        if os.path.exists(cleaned_path):
            st.subheader("Cleaned Dataset Preview")
            st.dataframe(smart_read_csv(cleaned_path).head(10), use_container_width=True)

    with tab3:
        st.subheader("Modeling & Insight Agent")
        st.write(outputs["modeling"] or "_No output returned._")

        plot_path = "data/feature_importance.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Feature Importance / Cluster Visualization")

        heatmap_path = "data/correlation_heatmap.png"
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Feature Correlation Heatmap")

        confusion_path = "data/confusion_matrix.png"
        if os.path.exists(confusion_path):
            st.image(confusion_path, caption="Confusion Matrix")

    with tab4:
        st.subheader("Evaluator Agent")
        st.write(outputs["evaluator"] or "_No output returned._")