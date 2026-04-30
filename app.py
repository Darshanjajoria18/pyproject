"""Dataset Doctor – Guided Data Cleaning Tutor (Streamlit app)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis_helpers import (
    apply_log_transform,
    apply_missing_fix,
    compute_health_score,
    count_outliers,
    dataframe_to_csv_bytes,
    detect_column_type,
    get_basic_info,
    missing_value_report,
    numeric_columns,
    recommend_missing_strategy,
    remove_outliers_iqr,
    summary_to_txt_bytes,
)
from explanation_engine import (
    build_cleaning_summary,
    current_vs_ideal,
    dataset_overview_text,
    health_score_explanation,
    missing_values_intro,
    outlier_intro,
    skewness_explanation,
    strategy_explanation,
    why_types_matter_text,
)


st.set_page_config(page_title="Dataset Doctor – Guided Data Cleaning Tutor", layout="wide")
st.title("🩺 Dataset Doctor – Guided Data Cleaning Tutor")
st.write("Upload a CSV and clean it step by step with beginner-friendly guidance.")


# Session-state bootstrapping
if "step" not in st.session_state:
    st.session_state.step = 1
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "change_log" not in st.session_state:
    st.session_state.change_log = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""


uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
if not uploaded:
    st.info("Please upload a CSV file to begin.")
    st.stop()


raw_df = pd.read_csv(uploaded)
if st.session_state.clean_df is None or st.session_state.get("uploaded_name") != uploaded.name:
    st.session_state.clean_df = raw_df.copy()
    st.session_state.step = 1
    st.session_state.change_log = []
    st.session_state.summary_text = ""
    st.session_state.uploaded_name = uploaded.name

clean_df = st.session_state.clean_df

# High-level dataset info (always visible)
st.header("Dataset Snapshot")
info = get_basic_info(clean_df)
col1, col2 = st.columns(2)
col1.metric("Rows", info["rows"])
col2.metric("Columns", info["columns"])
st.write("Column names:", info["column_names"])
dtype_df = pd.DataFrame(
    {"column": list(info["dtypes"].keys()), "dtype": list(info["dtypes"].values())}
)
st.dataframe(dtype_df, use_container_width=True)
st.dataframe(clean_df.head(), use_container_width=True)


# Step 1: Overview
if st.session_state.step >= 1:
    st.header("Step 1: Dataset Overview")
    st.write(dataset_overview_text(info["rows"], info["columns"]))
    with st.expander("Why row/column count and data types matter"):
        st.write(why_types_matter_text())

    if st.session_state.step == 1 and st.button("Proceed to Missing Value Analysis"):
        st.session_state.step = 2
        st.rerun()


# Step 2: Missing value analysis
if st.session_state.step >= 2:
    st.header("Step 2: Missing Value Analysis")
    st.write(missing_values_intro())

    report = missing_value_report(clean_df)
    st.dataframe(report, use_container_width=True)

    for _, row in report.iterrows():
        col_name = row["column"]
        detected_type = row["detected_type"]
        missing_pct = float(row["missing_percent"])
        skewness = row["skewness"]

        with st.expander(f"Column: {col_name}"):
            st.write(f"Detected type: **{detected_type}**")
            st.write(f"Missing values: **{missing_pct:.2f}%**")

            # Histogram for numeric columns
            if detected_type == "numeric":
                fig, ax = plt.subplots()
                clean_df[col_name].dropna().hist(ax=ax, bins=20)
                ax.set_title(f"Histogram of {col_name}")
                ax.set_xlabel(col_name)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            strategy = recommend_missing_strategy(clean_df[col_name], detected_type, skewness)
            st.write("Recommended fix:", f"**{strategy}**")
            st.write(strategy_explanation(detected_type, skewness, strategy))
            st.write(current_vs_ideal(missing_pct, strategy))

            if missing_pct > 0:
                choice = st.radio(
                    f"Apply this recommendation for '{col_name}'?",
                    ["Keep as is", f"Apply {strategy}"],
                    key=f"mv_choice_{col_name}",
                )
                if choice != "Keep as is" and st.button(
                    f"Confirm missing-value fix for {col_name}", key=f"mv_apply_{col_name}"
                ):
                    updated, note = apply_missing_fix(clean_df, col_name, strategy)
                    st.session_state.clean_df = updated
                    st.session_state.change_log.append(note)
                    st.success(note)
                    st.rerun()

    if st.session_state.step == 2 and st.button("Proceed to Distribution & Skewness"):
        st.session_state.step = 3
        st.rerun()


# Step 3: Distribution and skewness
if st.session_state.step >= 3:
    st.header("Step 3: Distribution & Skewness")
    num_cols = numeric_columns(clean_df)
    if not num_cols:
        st.info("No numeric columns found for distribution analysis.")
    else:
        for col in num_cols:
            skewness = clean_df[col].dropna().skew() if clean_df[col].dropna().shape[0] > 2 else 0.0
            with st.expander(f"Numeric column: {col}"):
                fig, ax = plt.subplots()
                clean_df[col].dropna().hist(ax=ax, bins=20)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
                st.write(f"Skewness: **{skewness:.3f}**")
                st.write(skewness_explanation(skewness))

                transform_choice = st.selectbox(
                    f"Optional transformation for {col}",
                    ["No transformation", "Apply log transform"],
                    key=f"transform_choice_{col}",
                )
                if transform_choice == "Apply log transform" and st.button(
                    f"Confirm transform for {col}", key=f"transform_apply_{col}"
                ):
                    updated, note = apply_log_transform(clean_df, col)
                    st.session_state.clean_df = updated
                    st.session_state.change_log.append(note)
                    st.success(note)
                    st.rerun()

    if st.session_state.step == 3 and st.button("Proceed to Outlier Detection"):
        st.session_state.step = 4
        st.rerun()


# Step 4: Outlier detection
if st.session_state.step >= 4:
    st.header("Step 4: Outlier Detection (IQR)")
    st.write(outlier_intro())
    num_cols = numeric_columns(clean_df)

    if not num_cols:
        st.info("No numeric columns available for outlier detection.")
    else:
        outlier_table = pd.DataFrame(
            {"column": num_cols, "outlier_count": [count_outliers(clean_df[c]) for c in num_cols]}
        )
        st.dataframe(outlier_table, use_container_width=True)

        outlier_action = st.radio(
            "Would you like to remove outliers?",
            ["Keep outliers", "Remove outliers with IQR"],
            key="outlier_action",
        )

        if outlier_action == "Remove outliers with IQR" and st.button("Confirm outlier removal"):
            updated, note = remove_outliers_iqr(clean_df, num_cols)
            st.session_state.clean_df = updated
            st.session_state.change_log.append(note)
            st.success(note)
            st.rerun()

    if st.session_state.step == 4 and st.button("Finish and Generate Final Outputs"):
        st.session_state.step = 5
        st.rerun()


# Final outputs
if st.session_state.step >= 5:
    st.header("Final Outputs")
    final_df = st.session_state.clean_df
    st.subheader("Cleaned Dataset Preview")
    st.dataframe(final_df.head(20), use_container_width=True)

    score, components = compute_health_score(final_df)
    score_text = health_score_explanation(score, components)
    st.subheader("Dataset Health Score")
    st.write(score_text)

    summary = build_cleaning_summary(st.session_state.change_log, score_text)
    st.session_state.summary_text = summary

    st.subheader("Cleaning Summary")
    st.write(summary)

    csv_bytes = dataframe_to_csv_bytes(final_df)
    txt_bytes = summary_to_txt_bytes(summary)

    st.download_button(
        "Download cleaned dataset (CSV)",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download cleaning summary (TXT)",
        data=txt_bytes,
        file_name="cleaning_summary.txt",
        mime="text/plain",
    )
