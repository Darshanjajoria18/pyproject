"""Text generation helpers for beginner-friendly tutoring explanations."""

from __future__ import annotations

from typing import Dict, List


def dataset_overview_text(rows: int, cols: int) -> str:
    return (
        f"Your dataset has {rows} rows and {cols} columns. "
        "Rows are individual records (like people, products, or events), and columns are the "
        "features that describe each record. Knowing the shape helps us estimate how much data we "
        "have and whether a cleaning step might remove too much information."
    )


def why_types_matter_text() -> str:
    return (
        "Data types tell us what each column represents. Numeric columns are good for averages and "
        "distributions, categorical columns represent labels, and datetime columns represent time. "
        "Using the wrong method on the wrong type can create misleading results."
    )


def missing_values_intro() -> str:
    return (
        "Missing values are blank or unknown entries. They matter because many models and reports "
        "expect complete data. If we ignore missing values, calculations can become biased or fail."
    )


def strategy_explanation(col_type: str, skewness: float, strategy: str) -> str:
    if col_type == "numeric":
        if strategy == "median":
            return (
                "This numeric column is skewed (values lean heavily to one side), so the median is "
                "usually safer than the mean because it is less affected by extreme values."
            )
        return (
            "This numeric column is fairly balanced, so using the mean is usually a reasonable way "
            "to fill missing values while keeping the center of the data similar."
        )
    if col_type == "categorical":
        return (
            "This is a categorical column, so the mode (most common category) is a simple and "
            "beginner-friendly way to fill missing entries."
        )
    if col_type == "datetime":
        return (
            "This appears to be a datetime column. A common beginner approach is to fill with the "
            "most common timestamp (mode), while noting that business context may suggest a better "
            "time-based method."
        )
    return "No strong recommendation is available for this column type, so we skip automatic filling."


def current_vs_ideal(current_missing_pct: float, strategy: str) -> str:
    return (
        f"Current state: {current_missing_pct:.2f}% values are missing in this column. "
        f"Ideal state: close to 0% missing values, achieved here by using '{strategy}'."
    )


def skewness_explanation(skewness: float) -> str:
    if skewness > 1:
        return "This column is right-skewed: most values are small, with a long tail of large values."
    if skewness < -1:
        return "This column is left-skewed: most values are large, with a tail of smaller values."
    return "This column is roughly balanced (not strongly skewed)."


def outlier_intro() -> str:
    return (
        "Outliers are values that are much higher or lower than most other values. They can be real, "
        "but they may also come from data entry errors. Outliers can strongly affect averages and "
        "some machine learning models."
    )


def health_score_explanation(score: int, components: Dict[str, float]) -> str:
    return (
        f"Health score: {score}/100. This score combines missing-value quality "
        f"({components['missing_component']}/100) and numeric outlier quality "
        f"({components['outlier_component']}/100). Higher is better."
    )


def build_cleaning_summary(change_log: List[str], score_text: str) -> str:
    if not change_log:
        changes = "No cleaning changes were applied."
    else:
        changes = "\n".join(f"- {item}" for item in change_log)

    return (
        "Dataset Doctor – Cleaning Summary\n\n"
        "What was cleaned:\n"
        f"{changes}\n\n"
        "Why these changes were recommended:\n"
        "- Missing values were handled to reduce blank entries and improve reliability.\n"
        "- Distribution checks were used to identify skew and optional transformations.\n"
        "- Outlier review was included because extreme values can distort analysis.\n\n"
        "How the dataset is improved:\n"
        "- The data is more complete and typically easier to analyze or model.\n"
        f"- {score_text}\n"
    )
