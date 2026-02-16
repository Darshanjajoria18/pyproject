"""Helper functions for dataset analysis and guided cleaning operations."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NUMERIC_TYPES = {"int64", "float64", "int32", "float32", "int16", "float16"}


def get_basic_info(df: pd.DataFrame) -> Dict[str, object]:
    """Return basic dataframe information used in the overview step."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def detect_column_type(series: pd.Series) -> str:
    """Classify a column into numeric, categorical, datetime, or other."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Check if object column can reasonably be interpreted as datetime.
    if pd.api.types.is_object_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return "datetime"
        return "categorical"

    if pd.api.types.is_categorical_dtype(series):
        return "categorical"

    return "other"


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-column missing value metrics used in step 2."""
    rows: List[Dict[str, object]] = []
    total_rows = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        missing_pct = float(series.isna().sum()) / total_rows * 100
        col_type = detect_column_type(series)
        skewness = np.nan
        if col_type == "numeric":
            non_null = series.dropna()
            if len(non_null) > 2:
                skewness = float(non_null.skew())
        rows.append(
            {
                "column": col,
                "missing_percent": round(missing_pct, 2),
                "detected_type": col_type,
                "skewness": skewness,
            }
        )
    return pd.DataFrame(rows)


def recommend_missing_strategy(series: pd.Series, detected_type: str, skewness: float) -> str:
    """Return recommended imputation approach for a column."""
    if detected_type == "numeric":
        if pd.notna(skewness) and abs(skewness) > 1:
            return "median"
        return "mean"
    if detected_type in {"categorical", "datetime"}:
        return "mode"
    return "drop"


def apply_missing_fix(df: pd.DataFrame, column: str, strategy: str) -> Tuple[pd.DataFrame, str]:
    """Apply selected missing-value fix to a single column and return note."""
    updated = df.copy()
    note = ""
    if strategy == "mean" and pd.api.types.is_numeric_dtype(updated[column]):
        val = updated[column].mean()
        updated[column] = updated[column].fillna(val)
        note = f"Filled missing values in '{column}' with mean ({val:.3f})."
    elif strategy == "median" and pd.api.types.is_numeric_dtype(updated[column]):
        val = updated[column].median()
        updated[column] = updated[column].fillna(val)
        note = f"Filled missing values in '{column}' with median ({val:.3f})."
    elif strategy == "mode":
        mode_vals = updated[column].mode(dropna=True)
        if not mode_vals.empty:
            val = mode_vals.iloc[0]
            updated[column] = updated[column].fillna(val)
            note = f"Filled missing values in '{column}' with mode ({val})."
    elif strategy == "drop":
        before = len(updated)
        updated = updated[updated[column].notna()].copy()
        removed = before - len(updated)
        note = f"Dropped {removed} rows with missing values in '{column}'."

    if not note:
        note = f"No changes were applied to '{column}'."
    return updated, note


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def apply_log_transform(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    """Apply log1p transform when possible."""
    updated = df.copy()
    min_val = updated[column].min(skipna=True)
    if pd.isna(min_val):
        return updated, f"Skipped log transform for '{column}' because values are empty."

    # Shift if negatives exist so log1p can be used safely.
    shift = 0.0
    if min_val <= -1:
        shift = abs(min_val) + 1.0

    updated[column] = np.log1p(updated[column] + shift)
    if shift:
        return updated, f"Applied log1p transform to '{column}' after shifting by {shift:.3f}."
    return updated, f"Applied log1p transform to '{column}'."


def iqr_outlier_bounds(series: pd.Series) -> Tuple[float, float]:
    """Return lower and upper outlier bounds using IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(lower), float(upper)


def count_outliers(series: pd.Series) -> int:
    """Count outliers in a numeric series using IQR bounds."""
    non_null = series.dropna()
    if non_null.empty:
        return 0
    lower, upper = iqr_outlier_bounds(non_null)
    return int(((non_null < lower) | (non_null > upper)).sum())


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, str]:
    """Remove rows that are outliers in any selected numeric column."""
    updated = df.copy()
    if not columns:
        return updated, "No numeric columns selected for outlier removal."

    mask = pd.Series(True, index=updated.index)
    for col in columns:
        non_null = updated[col].dropna()
        if non_null.empty:
            continue
        lower, upper = iqr_outlier_bounds(non_null)
        col_mask = updated[col].isna() | ((updated[col] >= lower) & (updated[col] <= upper))
        mask &= col_mask

    before = len(updated)
    updated = updated[mask].copy()
    removed = before - len(updated)
    return updated, f"Removed {removed} rows flagged as outliers using IQR across selected columns."


def compute_health_score(df: pd.DataFrame) -> Tuple[int, Dict[str, float]]:
    """Compute a simple 0-100 health score from missingness and outlier burden."""
    if df.empty:
        return 0, {"missing_component": 0.0, "outlier_component": 0.0}

    total_cells = df.shape[0] * df.shape[1]
    missing_ratio = (df.isna().sum().sum() / total_cells) if total_cells else 1.0
    missing_component = max(0.0, 100.0 * (1.0 - missing_ratio))

    num_cols = numeric_columns(df)
    if not num_cols:
        outlier_component = 100.0
    else:
        outlier_counts = sum(count_outliers(df[c]) for c in num_cols)
        total_numeric_cells = max(df.shape[0] * len(num_cols), 1)
        outlier_ratio = outlier_counts / total_numeric_cells
        outlier_component = max(0.0, 100.0 * (1.0 - min(outlier_ratio, 1.0)))

    score = int(round(0.7 * missing_component + 0.3 * outlier_component))
    return score, {
        "missing_component": round(missing_component, 2),
        "outlier_component": round(outlier_component, 2),
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode dataframe as UTF-8 CSV bytes for download button."""
    return df.to_csv(index=False).encode("utf-8")


def summary_to_txt_bytes(text: str) -> bytes:
    """Encode summary text as UTF-8 bytes for download button."""
    return text.encode("utf-8")
