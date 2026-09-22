"""
Generic, dataset-agnostic preprocessing.

Given ANY tabular CSV, this module:
  1. Detects the target column (a name hint, else the last non-ID-like column)
  2. Detects the task type (classification vs regression) from the target's values
  3. Cleans the dataframe (drops empty/near-empty columns, coerces numeric-looking
     strings, drops ID-like columns) and builds a model-ready feature matrix.

No column names or dataset shape are hardcoded anywhere in this file.
"""
import numpy as np
import pandas as pd

import re

ID_LIKE_TOKENS = ("id", "uuid", "guid", "index")
TARGET_NAME_HINTS = {"target", "label", "class", "outcome", "churn", "result", "prediction", "y", "output"}


def _tokenize(name: str):
    """Split a column name into lowercase word tokens, e.g. 'age_years' -> ['age', 'years']."""
    return [t for t in re.split(r"[^a-z0-9]+", name.lower().strip()) if t]


def _looks_like_id(series: pd.Series) -> bool:
    """A column is ID-like if every value is unique and it isn't numeric-continuous."""
    if len(series) <= 1:
        return False
    if series.dtype == object or str(series.dtype).startswith("string") or str(series.dtype) == "category":
        nunique = series.nunique(dropna=True)
        return nunique == len(series)
    return False


def detect_target_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    tokens = {c: _tokenize(c) for c in cols}

    # exact full-name match first (e.g. column literally called "Churn" or "y")
    for c in cols:
        if c.lower().strip() in TARGET_NAME_HINTS:
            return c
    # word-boundary match (e.g. "customer_churn" -> token "churn"), never raw substring
    # (raw substring matching would let a short hint like "y" false-match inside "category")
    for c in cols:
        if any(hint in tokens[c] for hint in TARGET_NAME_HINTS):
            return c
    # fallback: last column that isn't ID-like
    for c in reversed(cols):
        if not _looks_like_id(df[c]):
            return c
    return cols[-1]


def detect_task_type(y: pd.Series) -> str:
    y_non_null = y.dropna()
    if pd.api.types.is_numeric_dtype(y_non_null):
        nunique = y_non_null.nunique()
        if nunique <= 2:
            return "classification"  # binary (0/1, True/False) is always classification
        is_integer_like = bool(np.allclose(y_non_null % 1, 0))
        # few distinct, whole-number values relative to row count => treat as classes (e.g. 1-5 stars)
        if is_integer_like and nunique <= 20 and (nunique / max(len(y_non_null), 1)) <= 0.2:
            return "classification"
        return "regression"
    return "classification"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    if len(df) > 0:
        df = df.loc[:, df.isna().mean() < 0.6]  # drop columns that are >60% missing

    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].astype(str).str.strip()
            candidate = pd.to_numeric(stripped.replace("", np.nan), errors="coerce")
            # only convert if the vast majority of non-null values parse as numbers
            if stripped.replace("", np.nan).notna().sum() > 0 and candidate.notna().mean() > 0.9:
                df[col] = candidate
    return df


def build_features(df: pd.DataFrame, target_col: str):
    """
    Returns: X (DataFrame, fully numeric), y (Series, numeric/encoded),
             task_type (str), label_map (dict|None mapping encoded->original class),
             dropped_columns (list[str])
    """
    df = clean_dataframe(df)
    if target_col not in df.columns:
        target_col = detect_target_column(df)

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    dropped_columns = [
        c for c in X.columns
        if _looks_like_id(X[c]) or any(tok in c.lower() for tok in ID_LIKE_TOKENS)
    ]
    X = X.drop(columns=dropped_columns)

    task_type = detect_task_type(y_raw)

    label_map = None
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        y_cat = y_raw.astype("category")
        label_map = dict(enumerate(y_cat.cat.categories))
        y = y_cat.cat.codes.astype(int)
        y = y.where(y_cat.notna(), other=np.nan)
    else:
        y = y_raw

    # keep only rows where target is present
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    for col in list(X.columns):
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        elif pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        else:
            mode = X[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else "missing"
            X[col] = X[col].fillna(fill_val)

            nunique = X[col].nunique()
            if nunique <= 12:
                dummies = pd.get_dummies(X[col].astype(str), prefix=col)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                # high-cardinality categorical: safe fallback encoding (no explosion)
                X[col] = X[col].astype("category").cat.codes

    X = X.fillna(0)

    if task_type == "classification":
        y = y.astype(int)

    return X, y, task_type, label_map, dropped_columns
