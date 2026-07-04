"""Leakage-free preprocessing built on scikit-learn ``Pipeline``.

Feature types are detected automatically from the dataframe dtypes so the
pipeline generalises if new columns are added. All fitting happens inside the
``Pipeline``/``ColumnTransformer`` and is therefore confined to training folds
during cross-validation -- no statistic is ever computed on test data.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_feature_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_features, categorical_features) inferred from dtypes."""
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = x.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric, categorical


def build_preprocessor(
    x: pd.DataFrame,
    scale: bool = True,
    ohe: bool = True,
) -> ColumnTransformer:
    """Construct a ColumnTransformer.

    Parameters
    ----------
    scale : apply StandardScaler to numeric features (needed by linear / SVM /
        KNN / MLP models, harmless but unnecessary for trees).
    ohe : one-hot encode categoricals. When ``False`` an ordinal-style pass
        through is *not* provided here -- tree models use the same OHE for a
        fully leakage-free, comparable feature space.
    """
    numeric, categorical = detect_feature_types(x)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # Handle both new (sparse_output) and old (sparse) sklearn signatures.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if categorical and ohe:
        transformers.append(("cat", categorical_pipe, categorical))

    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Best-effort recovery of output feature names after fitting."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover
        return []
