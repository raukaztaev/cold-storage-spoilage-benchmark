"""Data loading, cleaning, and splitting utilities.

The cleaning logic mirrors and extends the preprocessing described in the
original paper: header repair, label normalisation ("BAD"/"Bad" -> "Bad"),
and exact-duplicate removal. Positive class is the spoilage-risk ("Bad") label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


@dataclass
class CleaningReport:
    """Structured summary of the cleaning process (for auditing / the paper)."""

    raw_rows: int = 0
    raw_columns: int = 0
    raw_class_counts: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    clean_rows: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    fruit_counts: Dict[str, int] = field(default_factory=dict)
    missing_values: int = 0

    def to_dict(self) -> Dict:
        return {
            "raw_rows": self.raw_rows,
            "raw_columns": self.raw_columns,
            "raw_class_counts": self.raw_class_counts,
            "duplicates_removed": self.duplicates_removed,
            "clean_rows": self.clean_rows,
            "class_counts": self.class_counts,
            "fruit_counts": self.fruit_counts,
            "missing_values": self.missing_values,
        }


def load_raw() -> pd.DataFrame:
    """Load the raw CSV exactly as downloaded."""
    return pd.read_csv(C.RAW_DATASET)


def clean_dataset(df: pd.DataFrame | None = None, save: bool = False):
    """Clean the raw dataset and return (clean_df, CleaningReport).

    Steps:
      1. Repair malformed headers.
      2. Normalise the class label: strip whitespace, title-case, so that the
         stray upper-case "BAD" is merged into "Bad".
      3. Drop exact duplicate rows (they would otherwise leak across the
         train/test split and inflate scores).
    """
    if df is None:
        df = load_raw()

    report = CleaningReport(raw_rows=len(df), raw_columns=df.shape[1])

    df = df.rename(columns=C.RAW_COLUMN_RENAME).copy()

    # Label normalisation.
    df[C.TARGET] = df[C.TARGET].astype(str).str.strip().str.capitalize()
    report.raw_class_counts = df[C.TARGET].value_counts().to_dict()

    # Deduplication.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report.duplicates_removed = before - len(df)

    report.clean_rows = len(df)
    report.class_counts = df[C.TARGET].value_counts().to_dict()
    report.fruit_counts = df["Fruit"].value_counts().to_dict()
    report.missing_values = int(df.isna().sum().sum())

    if save:
        df.to_csv(C.CLEAN_DATASET, index=False)

    return df, report


def load_clean(prefer_cache: bool = True) -> pd.DataFrame:
    """Load the cleaned dataset, regenerating it from raw if necessary."""
    if prefer_cache and C.CLEAN_DATASET.exists():
        return pd.read_csv(C.CLEAN_DATASET)
    df, _ = clean_dataset(save=True)
    return df


def encode_target(y: pd.Series) -> pd.Series:
    """Map string labels to integers with positive class (Bad/spoilage) = 1."""
    mapping = {C.NEGATIVE_LABEL: 0, C.POSITIVE_LABEL: 1}
    return y.map(mapping).astype(int)


def get_xy(df: pd.DataFrame):
    """Split a dataframe into feature matrix X and integer target y."""
    x = df[C.FEATURES].copy()
    y = encode_target(df[C.TARGET])
    return x, y


def get_splits(df: pd.DataFrame | None = None, test_size: float = C.TEST_SIZE):
    """Deterministic stratified train/test split shared by every notebook.

    Using a fixed seed guarantees that all notebooks operate on identical
    partitions, so results are directly comparable even when notebooks are run
    independently.
    """
    if df is None:
        df = load_clean()
    x, y = get_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=C.RANDOM_STATE, stratify=y
    )
    return x_train, x_test, y_train, y_test
