"""Error analysis (Step 9): characterise the samples a model gets wrong."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import config as C


def build_error_frame(x_test: pd.DataFrame, y_test: pd.Series,
                      y_pred: np.ndarray, y_proba=None) -> pd.DataFrame:
    """Return the test frame annotated with prediction outcome."""
    df = x_test.copy()
    df["y_true"] = y_test.values
    df["y_pred"] = np.asarray(y_pred)
    if y_proba is not None:
        df["p_bad"] = np.asarray(y_proba)
    df["correct"] = df["y_true"] == df["y_pred"]
    df["error_type"] = "correct"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "error_type"] = "false_negative"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "error_type"] = "false_positive"
    return df


def error_by_fruit(err_df: pd.DataFrame) -> pd.DataFrame:
    g = err_df.groupby("Fruit").agg(
        n=("correct", "size"),
        n_errors=("correct", lambda s: int((~s).sum())),
    )
    g["error_rate"] = (g["n_errors"] / g["n"]).round(4)
    return g.reset_index().sort_values("error_rate", ascending=False)


def error_feature_profile(err_df: pd.DataFrame) -> pd.DataFrame:
    """Compare feature means of correct vs incorrect predictions."""
    prof = err_df.groupby("correct")[C.NUMERIC_FEATURES].agg(["mean", "std"]).round(3)
    return prof


def borderline_analysis(err_df: pd.DataFrame, low=0.4, high=0.6) -> Dict:
    """Fraction of errors falling in the probability decision band."""
    if "p_bad" not in err_df:
        return {}
    errors = err_df[~err_df["correct"]]
    if len(errors) == 0:
        return {"n_errors": 0, "n_borderline": 0, "pct_borderline": 0.0}
    band = errors[(errors["p_bad"] >= low) & (errors["p_bad"] <= high)]
    return {
        "n_errors": int(len(errors)),
        "n_borderline": int(len(band)),
        "pct_borderline": round(100 * len(band) / len(errors), 2),
        "mean_error_confidence": round(float(np.abs(errors["p_bad"] - 0.5).mean() + 0.5), 4),
    }


def misclassified_samples(err_df: pd.DataFrame) -> pd.DataFrame:
    cols = C.FEATURES + ["y_true", "y_pred", "error_type"]
    if "p_bad" in err_df:
        cols.append("p_bad")
    return err_df.loc[~err_df["correct"], cols].reset_index(drop=True)
