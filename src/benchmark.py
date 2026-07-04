"""Training, cross-validation, and held-out evaluation of the model zoo."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from . import config as C
from .metrics import METRIC_ORDER, compute_metrics, get_scores
from .models import get_model_zoo


def _ci95(values: np.ndarray):
    """95% confidence interval of the mean via Student-t."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return (mean - h, mean + h)


def cross_validate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict | None = None,
    n_splits: int = C.CV_FOLDS,
) -> pd.DataFrame:
    """Stratified k-fold CV for every model; per-fold metrics -> mean/std/CI."""
    if models is None:
        models = get_model_zoo(x_train)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE
    )
    rows: List[Dict] = []

    x_arr = x_train.reset_index(drop=True)
    y_arr = y_train.reset_index(drop=True)

    for name, pipe in models.items():
        fold_metrics: Dict[str, List[float]] = {m: [] for m in METRIC_ORDER}
        for tr_idx, va_idx in skf.split(x_arr, y_arr):
            x_tr, x_va = x_arr.iloc[tr_idx], x_arr.iloc[va_idx]
            y_tr, y_va = y_arr.iloc[tr_idx], y_arr.iloc[va_idx]
            est = clone(pipe)
            est.fit(x_tr, y_tr)
            y_pred = est.predict(x_va)
            y_proba, y_score = get_scores(est, x_va)
            m = compute_metrics(y_va, y_pred, y_proba, y_score)
            for k in METRIC_ORDER:
                fold_metrics[k].append(m[k])

        row = {"Model": name}
        for k in METRIC_ORDER:
            arr = np.array(fold_metrics[k], dtype=float)
            lo, hi = _ci95(arr)
            row[f"{k}_mean"] = np.nanmean(arr)
            row[f"{k}_std"] = np.nanstd(arr, ddof=1) if np.sum(~np.isnan(arr)) > 1 else np.nan
            row[f"{k}_ci_low"] = lo
            row[f"{k}_ci_high"] = hi
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_on_test(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict | None = None,
):
    """Fit each model on the full training set and evaluate on the test set.

    Returns (results_df, fitted_models, predictions) where predictions maps
    model name -> dict(y_pred, y_proba, y_score) for downstream analysis.
    """
    if models is None:
        models = get_model_zoo(x_train)

    rows: List[Dict] = []
    fitted: Dict = {}
    predictions: Dict = {}

    for name, pipe in models.items():
        est = clone(pipe)

        t0 = time.perf_counter()
        est.fit(x_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = est.predict(x_test)
        predict_time = time.perf_counter() - t0

        y_proba, y_score = get_scores(est, x_test)
        m = compute_metrics(y_test, y_pred, y_proba, y_score)
        m["Model"] = name
        m["Train Time (s)"] = train_time
        m["Predict Time (s)"] = predict_time
        rows.append(m)

        fitted[name] = est
        predictions[name] = {
            "y_pred": np.asarray(y_pred),
            "y_proba": None if y_proba is None else np.asarray(y_proba),
            "y_score": None if y_score is None else np.asarray(y_score),
        }

    cols = ["Model"] + METRIC_ORDER + ["Train Time (s)", "Predict Time (s)"]
    results = pd.DataFrame(rows)[cols].sort_values(
        "F1", ascending=False
    ).reset_index(drop=True)
    return results, fitted, predictions
