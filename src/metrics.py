"""Unified metric computation for binary spoilage classification.

Positive class (label 1) is the spoilage-risk ("Bad") case, so recall here is
the fraction of genuinely unsafe storage states that the model flags -- the
operationally critical quantity for a food-safety alerting system.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_scores(estimator, x) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (y_proba, y_score) for the positive class.

    ``y_proba`` are calibrated probabilities in [0, 1] when the estimator
    exposes ``predict_proba``; otherwise ``None``. ``y_score`` is any monotone
    ranking score usable by ROC/PR-AUC (falls back to ``decision_function``).
    """
    y_proba = None
    y_score = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        y_proba = proba[:, 1]
        y_score = y_proba
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(x)
    return y_proba, y_score


def compute_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full metric suite. Probability-based metrics are NaN when
    calibrated probabilities are unavailable."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }

    ranking = y_score if y_score is not None else y_proba
    if ranking is not None and len(np.unique(y_true)) > 1:
        metrics["ROC AUC"] = roc_auc_score(y_true, ranking)
        metrics["PR AUC"] = average_precision_score(y_true, ranking)
    else:
        metrics["ROC AUC"] = np.nan
        metrics["PR AUC"] = np.nan

    if y_proba is not None:
        eps = 1e-15
        proba_clipped = np.clip(y_proba, eps, 1 - eps)
        metrics["Log Loss"] = log_loss(
            y_true, proba_clipped, labels=[0, 1]
        )
        metrics["Brier Score"] = brier_score_loss(y_true, y_proba)
    else:
        metrics["Log Loss"] = np.nan
        metrics["Brier Score"] = np.nan

    return metrics


METRIC_ORDER = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC AUC",
    "PR AUC",
    "MCC",
    "Cohen Kappa",
    "Log Loss",
    "Brier Score",
]

# Metrics where a larger value is better (used for ranking / formatting).
HIGHER_IS_BETTER = {
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC AUC",
    "PR AUC",
    "MCC",
    "Cohen Kappa",
}
