"""Statistical comparison of classifiers (Step 10).

- McNemar's test: pairwise comparison of two models on the same test set.
- Friedman test: omnibus test across all models over CV folds.
- Nemenyi post-hoc: pairwise critical-difference comparison after Friedman.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

from . import config as C
from .metrics import get_scores


def mcnemar_pairwise(y_true, predictions: Dict[str, Dict], names: List[str]) -> pd.DataFrame:
    """McNemar test for every model pair. Uses exact test for small discordance."""
    y_true = np.asarray(y_true)
    rows = []
    for a, b in combinations(names, 2):
        pa = np.asarray(predictions[a]["y_pred"])
        pb = np.asarray(predictions[b]["y_pred"])
        a_correct = pa == y_true
        b_correct = pb == y_true
        n01 = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
        n10 = int(np.sum(~a_correct & b_correct))  # a wrong, b right
        table = [[int(np.sum(a_correct & b_correct)), n01],
                 [n10, int(np.sum(~a_correct & ~b_correct))]]
        exact = (n01 + n10) < 25
        res = mcnemar(table, exact=exact, correction=True)
        rows.append({
            "Model A": a, "Model B": b,
            "b (A right, B wrong)": n01, "c (A wrong, B right)": n10,
            "statistic": round(float(res.statistic), 4),
            "p_value": float(res.pvalue),
            "significant_0.05": bool(res.pvalue < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def cv_score_matrix(x, y, models: Dict, scoring_fn, n_splits: int = C.CV_FOLDS) -> pd.DataFrame:
    """Return a folds x models matrix of a chosen score for Friedman/Nemenyi."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    data: Dict[str, List[float]] = {name: [] for name in models}
    for tr, va in skf.split(x, y):
        for name, pipe in models.items():
            est = clone(pipe)
            est.fit(x.iloc[tr], y.iloc[tr])
            pred = est.predict(x.iloc[va])
            data[name].append(scoring_fn(y.iloc[va], pred))
    return pd.DataFrame(data)


def friedman_test(score_matrix: pd.DataFrame) -> Dict:
    """Friedman omnibus test over the folds x models score matrix."""
    arrays = [score_matrix[c].values for c in score_matrix.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p),
            "significant_0.05": bool(p < 0.05),
            "n_models": score_matrix.shape[1], "n_folds": score_matrix.shape[0]}


def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi test (studentized range approx)."""
    # q_alpha values (Tukey/studentized range / sqrt(2)) for alpha=0.05.
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
           14: 3.354, 15: 3.391, 16: 3.426}
    q = q05.get(k, 3.426)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi_posthoc(score_matrix: pd.DataFrame) -> Dict:
    """Average ranks + Nemenyi critical difference; higher score = better."""
    # Rank per fold (1 = best). Negate so that argsort gives best first.
    ranks = (-score_matrix).rank(axis=1, method="average")
    avg_ranks = ranks.mean(axis=0)
    k = score_matrix.shape[1]
    n = score_matrix.shape[0]
    cd = _nemenyi_cd(k, n)

    names = list(score_matrix.columns)
    pairs = []
    for a, b in combinations(names, 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        pairs.append({"Model A": a, "Model B": b, "rank_diff": round(diff, 4),
                      "significant": bool(diff > cd)})
    return {
        "avg_ranks": avg_ranks.sort_values().round(4).to_dict(),
        "critical_difference": round(float(cd), 4),
        "pairwise": pd.DataFrame(pairs).sort_values("rank_diff", ascending=False)
        .reset_index(drop=True),
    }
