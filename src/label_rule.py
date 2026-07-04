"""Label-provenance analysis (reviewer comment 2).

The Mendeley source does not document how the ``Good``/``Bad`` label was
generated. We therefore probe, empirically, whether the label is (approximately)
a deterministic function of the recorded fruit type and the four environmental
sensors. If a compact per-fruit threshold rule reproduces the label almost
perfectly, that (a) recovers the most likely labelling protocol and (b) explains
why every capable classifier reaches near-perfect accuracy: the models are
re-discovering a rule, not learning a hard biological signal.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def _binary_target(df: pd.DataFrame) -> pd.Series:
    return (df[C.TARGET].astype(str).str.strip().str.lower()
            == C.POSITIVE_LABEL.lower()).astype(int)


def global_rule_fidelity(df: pd.DataFrame,
                         depths: List[int] | None = None) -> pd.DataFrame:
    """How well a single decision tree on (sensors) reproduces the label.

    Reports both the resubstitution accuracy (rule-recovery fidelity) and a
    stratified 5-fold CV accuracy (does the recovered rule generalise), for a
    range of tree depths.
    """
    if depths is None:
        depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 0]  # 0 -> unlimited
    y = _binary_target(df)
    x = df[C.NUMERIC_FEATURES].copy()
    rows: List[Dict] = []
    for d in depths:
        max_depth = None if d == 0 else d
        tree = DecisionTreeClassifier(max_depth=max_depth,
                                      random_state=C.RANDOM_STATE)
        tree.fit(x, y)
        resub = tree.score(x, y)
        cv = cross_val_score(
            DecisionTreeClassifier(max_depth=max_depth,
                                   random_state=C.RANDOM_STATE),
            x, y, cv=C.CV_FOLDS, scoring="accuracy",
        ).mean()
        rows.append({
            "Max depth": "unlimited" if max_depth is None else max_depth,
            "Leaves": int(tree.get_n_leaves()),
            "Resubstitution accuracy": round(float(resub), 4),
            "5-fold CV accuracy": round(float(cv), 4),
        })
    return pd.DataFrame(rows)


def per_fruit_threshold_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Best single-feature threshold per fruit that reproduces the label.

    For each fruit we fit a depth-1 tree on every numeric sensor and keep the
    most accurate one, recording the split feature, threshold, the direction of
    the ``Bad`` decision, and the reproduction accuracy.
    """
    rows: List[Dict] = []
    for fruit, g in df.groupby("Fruit"):
        yy = _binary_target(g)
        best = None
        for feat in C.NUMERIC_FEATURES:
            stump = DecisionTreeClassifier(
                max_depth=1, random_state=C.RANDOM_STATE
            ).fit(g[[feat]], yy)
            acc = stump.score(g[[feat]], yy)
            if best is None or acc > best["acc"]:
                thr = float(stump.tree_.threshold[0])
                bad_rate_low = float(yy[g[feat] <= thr].mean()) if (g[feat] <= thr).any() else 0.0
                bad_rate_high = float(yy[g[feat] > thr].mean()) if (g[feat] > thr).any() else 0.0
                direction = ">" if bad_rate_high >= bad_rate_low else "<="
                best = {"feat": feat, "thr": thr, "acc": float(acc),
                        "direction": direction}
        unit = C.FEATURE_UNITS.get(best["feat"], "")
        rows.append({
            "Fruit": fruit,
            "n": int(len(g)),
            "Bad rate": round(float(yy.mean()), 3),
            "Split feature": best["feat"],
            "Rule (Bad if)": f"{best['feat']} {best['direction']} {best['thr']:.1f} {unit}".strip(),
            "Reproduction accuracy": round(best["acc"], 3),
        })
    return pd.DataFrame(rows).sort_values("Fruit").reset_index(drop=True)


def per_fruit_tree_fidelity(df: pd.DataFrame, max_depth: int = 3) -> pd.DataFrame:
    """Fidelity of a small per-fruit tree (all four sensors, depth-limited)."""
    rows: List[Dict] = []
    for fruit, g in df.groupby("Fruit"):
        yy = _binary_target(g)
        x = g[C.NUMERIC_FEATURES]
        tree = DecisionTreeClassifier(
            max_depth=max_depth, random_state=C.RANDOM_STATE
        ).fit(x, yy)
        rows.append({
            "Fruit": fruit,
            "n": int(len(g)),
            f"Depth-{max_depth} tree accuracy": round(float(tree.score(x, yy)), 4),
            "Leaves": int(tree.get_n_leaves()),
        })
    return pd.DataFrame(rows).sort_values("Fruit").reset_index(drop=True)
