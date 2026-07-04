"""Exploratory data analysis helpers: statistics, association measures, and
data-quality diagnostics used in the dataset audit (Step 1)."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as C


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric features including skew / kurtosis."""
    num = df[C.NUMERIC_FEATURES]
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["kurtosis"] = num.kurtosis()
    desc["variance"] = num.var()
    return desc.round(4)


def stats_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per feature for each class label."""
    g = df.groupby(C.TARGET)[C.NUMERIC_FEATURES].agg(["mean", "std", "min", "max"])
    return g.round(4)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[C.NUMERIC_FEATURES].corr().round(4)


def feature_variance(df: pd.DataFrame) -> pd.Series:
    return df[C.NUMERIC_FEATURES].var().round(4)


def mutual_information(df: pd.DataFrame) -> pd.Series:
    """Mutual information between each feature and the (encoded) target."""
    from .data import encode_target

    x = df[C.FEATURES].copy()
    # Encode categorical for MI.
    for col in C.CATEGORICAL_FEATURES:
        x[col] = x[col].astype("category").cat.codes
    y = encode_target(df[C.TARGET])
    discrete = [x.columns.get_loc(c) for c in C.CATEGORICAL_FEATURES]
    mi = mutual_info_classif(
        x, y, discrete_features=discrete, random_state=C.RANDOM_STATE
    )
    return pd.Series(mi, index=x.columns).sort_values(ascending=False).round(4)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical variables."""
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def cramers_v_with_target(df: pd.DataFrame, n_bins: int = 4) -> pd.Series:
    """Cramer's V of each feature vs the target.

    Numeric features are quantile-binned first, so the measure is comparable
    across categorical (Fruit) and numeric sensors.
    """
    out: Dict[str, float] = {}
    target = df[C.TARGET]
    for col in C.FEATURES:
        if col in C.CATEGORICAL_FEATURES:
            binned = df[col]
        else:
            binned = pd.qcut(df[col], q=n_bins, duplicates="drop")
        out[col] = cramers_v(binned, target)
    return pd.Series(out).sort_values(ascending=False).round(4)


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for numeric features (multicollinearity)."""
    x = df[C.NUMERIC_FEATURES].astype(float).copy()
    x = (x - x.mean()) / x.std(ddof=0)
    x = x.assign(_const=1.0)
    rows = []
    for i, col in enumerate(C.NUMERIC_FEATURES):
        rows.append({"feature": col, "VIF": variance_inflation_factor(x.values, i)})
    return pd.DataFrame(rows).round(4)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-based outlier counts per numeric feature."""
    rows = []
    for col in C.NUMERIC_FEATURES:
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        rows.append(
            {
                "feature": col,
                "lower_fence": round(lo, 3),
                "upper_fence": round(hi, 3),
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(100 * mask.mean(), 3),
            }
        )
    return pd.DataFrame(rows)


def class_imbalance(df: pd.DataFrame) -> Dict:
    counts = df[C.TARGET].value_counts()
    total = int(counts.sum())
    majority = counts.max()
    minority = counts.min()
    return {
        "counts": counts.to_dict(),
        "proportions": (counts / total).round(4).to_dict(),
        "imbalance_ratio": round(majority / minority, 4),
    }


def leakage_scan(df: pd.DataFrame, corr_threshold: float = 0.98) -> pd.DataFrame:
    """Heuristic leakage scan: features almost perfectly predictive on their own.

    For each numeric feature we find the single-threshold split that best
    separates the classes and report its accuracy; near-1.0 values flag
    potential leakage or a trivially separable variable.
    """
    from .data import encode_target

    y = encode_target(df[C.TARGET]).to_numpy()
    rows = []
    for col in C.NUMERIC_FEATURES:
        x = df[col].to_numpy()
        thresholds = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 99)))
        best_acc = 0.0
        best_thr = np.nan
        for thr in thresholds:
            for direction in (1, -1):
                pred = ((x > thr).astype(int) if direction == 1 else (x <= thr).astype(int))
                acc = max((pred == y).mean(), (1 - (pred == y).mean()))
                if acc > best_acc:
                    best_acc = acc
                    best_thr = thr
        rows.append(
            {
                "feature": col,
                "best_single_split_acc": round(best_acc, 4),
                "threshold": round(float(best_thr), 4),
            }
        )
    res = pd.DataFrame(rows).sort_values(
        "best_single_split_acc", ascending=False
    ).reset_index(drop=True)
    res["flag"] = np.where(
        res["best_single_split_acc"] >= corr_threshold, "possible-leakage", "ok"
    )
    return res


def duplicate_analysis(raw: pd.DataFrame) -> Dict:
    """Duplicate diagnostics on the raw (renamed) dataframe."""
    n_dup = int(raw.duplicated().sum())
    return {
        "n_rows": len(raw),
        "n_exact_duplicates": n_dup,
        "pct_duplicates": round(100 * n_dup / len(raw), 3),
        "n_unique": len(raw) - n_dup,
    }
