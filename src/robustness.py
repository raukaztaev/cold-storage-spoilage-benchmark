"""Sensor-perturbation robustness analysis (Step 8).

We inject controlled, physically motivated perturbations into the test-set
sensor readings (calibration drift / noise) and measure how much each already
trained model degrades. This emulates the imperfect sensors of a real IoT
deployment without retraining.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

from . import config as C

# Perturbation scenarios: (label, {feature: shift/std}). Deterministic shifts
# use a fixed delta; noise scenarios add zero-mean Gaussian noise with the given
# std (magnitudes from the task brief).
PERTURBATIONS = [
    {"label": "Temp +1C", "type": "shift", "changes": {"Temp": 1.0}},
    {"label": "Temp +2C", "type": "shift", "changes": {"Temp": 2.0}},
    {"label": "Humidity +5%", "type": "shift", "changes": {"Humidity": 5.0}},
    {"label": "Humidity -5%", "type": "shift", "changes": {"Humidity": -5.0}},
    {"label": "CO2 +20ppm", "type": "shift", "changes": {"CO2": 20.0}},
    {"label": "CO2 -20ppm", "type": "shift", "changes": {"CO2": -20.0}},
    {"label": "Light +20Lux", "type": "shift", "changes": {"Light": 20.0}},
    {"label": "Light -20Lux", "type": "shift", "changes": {"Light": -20.0}},
    {"label": "Gaussian noise (all)", "type": "noise",
     "changes": {"Temp": 1.0, "Humidity": 3.0, "Light": 10.0, "CO2": 20.0}},
]


def _apply(x: pd.DataFrame, spec: Dict, rng: np.random.Generator) -> pd.DataFrame:
    xp = x.copy()
    for feat, val in spec["changes"].items():
        if spec["type"] == "shift":
            xp[feat] = xp[feat] + val
        else:  # noise
            xp[feat] = xp[feat] + rng.normal(0.0, val, size=len(xp))
    # Keep values physically plausible (humidity <= 100, non-negative light).
    if "Humidity" in xp:
        xp["Humidity"] = xp["Humidity"].clip(upper=100)
    if "Light" in xp:
        xp["Light"] = xp["Light"].clip(lower=0)
    return xp


def robustness_analysis(fitted: Dict, x_test: pd.DataFrame, y_test: pd.Series,
                        model_names: List[str]) -> pd.DataFrame:
    """Return a long-format dataframe: model x scenario -> F1 / MCC / drop."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows: List[Dict] = []

    baseline: Dict[str, float] = {}
    for name in model_names:
        est = fitted[name]
        base_f1 = f1_score(y_test, est.predict(x_test), zero_division=0)
        baseline[name] = base_f1
        rows.append({"Model": name, "Scenario": "Clean", "F1": base_f1,
                     "MCC": matthews_corrcoef(y_test, est.predict(x_test)),
                     "F1_drop": 0.0})

    for spec in PERTURBATIONS:
        xp = _apply(x_test, spec, rng)
        for name in model_names:
            est = fitted[name]
            pred = est.predict(xp)
            f1 = f1_score(y_test, pred, zero_division=0)
            rows.append({
                "Model": name,
                "Scenario": spec["label"],
                "F1": f1,
                "MCC": matthews_corrcoef(y_test, pred),
                "F1_drop": baseline[name] - f1,
            })
    return pd.DataFrame(rows)


def robustness_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean F1 drop across all perturbed scenarios per model."""
    perturbed = long_df[long_df["Scenario"] != "Clean"]
    summary = (
        perturbed.groupby("Model")
        .agg(mean_F1_under_perturbation=("F1", "mean"),
             worst_F1=("F1", "min"),
             mean_F1_drop=("F1_drop", "mean"),
             max_F1_drop=("F1_drop", "max"))
        .round(4)
        .sort_values("mean_F1_drop")
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Extended robustness (reviewer comment 5): multiple noise levels, combined
# miscalibration, temporal drift, and false-alarm / missed-risk accounting.
# ---------------------------------------------------------------------------
# Base per-sensor 1-sigma noise magnitudes (physically motivated). Noise levels
# scale these; e.g. level 2.0 doubles every sigma.
BASE_SIGMA = {"Temp": 1.0, "Humidity": 3.0, "Light": 10.0, "CO2": 20.0}
NOISE_LEVELS = [0.5, 1.0, 2.0, 3.0]

# Simultaneous multi-sensor miscalibration (correlated bias across sensors).
COMBINED_SHIFTS = [
    {"label": "Temp +2C & Humidity -5%", "changes": {"Temp": 2.0, "Humidity": -5.0}},
    {"label": "Temp +1C & CO2 +20ppm", "changes": {"Temp": 1.0, "CO2": 20.0}},
    {"label": "Light -20Lux & Humidity +5%", "changes": {"Light": -20.0, "Humidity": 5.0}},
    {"label": "All sensors biased", "changes": {"Temp": 2.0, "Humidity": -5.0,
                                                "Light": -20.0, "CO2": 20.0}},
]

# Monotonic drift ramps applied along the (pseudo-)stream: the offset grows
# linearly from 0 to the stated maximum across the ordered test rows. This
# emulates slow calibration drift over time; it is synthetic because the source
# dataset carries no timestamps.
DRIFT_SCENARIOS = [
    {"label": "Drift Light -> -40Lux", "changes": {"Light": -40.0}},
    {"label": "Drift Temp -> +3C", "changes": {"Temp": 3.0}},
    {"label": "Drift all sensors", "changes": {"Temp": 3.0, "Humidity": -10.0,
                                               "Light": -40.0, "CO2": 40.0}},
]


def _clip_physical(xp: pd.DataFrame) -> pd.DataFrame:
    if "Humidity" in xp:
        xp["Humidity"] = xp["Humidity"].clip(upper=100)
    if "Light" in xp:
        xp["Light"] = xp["Light"].clip(lower=0)
    return xp


def _apply_shift(x: pd.DataFrame, changes: Dict[str, float]) -> pd.DataFrame:
    xp = x.copy()
    for feat, val in changes.items():
        xp[feat] = xp[feat] + val
    return _clip_physical(xp)


def _apply_noise(x: pd.DataFrame, sigmas: Dict[str, float],
                 rng: np.random.Generator) -> pd.DataFrame:
    xp = x.copy()
    for feat, sd in sigmas.items():
        if sd > 0:
            xp[feat] = xp[feat] + rng.normal(0.0, sd, size=len(xp))
    return _clip_physical(xp)


def _apply_drift(x: pd.DataFrame, changes: Dict[str, float]) -> pd.DataFrame:
    xp = x.copy()
    n = len(xp)
    ramp = np.linspace(0.0, 1.0, n)
    for feat, max_val in changes.items():
        xp[feat] = xp[feat].to_numpy() + max_val * ramp
    return _clip_physical(xp)


def _rates(y_true, y_pred) -> Dict[str, float]:
    """Missed-risk (FNR, positive=spoilage) and false-alarm (FPR) rates."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == 1
    neg = y_true == 0
    fn = int(np.sum(pos & (y_pred == 0)))
    fp = int(np.sum(neg & (y_pred == 1)))
    fnr = fn / int(pos.sum()) if pos.sum() else float("nan")
    fpr = fp / int(neg.sum()) if neg.sum() else float("nan")
    return {"FNR": fnr, "FPR": fpr}


def _score(est, x, y_true) -> Dict[str, float]:
    pred = est.predict(x)
    out = {"F1": f1_score(y_true, pred, zero_division=0),
           "MCC": matthews_corrcoef(y_true, pred)}
    out.update(_rates(y_true, pred))
    return out


def robustness_extended(fitted: Dict, x_test: pd.DataFrame, y_test: pd.Series,
                        model_names: List[str], n_repeats: int = 5) -> pd.DataFrame:
    """Comprehensive per-scenario robustness with F1/MCC/FNR/FPR.

    Scenarios: clean baseline, single-sensor shifts, multi-level Gaussian noise
    (averaged over ``n_repeats`` seeds), combined multi-sensor bias, and
    monotonic temporal drift. FNR = missed spoilage rate; FPR = false-alarm rate.
    """
    rows: List[Dict] = []
    baseline: Dict[str, float] = {}

    for name in model_names:
        s = _score(fitted[name], x_test, y_test)
        baseline[name] = s["F1"]
        rows.append({"Model": name, "Scenario": "Clean", "Category": "Baseline",
                     "F1_drop": 0.0, **s})

    # Single-sensor deterministic shifts (reuse original grid).
    for spec in PERTURBATIONS:
        if spec["type"] != "shift":
            continue
        xp = _apply_shift(x_test, spec["changes"])
        for name in model_names:
            s = _score(fitted[name], xp, y_test)
            rows.append({"Model": name, "Scenario": spec["label"],
                         "Category": "Single shift",
                         "F1_drop": baseline[name] - s["F1"], **s})

    # Multi-level Gaussian noise, averaged across seeds.
    for level in NOISE_LEVELS:
        sigmas = {k: v * level for k, v in BASE_SIGMA.items()}
        for name in model_names:
            acc = {"F1": [], "MCC": [], "FNR": [], "FPR": []}
            for r in range(n_repeats):
                rng = np.random.default_rng(C.RANDOM_STATE + r)
                xp = _apply_noise(x_test, sigmas, rng)
                s = _score(fitted[name], xp, y_test)
                for k in acc:
                    acc[k].append(s[k])
            mean = {k: float(np.mean(v)) for k, v in acc.items()}
            rows.append({"Model": name,
                         "Scenario": f"Gaussian noise x{level:g}",
                         "Category": "Noise level",
                         "F1_drop": baseline[name] - mean["F1"], **mean})

    # Combined multi-sensor miscalibration.
    for spec in COMBINED_SHIFTS:
        xp = _apply_shift(x_test, spec["changes"])
        for name in model_names:
            s = _score(fitted[name], xp, y_test)
            rows.append({"Model": name, "Scenario": spec["label"],
                         "Category": "Combined shift",
                         "F1_drop": baseline[name] - s["F1"], **s})

    # Monotonic temporal drift.
    for spec in DRIFT_SCENARIOS:
        xp = _apply_drift(x_test, spec["changes"])
        for name in model_names:
            s = _score(fitted[name], xp, y_test)
            rows.append({"Model": name, "Scenario": spec["label"],
                         "Category": "Temporal drift",
                         "F1_drop": baseline[name] - s["F1"], **s})

    cols = ["Model", "Scenario", "Category", "F1", "MCC", "FNR", "FPR", "F1_drop"]
    return pd.DataFrame(rows)[cols]


def noise_level_curve(fitted: Dict, x_test: pd.DataFrame, y_test: pd.Series,
                      model_names: List[str], levels: List[float] | None = None,
                      n_repeats: int = 5) -> pd.DataFrame:
    """Model x noise-level table of mean F1, FNR (missed risk), FPR (false alarm)."""
    if levels is None:
        levels = [0.0] + NOISE_LEVELS
    rows: List[Dict] = []
    for level in levels:
        sigmas = {k: v * level for k, v in BASE_SIGMA.items()}
        for name in model_names:
            acc = {"F1": [], "FNR": [], "FPR": []}
            reps = 1 if level == 0 else n_repeats
            for r in range(reps):
                rng = np.random.default_rng(C.RANDOM_STATE + r)
                xp = x_test if level == 0 else _apply_noise(x_test, sigmas, rng)
                s = _score(fitted[name], xp, y_test)
                for k in acc:
                    acc[k].append(s[k])
            rows.append({"Model": name, "Noise level": level,
                         "F1": float(np.mean(acc["F1"])),
                         "FNR": float(np.mean(acc["FNR"])),
                         "FPR": float(np.mean(acc["FPR"]))})
    return pd.DataFrame(rows)


def robustness_summary_extended(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate over all perturbed scenarios (F1, drop, and the
    worst-case missed-risk / false-alarm rates a deployment would face)."""
    pert = long_df[long_df["Scenario"] != "Clean"]
    summary = (
        pert.groupby("Model")
        .agg(mean_F1=("F1", "mean"),
             worst_F1=("F1", "min"),
             mean_F1_drop=("F1_drop", "mean"),
             worst_FNR=("FNR", "max"),
             worst_FPR=("FPR", "max"))
        .round(4)
        .sort_values("mean_F1_drop")
        .reset_index()
    )
    return summary
