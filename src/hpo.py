"""Hyperparameter optimisation with Optuna for the four ensemble models.

Each objective performs stratified k-fold cross-validation on the *training*
set only and maximises mean F1, so no test information leaks into tuning.
"""
from __future__ import annotations

import json
from typing import Callable, Dict

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config as C
from .preprocessing import build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False
try:
    from catboost import CatBoostClassifier

    _HAS_CATBOOST = True
except Exception:  # pragma: no cover
    _HAS_CATBOOST = False


def _cv_score(estimator, x, y, scale: bool) -> float:
    pre = build_preprocessor(x, scale=scale, ohe=True)
    pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
    skf = StratifiedKFold(
        n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE
    )
    scores = cross_val_score(pipe, x, y, cv=skf, scoring="f1", n_jobs=-1)
    return float(np.mean(scores))


def _rf_objective(x, y) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }
        est = RandomForestClassifier(
            random_state=C.RANDOM_STATE, n_jobs=-1, **params
        )
        return _cv_score(est, x, y, scale=False)

    return objective


def _xgb_objective(x, y) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        est = XGBClassifier(
            eval_metric="logloss",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        return _cv_score(est, x, y, scale=False)

    return objective


def _lgbm_objective(x, y) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        est = LGBMClassifier(
            random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1, **params
        )
        return _cv_score(est, x, y, scale=False)

    return objective


def _catboost_objective(x, y) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 600, step=50),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        est = CatBoostClassifier(
            random_state=C.RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        return _cv_score(est, x, y, scale=False)

    return objective


_OBJECTIVES = {}
_OBJECTIVES["Random Forest"] = _rf_objective
if _HAS_XGB:
    _OBJECTIVES["XGBoost"] = _xgb_objective
if _HAS_LGBM:
    _OBJECTIVES["LightGBM"] = _lgbm_objective
if _HAS_CATBOOST:
    _OBJECTIVES["CatBoost"] = _catboost_objective


def optimise(x, y, n_trials: int = 40) -> Dict[str, Dict]:
    """Run Optuna for each ensemble model. Returns name -> {params, value}."""
    results: Dict[str, Dict] = {}
    for name, obj_factory in _OBJECTIVES.items():
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=C.RANDOM_STATE),
        )
        study.optimize(obj_factory(x, y), n_trials=n_trials, show_progress_bar=False)
        results[name] = {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": n_trials,
        }
    return results


def build_tuned_estimators(best: Dict[str, Dict]) -> Dict[str, object]:
    """Instantiate estimators from tuned parameters."""
    est: Dict[str, object] = {}
    if "Random Forest" in best:
        est["Random Forest"] = RandomForestClassifier(
            random_state=C.RANDOM_STATE, n_jobs=-1, **best["Random Forest"]["best_params"]
        )
    if _HAS_XGB and "XGBoost" in best:
        est["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **best["XGBoost"]["best_params"],
        )
    if _HAS_LGBM and "LightGBM" in best:
        est["LightGBM"] = LGBMClassifier(
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            **best["LightGBM"]["best_params"],
        )
    if _HAS_CATBOOST and "CatBoost" in best:
        est["CatBoost"] = CatBoostClassifier(
            random_state=C.RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            **best["CatBoost"]["best_params"],
        )
    return est


def save_best_params(best: Dict[str, Dict], path=None):
    path = path or (C.MODELS_DIR / "best_params.json")
    with open(path, "w") as f:
        json.dump(best, f, indent=2)


def load_best_params(path=None) -> Dict[str, Dict]:
    path = path or (C.MODELS_DIR / "best_params.json")
    with open(path) as f:
        return json.load(f)
