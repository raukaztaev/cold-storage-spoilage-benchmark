"""Model zoo for the spoilage-risk benchmark.

Fifteen classifiers spanning four families (linear, tree, boosting, other) are
defined with sensible, comparable defaults. Each classifier is wrapped in a
pipeline that shares the same leakage-free preprocessor, so every model sees an
identical feature space.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocessing import build_preprocessor

# Optional third-party boosters -------------------------------------------------
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


MODEL_FAMILIES = {
    "Logistic Regression": "Linear",
    "Ridge Classifier": "Linear",
    "SGD Classifier": "Linear",
    "Decision Tree": "Tree",
    "Random Forest": "Tree",
    "Extra Trees": "Tree",
    "AdaBoost": "Boosting",
    "Gradient Boosting": "Boosting",
    "XGBoost": "Boosting",
    "LightGBM": "Boosting",
    "CatBoost": "Boosting",
    "KNN": "Other",
    "GaussianNB": "Other",
    "SVM": "Other",
    "MLP": "Other",
}

# Models that require feature scaling for well-behaved optimisation / distances.
_NEEDS_SCALING = {
    "Logistic Regression",
    "Ridge Classifier",
    "SGD Classifier",
    "KNN",
    "SVM",
    "MLP",
    "GaussianNB",
}

# Models that expose calibrated probabilities via predict_proba out of the box.
_NO_PROBA = {"Ridge Classifier", "SGD Classifier"}


def get_estimators() -> Dict[str, object]:
    """Return a fresh dict of bare estimators (no preprocessing attached)."""
    rs = C.RANDOM_STATE
    est: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=rs),
        "Ridge Classifier": RidgeClassifier(random_state=rs),
        "SGD Classifier": SGDClassifier(
            loss="log_loss", max_iter=5000, random_state=rs
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=rs, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=rs, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "GaussianNB": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=rs),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=rs
        ),
    }

    if _HAS_XGB:
        est["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=rs,
            n_jobs=-1,
            tree_method="hist",
        )
    if _HAS_LGBM:
        est["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=rs,
            n_jobs=-1,
            verbose=-1,
        )
    if _HAS_CATBOOST:
        est["CatBoost"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            random_state=rs,
            verbose=False,
            allow_writing_files=False,
        )
    return est


def needs_scaling(name: str) -> bool:
    return name in _NEEDS_SCALING


def supports_proba(name: str) -> bool:
    return name not in _NO_PROBA


def build_pipeline(name: str, estimator, x: pd.DataFrame) -> Pipeline:
    """Wrap an estimator in a preprocessing pipeline sized to ``x``."""
    pre = build_preprocessor(x, scale=needs_scaling(name), ohe=True)
    return Pipeline([("preprocessor", pre), ("model", estimator)])


def get_model_zoo(x: pd.DataFrame) -> Dict[str, Pipeline]:
    """Return name -> fitted-ready pipeline for every model in the zoo."""
    return {
        name: build_pipeline(name, est, x)
        for name, est in get_estimators().items()
    }


def ordered_model_names() -> Tuple[str, ...]:
    return tuple(MODEL_FAMILIES.keys())
