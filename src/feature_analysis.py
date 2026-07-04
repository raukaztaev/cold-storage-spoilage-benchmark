"""Feature analysis (Step 11): SHAP, permutation importance, RFE, MI, and the
effect of feature selection on performance."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C
from .preprocessing import build_preprocessor


def permutation_importance_df(estimator, x_test, y_test, scoring="f1",
                              n_repeats: int = 20) -> pd.DataFrame:
    """Permutation importance on the *raw* feature columns via the pipeline."""
    result = permutation_importance(
        estimator, x_test, y_test, scoring=scoring, n_repeats=n_repeats,
        random_state=C.RANDOM_STATE, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": x_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def native_importance(pipeline, feature_names: Optional[List[str]] = None) -> Optional[pd.Series]:
    """Extract native feature_importances_ aggregated to original features.

    One-hot columns of the same source feature are summed so importances are
    reported per original sensor / Fruit variable.
    """
    model = pipeline.named_steps.get("model")
    pre = pipeline.named_steps.get("preprocessor")
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    try:
        out_names = list(pre.get_feature_names_out())
    except Exception:
        return None
    importances = model.feature_importances_
    agg: Dict[str, float] = {}
    for name, imp in zip(out_names, importances):
        # names look like 'num__Temp' or 'cat__Fruit_Orange'
        base = name.split("__", 1)[-1]
        if base.startswith("Fruit_") or base == "Fruit":
            key = "Fruit"
        else:
            key = base
        agg[key] = agg.get(key, 0.0) + float(imp)
    return pd.Series(agg).sort_values(ascending=False)


def rfe_selection(estimator_factory, x, y) -> Dict:
    """Recursive Feature Elimination with CV on the raw feature space.

    Uses OHE preprocessing then RFECV on the transformed matrix. Returns the
    optimal number of features and their names.
    """
    pre = build_preprocessor(x, scale=False, ohe=True)
    x_t = pre.fit_transform(x)
    names = list(pre.get_feature_names_out())
    est = estimator_factory()
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rfecv = RFECV(est, step=1, cv=skf, scoring="f1", n_jobs=-1, min_features_to_select=1)
    rfecv.fit(x_t, y)
    selected = [n for n, keep in zip(names, rfecv.support_) if keep]
    return {
        "n_features_in": len(names),
        "optimal_n_features": int(rfecv.n_features_),
        "selected_features": selected,
        "ranking": dict(zip(names, rfecv.ranking_.tolist())),
    }


def feature_selection_impact(x, y, base_estimator_factory,
                             feature_subsets: Dict[str, List[str]]) -> pd.DataFrame:
    """Compare CV F1 for different feature subsets (all original columns)."""
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    for label, cols in feature_subsets.items():
        xs = x[cols]
        pre = build_preprocessor(xs, scale=False, ohe=True)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("preprocessor", pre), ("model", base_estimator_factory())])
        scores = cross_val_score(pipe, xs, y, cv=skf, scoring="f1", n_jobs=-1)
        rows.append({"subset": label, "n_features": len(cols),
                     "cv_f1_mean": round(float(scores.mean()), 4),
                     "cv_f1_std": round(float(scores.std()), 4)})
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


def compute_shap(pipeline, x_train, x_test, max_background: int = 200,
                 max_explain: int = 500):
    """Compute SHAP values for a tree-based pipeline on the transformed space.

    Returns (shap_values_for_positive_class, transformed_test_df, feature_names).
    """
    import shap

    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    x_test_t = pre.transform(x_test)
    names = list(pre.get_feature_names_out())
    x_test_df = pd.DataFrame(np.asarray(x_test_t), columns=names)
    if len(x_test_df) > max_explain:
        x_test_df = x_test_df.sample(max_explain, random_state=C.RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test_df)
    # For binary classifiers shap may return a list [class0, class1] or a 3D array.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    return sv, x_test_df, names
