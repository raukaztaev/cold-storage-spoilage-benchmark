"""Publication-quality figure generation.

Every function saves a PNG (300 dpi) into ``figures/`` and returns the path so
notebooks can display and reference the artefacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve

from . import config as C

matplotlib.use("Agg")

PALETTE = "colorblind"


def setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )


def _save(fig, name: str) -> Path:
    path = C.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# EDA figures
# ---------------------------------------------------------------------------
def plot_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = [C.NEGATIVE_LABEL, C.POSITIVE_LABEL]
    sns.countplot(data=df, x=C.TARGET, order=order, ax=axes[0], hue=C.TARGET, legend=False)
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    fruit_order = df["Fruit"].value_counts().index
    sns.countplot(data=df, x="Fruit", order=fruit_order, ax=axes[1], hue="Fruit", legend=False)
    axes[1].set_title("Fruit Distribution")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save(fig, "dataset_distribution.png")


def plot_pairplot(df: pd.DataFrame) -> Path:
    g = sns.pairplot(
        df[C.NUMERIC_FEATURES + [C.TARGET]],
        hue=C.TARGET,
        hue_order=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL],
        corner=True,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 12},
    )
    g.figure.suptitle("Pairwise Feature Relationships by Class", y=1.02)
    path = C.FIGURES_DIR / "pairplot.png"
    g.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    return path


def plot_violin_box(df: pd.DataFrame) -> List[Path]:
    paths = []
    for kind, fname in (("violin", "violin_plots.png"), ("box", "box_plots.png")):
        fig, axes = plt.subplots(1, len(C.NUMERIC_FEATURES), figsize=(16, 4))
        for ax, col in zip(axes, C.NUMERIC_FEATURES):
            if kind == "violin":
                sns.violinplot(
                    data=df, x=C.TARGET, y=col, order=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL],
                    ax=ax, hue=C.TARGET, legend=False, cut=0,
                )
            else:
                sns.boxplot(
                    data=df, x=C.TARGET, y=col, order=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL],
                    ax=ax, hue=C.TARGET, legend=False,
                )
            ax.set_title(f"{col} ({C.FEATURE_UNITS.get(col, '')})")
            ax.set_xlabel("")
        fig.suptitle(f"{kind.capitalize()} plots of sensor features by class", y=1.03)
        fig.tight_layout()
        paths.append(_save(fig, fname))
    return paths


def plot_kde(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(C.NUMERIC_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, C.NUMERIC_FEATURES):
        sns.kdeplot(
            data=df, x=col, hue=C.TARGET, hue_order=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL],
            fill=True, common_norm=False, alpha=0.4, ax=ax,
        )
        ax.set_title(f"{col} ({C.FEATURE_UNITS.get(col, '')})")
    fig.suptitle("Kernel density estimates by class", y=1.03)
    fig.tight_layout()
    return _save(fig, "kde_plots.png")


def plot_correlation(corr: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                vmin=-1, vmax=1, square=True)
    ax.set_title("Feature Correlation Matrix (Pearson)")
    fig.tight_layout()
    return _save(fig, "correlation_matrix.png")


def plot_bar_series(s: pd.Series, title: str, xlabel: str, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4))
    s_sorted = s.sort_values()
    sns.barplot(x=s_sorted.values, y=s_sorted.index, ax=ax, hue=s_sorted.index, legend=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return _save(fig, fname)


# ---------------------------------------------------------------------------
# Model evaluation figures
# ---------------------------------------------------------------------------
def plot_roc_curves(y_true, predictions: Dict[str, Dict], names: List[str], fname="roc_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in names:
        score = predictions[name].get("y_score")
        if score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, fname)


def plot_pr_curves(y_true, predictions: Dict[str, Dict], names: List[str], fname="pr_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in names:
        score = predictions[name].get("y_score")
        if score is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, score)
        ax.plot(rec, prec, lw=1.8, label=f"{name} (AP={auc(rec, prec):.3f})")
    baseline = np.mean(y_true)
    ax.axhline(baseline, ls="--", color="k", lw=1, alpha=0.6, label=f"Baseline={baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save(fig, fname)


def plot_confusion_matrix(y_true, y_pred, name: str, fname: Optional[str] = None) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fname = fname or f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_confusion_grid(y_true, predictions: Dict[str, Dict], names: List[str], fname="confusion_grid.png") -> Path:
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_true, predictions[name]["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=10)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices", y=1.01)
    fig.tight_layout()
    return _save(fig, fname)


def plot_learning_curve(estimator, x, y, name: str, fname: Optional[str] = None) -> Path:
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, x, y, cv=C.CV_FOLDS, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=C.RANDOM_STATE, shuffle=True,
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    tm, ts = train_scores.mean(1), train_scores.std(1)
    vm, vs = val_scores.mean(1), val_scores.std(1)
    ax.plot(train_sizes, tm, "o-", label="Training F1")
    ax.fill_between(train_sizes, tm - ts, tm + ts, alpha=0.15)
    ax.plot(train_sizes, vm, "s-", label="CV F1")
    ax.fill_between(train_sizes, vm - vs, vm + vs, alpha=0.15)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Learning Curve - {name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fname = fname or f"learning_curve_{name.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_calibration(y_true, predictions: Dict[str, Dict], names: List[str], fname="calibration_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    for name in names:
        proba = predictions[name].get("y_proba")
        if proba is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", lw=1.6, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save(fig, fname)


def plot_feature_importance(importances: pd.Series, name: str, fname: Optional[str] = None) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4))
    s = importances.sort_values()
    sns.barplot(x=s.values, y=s.index, ax=ax, hue=s.index, legend=False)
    ax.set_title(f"Feature Importance - {name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fname = fname or f"feature_importance_{name.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_permutation_importance(imp_df: pd.DataFrame, name: str, fname="permutation_importance.png") -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4))
    imp_df = imp_df.sort_values("importance_mean")
    ax.barh(imp_df["feature"], imp_df["importance_mean"], xerr=imp_df["importance_std"])
    ax.set_title(f"Permutation Importance - {name}")
    ax.set_xlabel("Mean F1 drop")
    fig.tight_layout()
    return _save(fig, fname)


def plot_metric_bar(results: pd.DataFrame, metric: str, fname: Optional[str] = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    d = results.sort_values(metric, ascending=True)
    sns.barplot(x=d[metric], y=d["Model"], ax=ax, hue=d["Model"], legend=False)
    ax.set_title(f"Model Comparison by {metric}")
    ax.set_xlabel(metric)
    fig.tight_layout()
    fname = fname or f"comparison_{metric.lower().replace(' ', '_')}.png"
    return _save(fig, fname)
