"""Phase B driver: label-provenance analysis + extended robustness.

Reuses the already-trained pipelines in ``models/`` (no retraining) and the
deterministic split from ``src.data``. Regenerates the tables/figures consumed
by the revised article (reviewer comments 2 and 5).

Run from the project root:
    OMP_NUM_THREADS=1 python tools/run_phaseB.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import label_rule as LR  # noqa: E402
from src import robustness as R  # noqa: E402
from src import viz  # noqa: E402
from src.models import ordered_model_names  # noqa: E402
from src.utils import load_model, save_table  # noqa: E402

matplotlib.use("Agg")


def _sep(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def run_label_rule(df: pd.DataFrame) -> None:
    _sep("LABEL-PROVENANCE ANALYSIS (reviewer comment 2)")

    fidelity = LR.global_rule_fidelity(df)
    save_table(fidelity, "label_rule_fidelity")
    print("\nGlobal decision-tree fidelity to the label:")
    print(fidelity.to_string(index=False))

    per_fruit = LR.per_fruit_threshold_rule(df)
    save_table(per_fruit, "label_rule_thresholds")
    print("\nRecovered per-fruit single-threshold rule:")
    print(per_fruit.to_string(index=False))

    tree_fid = LR.per_fruit_tree_fidelity(df, max_depth=3)
    save_table(tree_fid, "label_rule_per_fruit_tree")
    print("\nPer-fruit depth-3 tree fidelity:")
    print(tree_fid.to_string(index=False))

    # Figure: per-fruit distribution of the discriminating feature by class.
    viz.setup_style()
    fruits = per_fruit["Fruit"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (_, row) in zip(axes.ravel(), per_fruit.iterrows()):
        fruit = row["Fruit"]
        feat = row["Split feature"]
        g = df[df["Fruit"] == fruit]
        y = (g[C.TARGET].str.strip().str.lower() == "bad")
        thr = float(row["Rule (Bad if)"].split()[2])
        ax.hist(g.loc[~y, feat], bins=30, alpha=0.6, label="Good", color="#4C72B0")
        ax.hist(g.loc[y, feat], bins=30, alpha=0.6, label="Bad", color="#C44E52")
        ax.axvline(thr, color="k", ls="--", lw=1.4, label=f"threshold={thr:g}")
        unit = C.FEATURE_UNITS.get(feat, "")
        ax.set_title(f"{fruit} - split on {feat} (acc {row['Reproduction accuracy']:.3f})")
        ax.set_xlabel(f"{feat} ({unit})" if unit else feat)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Recovered per-fruit labelling rule", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "label_rule_per_fruit.png")
    plt.close(fig)
    print(f"\nSaved figure -> {C.rel(C.FIGURES_DIR / 'label_rule_per_fruit.png')}")


def run_robustness(fitted: dict, x_test, y_test, names: list) -> None:
    _sep("EXTENDED ROBUSTNESS (reviewer comment 5)")

    long = R.robustness_extended(fitted, x_test, y_test, names, n_repeats=5)
    save_table(long, "robustness_extended_long")

    summary = R.robustness_summary_extended(long)
    save_table(summary, "robustness_extended_summary")
    print("\nPer-model robustness summary (sorted by mean F1 drop):")
    print(summary.to_string(index=False))

    curve = R.noise_level_curve(fitted, x_test, y_test, names, n_repeats=5)
    save_table(curve, "robustness_noise_curve")

    # Figure 1: F1 and missed-risk (FNR) vs noise level for representative models.
    viz.setup_style()
    highlight = [m for m in ["XGBoost", "CatBoost", "Extra Trees",
                             "Random Forest", "Logistic Regression"] if m in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in highlight:
        sub = curve[curve["Model"] == m].sort_values("Noise level")
        axes[0].plot(sub["Noise level"], sub["F1"], marker="o", label=m)
        axes[1].plot(sub["Noise level"], sub["FNR"], marker="o", label=m)
    axes[0].set_title("F1 vs noise level")
    axes[0].set_xlabel("Noise level (x base sigma)")
    axes[0].set_ylabel("F1")
    axes[1].set_title("Missed-spoilage rate (FNR) vs noise level")
    axes[1].set_xlabel("Noise level (x base sigma)")
    axes[1].set_ylabel("FNR")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "robustness_noise_curve.png")
    plt.close(fig)
    print(f"Saved figure -> {C.rel(C.FIGURES_DIR / 'robustness_noise_curve.png')}")

    # Figure 2: heatmap of F1 across every scenario x model.
    pivot = long.pivot(index="Model", columns="Scenario", values="F1")
    order = summary.set_index("Model").index.tolist()
    pivot = pivot.loc[[m for m in order if m in pivot.index]]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.2, vmax=1.0,
                cbar_kws={"label": "F1"}, ax=ax)
    ax.set_title("Robustness: F1 across perturbation scenarios", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "robustness_extended_heatmap.png")
    plt.close(fig)
    print(f"Saved figure -> {C.rel(C.FIGURES_DIR / 'robustness_extended_heatmap.png')}")


def main() -> None:
    df = D.load_clean()
    x_train, x_test, y_train, y_test = D.get_splits(df)
    print(f"Clean rows: {len(df)} | train: {len(x_train)} | test: {len(x_test)} "
          f"| Bad frac (test): {y_test.mean():.3f}")

    names = list(ordered_model_names())
    fitted = {}
    for name in names:
        try:
            fitted[name] = load_model(name)
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] could not load {name}: {exc}")
    names = [n for n in names if n in fitted]
    print(f"Loaded {len(names)} trained models.")

    run_label_rule(df)
    run_robustness(fitted, x_test, y_test, names)
    _sep("PHASE B COMPLETE")


if __name__ == "__main__":
    main()
