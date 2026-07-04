"""Reproducible spoilage-risk benchmark package.

Modules:
    config            paths, seeds, schema
    data              loading, cleaning, splitting
    preprocessing     leakage-free sklearn pipelines
    models            15-model zoo across four families
    metrics           unified metric suite
    benchmark         cross-validation and held-out evaluation
    hpo               Optuna hyperparameter optimisation
    stats_tests       McNemar / Friedman / Nemenyi
    robustness        sensor-perturbation stress tests
    error_analysis    misclassification diagnostics
    feature_analysis  SHAP / permutation / RFE / MI
    computational     latency, memory, model size
    viz               publication-quality figures
    utils             table / model persistence
"""
__version__ = "1.0.0"
