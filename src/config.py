"""Central configuration: paths, constants, and reproducibility settings.

All paths are resolved relative to the repository root so that every notebook
and script is runnable from any working directory.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# src/ -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATASET = RAW_DIR / "Dataset.csv"
CLEAN_DATASET = PROCESSED_DIR / "clean_dataset.csv"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------
TARGET = "Class"
# Positive class = "Bad" (spoilage risk) -- the event we want to detect.
POSITIVE_LABEL = "Bad"
NEGATIVE_LABEL = "Good"

RAW_COLUMN_RENAME = {
    "Humid (%)": "Humidity",
    "Light (Fux)": "Light",
    "CO2 (pmm)": "CO2",
}

NUMERIC_FEATURES = ["Temp", "Humidity", "Light", "CO2"]
CATEGORICAL_FEATURES = ["Fruit"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable feature units for plots / tables.
FEATURE_UNITS = {
    "Temp": "\u00b0C",
    "Humidity": "%",
    "Light": "Lux",
    "CO2": "ppm",
}


def rel(path: Path) -> str:
    """Return a path relative to the project root for compact logging."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
