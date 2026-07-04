"""I/O helpers: persisting tables (csv + LaTeX), models, and JSON summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from . import config as C


def save_table(df: pd.DataFrame, name: str, index: bool = False,
               float_format: str = "%.4f", caption: str | None = None,
               label: str | None = None) -> Dict[str, Path]:
    """Save a dataframe as CSV and LaTeX (booktabs) into ``tables/``."""
    csv_path = C.TABLES_DIR / f"{name}.csv"
    tex_path = C.TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=index)
    try:
        df.to_latex(tex_path, index=index, float_format=float_format,
                    caption=caption, label=label, escape=True, longtable=False)
    except Exception:
        with open(tex_path, "w") as f:
            f.write(df.to_latex(index=index, float_format=float_format))
    return {"csv": csv_path, "tex": tex_path}


def save_json(obj, name: str, subdir: Path | None = None) -> Path:
    target = (subdir or C.TABLES_DIR) / f"{name}.json"
    with open(target, "w") as f:
        json.dump(_to_serialisable(obj), f, indent=2)
    return target


def _to_serialisable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def save_model(model, name: str) -> Path:
    path = C.MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = C.MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
    return joblib.load(path)


def save_benchmark(df: pd.DataFrame, stem: str = "benchmark") -> Dict[str, Path]:
    """Save the main benchmark table to CSV and XLSX (Step 13 requirement)."""
    csv_path = C.TABLES_DIR / f"{stem}.csv"
    xlsx_path = C.TABLES_DIR / f"{stem}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path}
