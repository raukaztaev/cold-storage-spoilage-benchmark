"""Computational profiling (Step 12): train/predict time, model size, memory,
and a simple deployment-feasibility view for IoT edge devices."""
from __future__ import annotations

import io
import pickle
import time
import tracemalloc
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config as C


def profile_models(models: Dict, x_train, y_train, x_test, y_test,
                   model_names: List[str], n_predict_repeats: int = 5) -> pd.DataFrame:
    """Measure fit/predict latency, peak memory during fit, and serialized size."""
    rows = []
    for name in model_names:
        pipe = clone(models[name])

        tracemalloc.start()
        t0 = time.perf_counter()
        pipe.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Average single-batch predict latency over repeats.
        times = []
        for _ in range(n_predict_repeats):
            t0 = time.perf_counter()
            pipe.predict(x_test)
            times.append(time.perf_counter() - t0)
        predict_time = float(np.mean(times))
        per_sample_ms = 1000.0 * predict_time / len(x_test)

        buf = io.BytesIO()
        pickle.dump(pipe, buf)
        size_kb = buf.tell() / 1024.0

        rows.append({
            "Model": name,
            "Fit Time (s)": round(fit_time, 4),
            "Predict Time (s)": round(predict_time, 5),
            "Latency (ms/sample)": round(per_sample_ms, 5),
            "Throughput (samples/s)": round(len(x_test) / predict_time, 1),
            "Peak Fit Memory (MB)": round(peak / 1e6, 3),
            "Model Size (KB)": round(size_kb, 2),
        })
    return pd.DataFrame(rows).sort_values("Model Size (KB)").reset_index(drop=True)


def deployment_view(profile_df: pd.DataFrame, size_budget_kb: float = 500.0,
                    latency_budget_ms: float = 1.0) -> pd.DataFrame:
    """Flag which models fit a nominal edge budget (size + per-sample latency)."""
    d = profile_df.copy()
    d["edge_feasible"] = (d["Model Size (KB)"] <= size_budget_kb) & \
                         (d["Latency (ms/sample)"] <= latency_budget_ms)
    return d
