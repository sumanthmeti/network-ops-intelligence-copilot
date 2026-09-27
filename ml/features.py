"""Single source of truth for model inputs, imported by training AND serving.

Train/serve skew (the endpoint computing a feature slightly differently from the
training code) is one of the most common silent ML production bugs. Keeping the
feature list and the derivations in one module that both sides import prevents it.

Must stay compatible with the SageMaker scikit-learn container's Python/pandas,
so: no match statements, no 3.10+ only syntax at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COL = "anomaly_type"
NORMAL = "normal"
CLASSES = [NORMAL, "latency_spike", "packet_loss_burst", "route_flap_storm", "session_down"]

# Columns produced by spark_jobs/feature_engineering.py that the model uses.
SPARK_FEATURES = [
    "latency_ms_zscore",
    "packet_loss_pct_zscore",
    "traffic_mbps_zscore",
    "packet_loss_pct",          # absolute loss matters, not just relative: 5% is bad on any router
    "route_flap_sum_60m",
    "route_flap_sum_15m",
    "bgp_session_down",
    "bgp_state_changes_60m",
    "traffic_pct_change",
    "hour_of_day",
]

# Derived here (cheap, row-local) rather than in Spark.
DERIVED_FEATURES = ["latency_ratio", "traffic_ratio", "role_core", "role_edge"]

FEATURES = SPARK_FEATURES + DERIVED_FEATURES


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios to the router's own baseline are role-independent, unlike raw latency:
    25 ms is a 5x spike on a core router and a normal reading on an edge router."""
    out = df.copy()
    lat_mean = out["latency_ms_mean_60m"].replace(0, np.nan)
    trf_mean = out["traffic_mbps_mean_60m"].replace(0, np.nan)
    out["latency_ratio"] = out["latency_ms"] / lat_mean
    out["traffic_ratio"] = out["traffic_mbps"] / trf_mean
    out["role_core"] = (out["role"] == "core").astype(int)
    out["role_edge"] = (out["role"] == "edge").astype(int)
    return out


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix in the exact column order the model was trained with.
    Missing values stay NaN: HistGradientBoosting handles them natively, and early
    rows with no history legitimately have no z-score."""
    return add_derived(df)[FEATURES].astype(float).to_numpy()


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COL].fillna(NORMAL).to_numpy()
