import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "data"))
from features import FEATURES, to_matrix  # noqa: E402
from rules import rule_alerts  # noqa: E402
from train import class_weights, fit, pick_threshold, score, split_by_time  # noqa: E402


def feature_frame(n=60, seed=0, anomalies=()):
    """Tiny frame with every column the model needs; rows in `anomalies` get a clear signature."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "router_id": "R1", "role": "edge", "date": "2026-09-01",
        "latency_ms": rng.normal(20, 1, n), "latency_ms_mean_60m": 20.0,
        "traffic_mbps": rng.normal(500, 10, n), "traffic_mbps_mean_60m": 500.0,
        "latency_ms_zscore": rng.normal(0, 1, n), "packet_loss_pct_zscore": rng.normal(0, 1, n),
        "traffic_mbps_zscore": rng.normal(0, 1, n), "packet_loss_pct": 0.02,
        "route_flap_sum_60m": 0, "route_flap_sum_15m": 0, "bgp_session_down": 0,
        "bgp_state_changes_60m": 0, "traffic_pct_change": 0.0, "hour_of_day": 12,
        "route_flap_count": 0, "bgp_session_state": "Established",
        "anomaly_type": None, "is_anomaly": False,
    })
    for i in anomalies:
        df.loc[i, ["packet_loss_pct", "packet_loss_pct_zscore"]] = [8.0, 30.0]
        df.loc[i, ["anomaly_type", "is_anomaly"]] = ["packet_loss_burst", True]
    return df


def test_split_by_time_has_no_overlap_and_test_is_latest():
    df = pd.DataFrame({"date": [f"2026-09-0{d}" for d in range(1, 8) for _ in range(3)]})
    train, val, test = split_by_time(df)
    assert set(test["date"]) == {"2026-09-06", "2026-09-07"}
    assert set(val["date"]) == {"2026-09-05"}
    assert train["date"].max() < val["date"].min() < test["date"].min()


def test_class_weights_power():
    y = np.array(["normal"] * 99 + ["spike"])
    assert np.allclose(class_weights(y, 0.0), 1.0)
    w = class_weights(y, 1.0)
    # Fully balanced: each class carries the same total weight.
    assert w[y == "normal"].sum() == pytest.approx(w[y == "spike"].sum())
    assert class_weights(y, 0.5)[-1] < w[-1]


def test_pick_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    t = pick_threshold(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert 0.3 < t <= 0.8


def test_to_matrix_keeps_column_order_and_nans():
    df = feature_frame(3)
    df.loc[0, "latency_ms_zscore"] = np.nan
    X = to_matrix(df)
    assert X.shape == (3, len(FEATURES))
    assert np.isnan(X[0, FEATURES.index("latency_ms_zscore")])


def test_unweighted_boosting_does_not_diverge():
    """Regression test: with l2_regularization=0, near-pure leaves made Newton steps
    explode (raw scores ~1e6) and the anomaly score collapsed to exactly 0."""
    df = pd.concat([feature_frame(400, seed=s, anomalies=range(0, 400, 50)) for s in range(3)], ignore_index=True)
    p = Namespace(model_type="hgb", learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
                  l2_regularization=1.0, class_weight_power=0.0)
    model = fit(df, p)
    assert np.abs(model._raw_predict(to_matrix(df))).max() < 1e3
    s, ptype = score(model, df)
    assert s[df["is_anomaly"]].min() > s[~df["is_anomaly"]].max()
    assert (ptype[df["is_anomaly"].to_numpy()] == "packet_loss_burst").all()


def test_rules_priority_and_types():
    df = feature_frame(4)
    df.loc[0, "latency_ms"] = 100
    df.loc[1, "packet_loss_pct"] = 5
    df.loc[2, ["route_flap_count", "bgp_session_state"]] = [10, "Active"]  # flap wins over session
    _, alert, ptype = rule_alerts(df)
    assert list(ptype) == ["latency_spike", "packet_loss_burst", "route_flap_storm", "normal"]
    assert list(alert) == [True, True, True, False]
