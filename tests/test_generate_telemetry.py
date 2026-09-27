import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from generate_telemetry import ANOMALY_TYPES, Config, generate  # noqa: E402

SMALL = Config(routers=20, days=2, episodes_per_router_day=1.0, seed=7)


@pytest.fixture(scope="module")
def df():
    return generate(SMALL)


def test_row_count_is_routers_times_intervals(df):
    assert len(df) == 20 * 2 * 24 * 12


def test_one_sample_per_router_per_interval(df):
    assert not df.duplicated(["router_id", "timestamp"]).any()


def test_same_seed_is_reproducible():
    pd.testing.assert_frame_equal(generate(SMALL), generate(SMALL))


def test_labels_are_consistent(df):
    assert (df["is_anomaly"] == df["anomaly_type"].notna()).all()
    assert (df.loc[df["is_anomaly"], "episode_id"] >= 0).all()
    assert (df.loc[~df["is_anomaly"], "episode_id"] == -1).all()
    assert set(df["anomaly_type"].dropna()) <= set(ANOMALY_TYPES)


def test_metric_ranges(df):
    assert (df["packet_loss_pct"].between(0, 100)).all()
    assert (df["latency_ms"] > 0).all()
    assert (df["traffic_mbps"] >= 0).all()
    assert (df["route_flap_count"] >= 0).all()


def test_healthy_rows_are_established_except_maintenance(df):
    healthy = df[~df["is_anomaly"] & (df["benign_event"] != "maintenance")]
    assert (healthy["bgp_session_state"] == "Established").all()


def test_benign_events_never_overlap_incidents(df):
    assert df.loc[df["is_anomaly"], "benign_event"].isna().all()
    assert {"microburst", "traffic_surge", "maintenance"} <= set(df["benign_event"].dropna())


def test_maintenance_happens_in_night_window(df):
    hours = df.loc[df["benign_event"] == "maintenance", "timestamp"].dt.hour
    assert hours.max() < 5


def test_anomalies_actually_move_their_metric(df):
    """Each anomaly type must be distinguishable from normal on its signature metric,
    otherwise downstream detectors are being scored against meaningless labels."""
    normal = df[~df["is_anomaly"]]
    by = {k: g for k, g in df[df["is_anomaly"]].groupby("anomaly_type")}
    assert by["latency_spike"]["latency_ms"].median() > 2.5 * normal["latency_ms"].median()
    assert by["packet_loss_burst"]["packet_loss_pct"].median() > 50 * normal["packet_loss_pct"].median()
    assert by["route_flap_storm"]["route_flap_count"].median() >= 8
    assert (by["session_down"]["bgp_session_state"] != "Established").all()
