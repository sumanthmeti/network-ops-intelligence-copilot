"""Unit tests for the Spark feature logic on tiny hand-built DataFrames, so the
expected values can be checked by hand. No file I/O, so these run on Windows
without Hadoop's winutils."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spark_jobs"))
from feature_engineering import build_features  # noqa: E402

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
COLS = ["router_id", "timestamp", "latency_ms", "packet_loss_pct", "traffic_mbps",
        "route_flap_count", "bgp_session_state"]


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[2]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.session.timeZone", "UTC")
         # Arrow ships pandas frames straight to the JVM. Without it, createDataFrame
         # spins up Python worker processes, which crash on Windows with PySpark 3.5.
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
         .getOrCreate())
    yield s
    s.stop()


def row(router, minute, latency=10.0, loss=0.0, traffic=100.0, flaps=0, state="Established"):
    return (router, T0 + timedelta(minutes=minute), latency, loss, traffic, flaps, state)


def run(spark, rows, **kw):
    out = build_features(spark.createDataFrame(pd.DataFrame(rows, columns=COLS)), **kw)
    return {(r.router_id, r.timestamp.astimezone(timezone.utc)): r for r in out.collect()}


def at(result, router, minute):
    return result[(router, T0 + timedelta(minutes=minute))]


def test_baseline_excludes_current_row(spark):
    # 6 samples of 10ms then a 100ms spike. If the spike were in its own window,
    # the mean would be ~22.9; it must be exactly 10 (the prior samples only).
    rows = [row("R1", 5 * i) for i in range(6)] + [row("R1", 30, latency=100.0)]
    r = at(run(spark, rows, window_minutes=60, min_history=3), "R1", 30)
    assert r.latency_ms_mean_60m == pytest.approx(10.0)


def test_zscore_matches_hand_calculation(spark):
    lat = [10, 12, 10, 12, 10, 12]  # mean 11, sample std sqrt(1.2)
    rows = [row("R1", 5 * i, latency=float(v)) for i, v in enumerate(lat)] + [row("R1", 30, latency=20.0)]
    r = at(run(spark, rows, window_minutes=60, min_history=3), "R1", 30)
    assert r.latency_ms_zscore == pytest.approx((20 - 11) / 1.2 ** 0.5)


def test_zscore_null_without_enough_history_or_variance(spark):
    rows = [row("R1", 5 * i) for i in range(8)]  # constant latency -> std 0
    res = run(spark, rows, window_minutes=60, min_history=6)
    assert at(res, "R1", 10).latency_ms_zscore is None   # only 2 prior samples
    assert at(res, "R1", 35).latency_ms_zscore is None   # enough samples, zero variance


def test_windows_are_time_based_not_row_based(spark):
    # Samples at t=0 and t=5, then a 3-hour gap. With a 60-minute window the sample
    # at t=185 has NO history; a rowsBetween window would wrongly include t=0/t=5.
    rows = [row("R1", 0, flaps=5), row("R1", 5, flaps=5), row("R1", 185, flaps=1)]
    r = at(run(spark, rows, window_minutes=60), "R1", 185)
    assert r.baseline_count == 0
    assert r.route_flap_sum_60m == 1


def test_routers_do_not_leak_into_each_other(spark):
    rows = [row("R1", 5 * i, latency=10.0) for i in range(4)] + \
           [row("R2", 5 * i, latency=500.0) for i in range(4)]
    res = run(spark, rows, window_minutes=60)
    assert at(res, "R1", 15).latency_ms_mean_60m == pytest.approx(10.0)
    assert at(res, "R2", 15).latency_ms_mean_60m == pytest.approx(500.0)


def test_flap_sums_and_bgp_state_changes(spark):
    rows = [
        row("R1", 0),
        row("R1", 5, flaps=3, state="Active"),
        row("R1", 10, flaps=4, state="Established"),
        row("R1", 30, flaps=2),
    ]
    res = run(spark, rows, window_minutes=60, short_window_minutes=15)
    r = at(res, "R1", 30)
    assert r.route_flap_sum_60m == 9
    assert r.route_flap_sum_15m == 2          # only t=30 is within [15, 30]
    assert r.bgp_state_changes_60m == 2       # Established->Active->Established
    assert at(res, "R1", 5).bgp_session_down == 1


def test_traffic_pct_change(spark):
    rows = [row("R1", 0, traffic=100.0), row("R1", 5, traffic=5.0)]
    assert at(run(spark, rows), "R1", 5).traffic_pct_change == pytest.approx(-0.95)
