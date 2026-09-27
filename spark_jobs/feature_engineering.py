"""PySpark feature engineering: per-router rolling-window features for anomaly detection.

Why Spark here (and not pandas):
    The core operation is "for every router, compute rolling statistics over its own
    recent history". Routers are independent of each other, so the work partitions
    cleanly by router_id: Spark shuffles rows so each router's history lands in one
    partition, sorts it by time, and computes windows on all partitions in parallel.
    The simulated fleet (200 routers x 7 days x 5 min = ~400K rows) fits in pandas
    fine -- the justification is production scale: thousands of routers at sub-minute
    intervals with weeks of retention is tens to hundreds of millions of rows, past
    comfortable single-machine memory. The same code runs unchanged on a cluster
    (EMR / Glue / Databricks); only the master URL and input path change.

Design decisions worth being able to defend:
    * Time-based windows (rangeBetween on epoch seconds), not row-count windows
      (rowsBetween). If a router misses samples, "last 12 rows" silently becomes
      "last 3 hours"; "last 3600 seconds" stays correct.
    * The baseline window EXCLUDES the current row (ends at -1 second). Otherwise a
      spike inflates its own mean and std and hides itself -- z-scores get dampened
      exactly when they matter most.
    * Z-scores are null (not 0, not infinity) when the router has too little
      history or zero variance. The model downstream decides how to treat nulls;
      the feature job does not invent values.
    * Output is partitioned by date, not router_id. Partitioning by router_id would
      create 200+ directories of tiny files (the small-files problem) and most
      downstream reads are time-range reads ("train on last week").

Usage:
    python spark_jobs/feature_engineering.py
    python spark_jobs/feature_engineering.py --input data/raw/telemetry.parquet \\
        --output data/features --window-minutes 60 --master local[*]
    # AWS Glue / EMR: no --master, S3 paths (see infra/glue.tf)
    --input s3://<bucket>/raw/telemetry.parquet --output s3://<bucket>/features
"""

from __future__ import annotations

import argparse
import time

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

# Metrics that get mean / std / z-score treatment.
CONTINUOUS_METRICS = ["latency_ms", "packet_loss_pct", "traffic_mbps"]


def build_features(
    df: DataFrame,
    window_minutes: int = 60,
    short_window_minutes: int = 15,
    min_history: int = 6,
) -> DataFrame:
    """Add per-router rolling features. Pure DataFrame -> DataFrame, so it is unit-testable.

    Args:
        df: raw telemetry with at least router_id, timestamp, the metrics above,
            route_flap_count and bgp_session_state.
        window_minutes: length of the baseline window.
        short_window_minutes: length of the short window used for flap bursts.
        min_history: minimum samples in the baseline window before a z-score is emitted.
    """
    df = df.withColumn("ts_epoch", F.col("timestamp").cast("long"))

    by_router = Window.partitionBy("router_id").orderBy("ts_epoch")
    # Trailing baseline: [t - window, t - 1s]. Excludes the current sample.
    baseline = by_router.rangeBetween(-window_minutes * 60, -1)
    # Trailing inclusive windows for counts: [t - window, t].
    long_incl = by_router.rangeBetween(-window_minutes * 60, 0)
    short_incl = by_router.rangeBetween(-short_window_minutes * 60, 0)

    df = df.withColumn("baseline_count", F.count("ts_epoch").over(baseline))

    for m in CONTINUOUS_METRICS:
        mean_col, std_col = f"{m}_mean_{window_minutes}m", f"{m}_std_{window_minutes}m"
        df = (
            df.withColumn(mean_col, F.avg(m).over(baseline))
            .withColumn(std_col, F.stddev_samp(m).over(baseline))
            .withColumn(
                f"{m}_zscore",
                F.when(
                    (F.col("baseline_count") >= min_history) & (F.col(std_col) > 0),
                    (F.col(m) - F.col(mean_col)) / F.col(std_col),
                ),
            )
        )

    session_down = (F.col("bgp_session_state") != "Established").cast("int")
    prev_state = F.lag("bgp_session_state").over(by_router)
    prev_traffic = F.lag("traffic_mbps").over(by_router)

    df = (
        df.withColumn(f"route_flap_sum_{window_minutes}m", F.sum("route_flap_count").over(long_incl))
        .withColumn(f"route_flap_sum_{short_window_minutes}m", F.sum("route_flap_count").over(short_incl))
        .withColumn("bgp_session_down", session_down)
        .withColumn(
            "bgp_state_changed",
            F.when(prev_state.isNull(), 0)
            .when(F.col("bgp_session_state") != prev_state, 1)
            .otherwise(0),
        )
        .withColumn(
            f"bgp_state_changes_{window_minutes}m",
            F.sum("bgp_state_changed").over(long_incl),
        )
        .withColumn(
            "traffic_pct_change",
            F.when(prev_traffic > 0, (F.col("traffic_mbps") - prev_traffic) / prev_traffic),
        )
        .withColumn("hour_of_day", F.hour("timestamp"))
        .withColumn("date", F.to_date("timestamp"))
        .drop("ts_epoch", "bgp_state_changed")
    )
    return df


def create_spark(master: str | None, shuffle_partitions: int) -> SparkSession:
    builder = SparkSession.builder.appName("network-ops-feature-engineering")
    # Only set the master when explicitly asked (local runs). On EMR Serverless /
    # spark-submit the platform supplies it; hard-coding local[*] here would
    # silently run the whole job on the driver and ignore the executors.
    if master:
        builder = builder.master(master)
    return (
        builder
        # Default is 200 shuffle partitions -- tuned for clusters. On one machine
        # that means 200 tiny tasks with scheduling overhead dominating.
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Pin the session timezone so hour_of_day / date don't depend on the laptop.
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/raw/telemetry.parquet")
    p.add_argument("--output", default="data/features")
    p.add_argument("--window-minutes", type=int, default=60)
    p.add_argument("--short-window-minutes", type=int, default=15)
    p.add_argument("--master", default=None, help="e.g. local[*]; omit on EMR/spark-submit")
    p.add_argument("--shuffle-partitions", type=int, default=16)
    # parse_known_args, not parse_args: AWS Glue appends its own arguments
    # (--JOB_ID, --JOB_RUN_ID, --job-bookmark-option, ...) to every run.
    a, _ = p.parse_known_args()

    spark = create_spark(a.master, a.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    raw = spark.read.parquet(a.input)
    features = build_features(raw, a.window_minutes, a.short_window_minutes)
    features.write.mode("overwrite").partitionBy("date").parquet(a.output)

    out = spark.read.parquet(a.output)
    n = out.count()
    print(f"Wrote {n:,} rows, {len(out.columns)} columns -> {a.output} in {time.time() - t0:.1f}s")
    print("Mean |z| by anomaly type (sanity check -- anomalies should be far from 0):")
    (
        out.groupBy(F.coalesce("anomaly_type", F.lit("normal")).alias("anomaly_type"))
        .agg(
            *[F.round(F.avg(F.abs(f"{m}_zscore")), 2).alias(f"|z| {m}") for m in CONTINUOUS_METRICS],
            F.round(F.avg(f"route_flap_sum_{a.short_window_minutes}m"), 2).alias("flaps_15m"),
            F.round(F.avg("bgp_session_down"), 2).alias("session_down"),
        )
        .orderBy("anomaly_type")
        .show(truncate=False)
    )
    spark.stop()


if __name__ == "__main__":
    main()
