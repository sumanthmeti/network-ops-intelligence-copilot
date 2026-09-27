"""Generate simulated BGP/OSPF router telemetry with labeled anomalies.

Each router emits one sample per interval with:
    latency_ms, packet_loss_pct, route_flap_count, bgp_session_state, traffic_mbps

Normal behaviour is driven by the router's role (core / aggregation / edge) plus a
diurnal traffic cycle, so "normal" is not a flat line -- a detector has to learn
that 14:00 traffic is higher than 03:00 traffic.

Anomalies are injected as *episodes* (a contiguous run of intervals on one router),
not as isolated points, because that is how real incidents look: a BGP session
doesn't go down for exactly one 5-minute sample. Every anomalous row is labeled
with is_anomaly / anomaly_type / episode_id so the Phase 3 model and the Phase 6
A/B test have ground truth to score against.

Defaults (200 routers, 7 days, 5-minute interval) produce 403,200 rows.

Usage:
    python data/generate_telemetry.py
    python data/generate_telemetry.py --routers 50 --days 1 --out data/raw/small.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = np.array(["core", "aggregation", "edge"])
ROLE_WEIGHTS = [0.15, 0.35, 0.50]
SITES = np.array(["BLR", "HYD", "MUM", "DEL", "CHN", "PUN"])

# Per-role baselines. Core routers carry more traffic and sit on shorter,
# better-provisioned paths; edge routers are the opposite.
ROLE_BASE_LATENCY_MS = {"core": 4.0, "aggregation": 9.0, "edge": 18.0}
ROLE_CAPACITY_MBPS = {"core": 40_000.0, "aggregation": 10_000.0, "edge": 1_000.0}

ANOMALY_TYPES = ["latency_spike", "packet_loss_burst", "route_flap_storm", "session_down"]

# BGP FSM states. Established is the only healthy steady state.
BGP_ESTABLISHED = "Established"
BGP_DOWN_STATES = np.array(["Idle", "Active", "Connect"])
BGP_FLAPPING_STATES = np.array(["Established", "Active", "OpenSent", "OpenConfirm"])


@dataclass(frozen=True)
class Config:
    routers: int = 200
    days: float = 7.0
    interval_minutes: int = 5
    start: str = "2026-09-01T00:00:00Z"
    # Expected anomaly episodes per router per day.
    episodes_per_router_day: float = 0.25
    min_episode_intervals: int = 3
    max_episode_intervals: int = 18
    seed: int = 42


def _router_metadata(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    roles = rng.choice(ROLES, size=cfg.routers, p=ROLE_WEIGHTS)
    return pd.DataFrame(
        {
            "router_id": [f"R{i:04d}" for i in range(1, cfg.routers + 1)],
            "site": rng.choice(SITES, size=cfg.routers),
            "role": roles,
            # Each router gets its own baseline jitter so routers of the same role
            # are not identical -- this is why features must be computed per router.
            "base_latency": np.array([ROLE_BASE_LATENCY_MS[r] for r in roles])
            * rng.uniform(0.7, 1.4, cfg.routers),
            "capacity": np.array([ROLE_CAPACITY_MBPS[r] for r in roles]),
            "base_util": rng.uniform(0.25, 0.55, cfg.routers),
        }
    )


def generate(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    meta = _router_metadata(cfg, rng)

    n_steps = int(cfg.days * 24 * 60 / cfg.interval_minutes)
    timestamps = pd.date_range(cfg.start, periods=n_steps, freq=f"{cfg.interval_minutes}min")
    R, T = cfg.routers, n_steps

    # --- Normal behaviour: all arrays are shape (R, T) -------------------------
    hour = (timestamps.hour + timestamps.minute / 60.0).to_numpy()
    # Diurnal load curve: trough ~04:00, peak ~16:00, range roughly 0.55x..1.45x.
    diurnal = 1.0 + 0.45 * np.sin(2 * np.pi * (hour - 10.0) / 24.0)
    util = meta["base_util"].to_numpy()[:, None] * diurnal[None, :]
    util = np.clip(util * rng.normal(1.0, 0.05, (R, T)), 0.01, 0.98)
    traffic = util * meta["capacity"].to_numpy()[:, None]

    # Latency rises non-linearly as links approach saturation (queueing delay).
    base_lat = meta["base_latency"].to_numpy()[:, None]
    latency = base_lat * (1.0 + 0.6 * util**3) + rng.gamma(2.0, 0.15 * base_lat, (R, T)) * 0.3

    # Packet loss is near zero normally, with a long right tail.
    packet_loss = rng.exponential(0.02, (R, T)) * (1.0 + 2.0 * util**4)

    # Occasional isolated flaps are normal background noise on a large fleet.
    flaps = rng.poisson(0.03, (R, T))

    bgp_state = np.full((R, T), BGP_ESTABLISHED, dtype=object)
    anomaly_type = np.full((R, T), None, dtype=object)
    episode_id = np.full((R, T), -1, dtype=np.int64)

    # --- Inject labeled anomaly episodes ---------------------------------------
    n_episodes = rng.poisson(cfg.episodes_per_router_day * cfg.days, R)
    next_episode = 0
    for r in range(R):
        for _ in range(n_episodes[r]):
            dur = int(rng.integers(cfg.min_episode_intervals, cfg.max_episode_intervals + 1))
            start = int(rng.integers(0, T - dur))
            sl = slice(start, start + dur)
            kind = ANOMALY_TYPES[rng.integers(len(ANOMALY_TYPES))]

            if kind == "latency_spike":
                latency[r, sl] *= rng.uniform(3.0, 8.0, dur)
            elif kind == "packet_loss_burst":
                packet_loss[r, sl] = rng.uniform(2.0, 15.0, dur)
                latency[r, sl] *= rng.uniform(1.1, 1.6, dur)  # retransmits add delay
            elif kind == "route_flap_storm":
                flaps[r, sl] = rng.integers(8, 50, dur)
                bgp_state[r, sl] = rng.choice(BGP_FLAPPING_STATES, dur)
                latency[r, sl] *= rng.uniform(1.3, 2.5, dur)  # reconvergence
                packet_loss[r, sl] += rng.uniform(0.3, 3.0, dur)
            elif kind == "session_down":
                bgp_state[r, sl] = rng.choice(BGP_DOWN_STATES, dur)
                # Routes are withdrawn, so traffic through this router collapses.
                traffic[r, sl] *= rng.uniform(0.0, 0.08, dur)
                flaps[r, start] += 1  # the session dropping is itself a flap

            anomaly_type[r, sl] = kind
            episode_id[r, sl] = next_episode
            next_episode += 1

    # --- Flatten to long format (one row per router per timestamp) -------------
    df = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps.to_numpy(), R),
            "router_id": np.repeat(meta["router_id"].to_numpy(), T),
            "site": np.repeat(meta["site"].to_numpy(), T),
            "role": np.repeat(meta["role"].to_numpy(), T),
            "latency_ms": latency.ravel().round(3),
            "packet_loss_pct": np.clip(packet_loss.ravel(), 0.0, 100.0).round(4),
            "route_flap_count": flaps.ravel().astype(np.int32),
            "bgp_session_state": bgp_state.ravel(),
            "traffic_mbps": traffic.ravel().round(2),
            "anomaly_type": anomaly_type.ravel(),
            "episode_id": episode_id.ravel(),
        }
    )
    df["is_anomaly"] = df["anomaly_type"].notna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--routers", type=int, default=Config.routers)
    p.add_argument("--days", type=float, default=Config.days)
    p.add_argument("--interval-minutes", type=int, default=Config.interval_minutes)
    p.add_argument("--start", default=Config.start)
    p.add_argument("--episodes-per-router-day", type=float, default=Config.episodes_per_router_day)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--out", type=Path, default=Path("data/raw/telemetry.parquet"))
    a = p.parse_args()

    cfg = Config(
        routers=a.routers,
        days=a.days,
        interval_minutes=a.interval_minutes,
        start=a.start,
        episodes_per_router_day=a.episodes_per_router_day,
        seed=a.seed,
    )
    df = generate(cfg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # pandas timestamps are nanosecond precision, but Spark 3.x only reads
    # microsecond Parquet timestamps ("Illegal Parquet type: INT64 (TIMESTAMP(NANOS,true))").
    # Coerce at write time so any Spark/Athena/Glue consumer can read the file.
    df.to_parquet(a.out, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)

    print(f"Wrote {len(df):,} rows for {cfg.routers} routers -> {a.out}")
    print(f"Anomalous rows: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.2%}), "
          f"episodes: {df.loc[df['is_anomaly'], 'episode_id'].nunique():,}")
    print(df.loc[df["is_anomaly"]].groupby("anomaly_type")["episode_id"].nunique().to_string())


if __name__ == "__main__":
    main()
