"""Static-threshold alerting: the way a typical NOC alerts today, and the baseline
every ML model here must beat (Phase 3) and is A/B tested against (Phase 6).

Deliberately simple and fleet-wide: one set of thresholds for every router. That is
its weakness: 40 ms is an outage on a core router and a normal day on an edge router.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = {
    "latency_ms": 45.0,        # above typical edge latency at peak
    "packet_loss_pct": 1.0,    # >1% loss is a classic SLA breach line
    "route_flap_count": 3,     # flaps in a single 5-minute sample
}


def rule_alerts(df: pd.DataFrame, t: dict = THRESHOLDS):
    """Returns (score, alert, predicted_type). Checks in priority order, so a row
    matching several rules is typed by the most severe one."""
    session_down = (df["bgp_session_state"] != "Established").to_numpy()
    flaps = (df["route_flap_count"] >= t["route_flap_count"]).to_numpy()
    loss = (df["packet_loss_pct"] > t["packet_loss_pct"]).to_numpy()
    latency = (df["latency_ms"] > t["latency_ms"]).to_numpy()

    ptype = np.select(
        [flaps, session_down, loss, latency],
        ["route_flap_storm", "session_down", "packet_loss_burst", "latency_spike"],
        default="normal",
    )
    alert = ptype != "normal"
    # A graded score (how many rules fired) so PR-AUC is computable for the baseline.
    score = session_down.astype(float) + flaps + loss + latency
    return score, alert, ptype
