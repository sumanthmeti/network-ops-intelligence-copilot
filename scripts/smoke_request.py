"""Send real test-day rows to a running API and check the answers make sense.

    python scripts/smoke_request.py http://localhost:8000

Used by CI against the Docker container and against the Kubernetes Service.
Exits non-zero if anything is off.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.main import TelemetryRow  # noqa: E402


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def payload(row: pd.Series) -> dict:
    d = {k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k]))
         for k in TelemetryRow.model_fields if k != "timestamp"}
    d["timestamp"] = str(row["timestamp"])
    return d


def main(base: str) -> None:
    preds = pd.read_parquet(ROOT / "artifacts/output/test_predictions.parquet")
    incident = preds[preds["alert"] & preds["is_anomaly"] & (preds["anomaly_type"] == "route_flap_storm")].iloc[0]
    quiet = preds[~preds["alert"] & ~preds["is_anomaly"]].iloc[0]

    r = post(f"{base}/v1/triage", payload(incident))
    print("incident ->", {k: r[k] for k in ["outcome", "severity", "predicted_type", "attempts", "latency_ms"]})
    assert r["predicted_type"] == "route_flap_storm", r
    assert r["outcome"] in ("ticketed", "logged"), r

    r = post(f"{base}/v1/triage", payload(quiet))
    print("quiet    ->", {k: r[k] for k in ["outcome", "anomaly_score"]})
    assert r["outcome"] == "no_incident", r

    r = post(f"{base}/v1/detect", {"rows": [payload(incident), payload(quiet)]})
    assert [p["alert"] for p in r["predictions"]] == [True, False], r
    print("smoke test passed")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
