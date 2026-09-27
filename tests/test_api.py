import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.main import TelemetryRow, create_app  # noqa: E402
from agent.change_calendar import ChangeCalendar  # noqa: E402
from agent.llm import OfflineTemplateLLM  # noqa: E402
from tests.test_agent import FakeDetector, row  # noqa: E402


@pytest.fixture
def client():
    app = create_app(FakeDetector(), OfflineTemplateLLM(fail_first=True), ChangeCalendar([]))
    with TestClient(app) as c:
        yield c


def test_probes(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"


def test_detect(client):
    r = client.post("/v1/detect", json={"rows": [row(), row("R0002")]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["router_id"] for p in preds] == ["R0001", "R0002"] and preds[0]["alert"] is True


def test_triage_runs_the_agent_including_retry(client):
    body = client.post("/v1/triage", json=row()).json()
    assert body["outcome"] == "ticketed" and body["severity"] == "P1"
    assert body["attempts"] == 2 and len(body["validation_errors"]) == 1
    assert body["report"]["router_id"] == "R0001"
    assert client.get("/v1/incidents").json()[0]["outcome"] == "ticketed"
    assert 'noc_triage_outcomes_total{outcome="ticketed"} 1' in client.get("/metrics").text


def test_request_validation_rejects_bad_input(client):
    assert client.post("/v1/triage", json=row() | {"router_id": "router-1"}).status_code == 422
    assert client.post("/v1/triage", json=row() | {"packet_loss_pct": 150}).status_code == 422
    assert client.post("/v1/detect", json={"rows": []}).status_code == 422


@pytest.mark.skipif(not (ROOT / "artifacts/model/model.joblib").exists(), reason="needs trained model")
def test_real_model_end_to_end(monkeypatch):
    """Integration: real champion model + real calendar, via the default component loader."""
    import pandas as pd
    if not (ROOT / "data/raw/change_calendar.json").exists():
        ChangeCalendar.build_from_telemetry(pd.read_parquet(ROOT / "data/raw/telemetry.parquet"))
    preds = pd.read_parquet(ROOT / "artifacts/output/test_predictions.parquet")
    hit = preds[preds["alert"] & (preds["anomaly_type"] == "session_down")].iloc[0]
    payload = hit.drop(["anomaly_score", "alert", "predicted_type"]).to_dict()
    payload = {k: (None if pd.isna(v) else v) for k, v in payload.items() if k in
               TelemetryRow.model_fields}
    payload["timestamp"] = str(hit["timestamp"])
    with TestClient(create_app()) as c:
        body = c.post("/v1/triage", json=payload).json()
    assert body["predicted_type"] == "session_down" and body["outcome"] in ("ticketed", "suppressed")
