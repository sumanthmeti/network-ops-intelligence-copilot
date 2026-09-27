"""FastAPI service: anomaly detection + incident triage over HTTP.

    GET  /healthz      liveness: the process is up (K8s restarts the pod if this fails)
    GET  /readyz       readiness: model + calendar loaded (K8s only routes traffic when this passes)
    POST /v1/detect    score telemetry rows with the anomaly model
    POST /v1/triage    run one row through the full LangGraph agent
    GET  /v1/incidents recent triage outcomes held by THIS pod (in memory, per replica;
                       a real deployment would write to a database or queue instead)
    GET  /metrics      Prometheus-format counters

Configuration is environment-only (12-factor), so the same image runs everywhere:
    LLM_BACKEND=offline|bedrock   DETECTOR_BACKEND=local|sagemaker
    MODEL_PATH, CHANGE_CALENDAR_PATH, BEDROCK_MODEL, AWS_REGION
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.change_calendar import ChangeCalendar  # noqa: E402
from agent.graph import build_graph  # noqa: E402


class TelemetryRow(BaseModel):
    """One router sample with the Spark features. Nullable features are legitimately
    missing (e.g. no z-score in the first hour of history); the model handles NaN."""
    router_id: str = Field(pattern=r"^R\d{4}$")
    site: str
    role: str = Field(pattern="^(core|aggregation|edge)$")
    timestamp: str
    latency_ms: float
    packet_loss_pct: float = Field(ge=0, le=100)
    traffic_mbps: float = Field(ge=0)
    route_flap_count: int = Field(ge=0)
    bgp_session_state: str
    latency_ms_mean_60m: float | None = None
    traffic_mbps_mean_60m: float | None = None
    latency_ms_zscore: float | None = None
    packet_loss_pct_zscore: float | None = None
    traffic_mbps_zscore: float | None = None
    route_flap_sum_60m: float | None = None
    route_flap_sum_15m: float | None = None
    bgp_session_down: int | None = None
    bgp_state_changes_60m: float | None = None
    traffic_pct_change: float | None = None
    hour_of_day: int | None = Field(default=None, ge=0, le=23)


class DetectRequest(BaseModel):
    rows: list[TelemetryRow] = Field(min_length=1, max_length=1000)


def _default_components():
    from agent.detector import make_detector
    from agent.llm import BedrockClaude, OfflineTemplateLLM
    detector = make_detector()
    llm = BedrockClaude() if os.environ.get("LLM_BACKEND") == "bedrock" else OfflineTemplateLLM()
    calendar = ChangeCalendar.load(os.environ.get("CHANGE_CALENDAR_PATH", ROOT / "data/raw/change_calendar.json"))
    return detector, llm, calendar


def create_app(detector=None, llm=None, calendar=None) -> FastAPI:
    """Components are injectable so tests can run without a trained model or an LLM."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        d, l, c = (detector, llm, calendar) if detector else _default_components()
        app.state.detector = d
        app.state.incidents = deque(maxlen=200)
        app.state.counts = Counter()

        def sink(outcome, state):
            app.state.counts[outcome] += 1
            app.state.incidents.appendleft({"outcome": outcome, "severity": state.get("severity"),
                                            "router_id": state["alert"]["router_id"],
                                            "timestamp": state["alert"]["timestamp"],
                                            "attempts": state.get("attempts", 0),
                                            "change_id": state.get("change_id"),
                                            "report": state.get("report")})
        app.state.graph = build_graph(d, l, c or ChangeCalendar([]), sink=sink)
        app.state.llm_backend = l.__class__.__name__
        app.state.ready = True
        yield

    app = FastAPI(title="Network Ops Intelligence Copilot", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(503, "not ready")
        return {"status": "ready", "llm_backend": request.app.state.llm_backend}

    @app.post("/v1/detect")
    def detect(req: DetectRequest, request: Request):
        rows = pd.DataFrame([r.model_dump() for r in req.rows])
        preds = request.app.state.detector.predict(rows)
        request.app.state.counts["rows_scored"] += len(preds)
        return {"predictions": [p | {"router_id": r.router_id, "timestamp": r.timestamp}
                                for p, r in zip(preds, req.rows)]}

    @app.post("/v1/triage")
    def triage(row: TelemetryRow, request: Request):
        t0 = time.perf_counter()
        state = request.app.state.graph.invoke({"row": row.model_dump()})
        return {
            "outcome": state["outcome"],
            "severity": state.get("severity"),
            "anomaly_score": state["alert"]["anomaly_score"],
            "predicted_type": state["alert"]["predicted_type"],
            "change_id": state.get("change_id"),
            "attempts": state.get("attempts", 0),
            "validation_errors": state.get("errors", []),
            "report": state.get("report"),
            "llm_backend": state.get("llm_backend"),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @app.get("/v1/incidents")
    def incidents(request: Request, limit: int = 20):
        return list(request.app.state.incidents)[:limit]

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request):
        lines = ["# TYPE noc_triage_outcomes_total counter"]
        lines += [f'noc_triage_outcomes_total{{outcome="{k}"}} {v}'
                  for k, v in request.app.state.counts.items() if k != "rows_scored"]
        lines += ["# TYPE noc_rows_scored_total counter",
                  f"noc_rows_scored_total {request.app.state.counts['rows_scored']}"]
        return "\n".join(lines) + "\n"

    return app


# Real components load in the lifespan hook at server start, not at import.
app = create_app()
