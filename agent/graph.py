"""LangGraph incident-triage agent.

    detect â”€â”€(no alert)â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶ END  [no_incident]
       â”‚
    triage â”€â”€(planned change)â”€â”€â–¶ suppress â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶ END [suppressed]
       â”‚ â”€â”€â”€(P3)â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶ log_only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶ END [logged]
       â”‚ (P1/P2)
    write_report â—€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       â”‚                       â”‚ invalid & attempts < max: error fed back to the LLM
    validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚ valid                          â”‚ invalid & attempts == max, or LLM refusal
    publish â”€â”€â–¶ END [ticketed]      manual_review â”€â”€â–¶ END [manual_review]

Why a graph and not one big function: every branch is an explicit, testable edge; the
state at each step is inspectable; and the retry loop is a cycle in the graph, not
hidden control flow. The deterministic steps (triage, suppression, severity) are
plain code on purpose: paging decisions must be auditable and repeatable, so the LLM
only writes the human-readable report, and even that is validated before anyone sees it.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent.change_calendar import ChangeCalendar
from agent.llm import LLM
from agent.schemas import IncidentReport

MAX_ATTEMPTS = 3

ROLE_WEIGHT = {"core": 3, "aggregation": 2, "edge": 1}
TYPE_WEIGHT = {"session_down": 3, "route_flap_storm": 3, "packet_loss_burst": 2, "latency_spike": 1}


class TriageState(TypedDict, total=False):
    row: dict                    # one telemetry+feature row (input)
    alert: dict                  # facts the report must be consistent with
    severity: Literal["P1", "P2", "P3"]
    change_id: str | None
    messages: list[dict]         # conversation with the LLM (grows on retries)
    attempts: int
    errors: list[str]            # validation errors, one entry per failed attempt
    report: dict | None
    outcome: str
    llm_backend: str
    last_llm: dict               # raw text + stop_reason of the latest LLM call


def severity_for(role: str, anomaly_type: str) -> str:
    """Blast radius x failure mode. Core routers carry the most traffic; a dead BGP
    session or a flap storm withdraws routes, which is worse than added latency."""
    s = ROLE_WEIGHT.get(role, 1) * TYPE_WEIGHT[anomaly_type]
    return "P1" if s >= 6 else "P2" if s >= 3 else "P3"


SYSTEM_PROMPT = """You are a network operations assistant. You write incident reports for \
on-call engineers from router telemetry alerts.

Return ONLY a JSON object, no prose and no markdown fences, matching this JSON schema:
{schema}

Rules:
- router_id, severity and anomaly_type must be exactly the values given in the alert. \
Severity was assigned by deterministic triage; do not re-grade it.
- Base every statement on the alert facts. If a cause is uncertain, say "possible" and \
lower confidence instead of inventing details.
- recommended_actions must be concrete steps an engineer can run now."""


def build_graph(detector, llm: LLM, calendar: ChangeCalendar, sink=None, max_attempts: int = MAX_ATTEMPTS):
    """sink: optional callable(outcome, state) that persists tickets / review items."""
    system = SYSTEM_PROMPT.format(schema=json.dumps(IncidentReport.model_json_schema()))

    def detect(state: TriageState) -> TriageState:
        row = state["row"]
        pred = detector.predict(pd.DataFrame([row]))[0]
        alert = {
            "router_id": row["router_id"], "site": row["site"], "role": row["role"],
            "timestamp": str(row["timestamp"]), "latency_ms": float(row["latency_ms"]),
            "packet_loss_pct": float(row["packet_loss_pct"]), "traffic_mbps": float(row["traffic_mbps"]),
            "route_flap_count": int(row["route_flap_count"]), "bgp_session_state": row["bgp_session_state"],
            "anomaly_score": pred["anomaly_score"], "predicted_type": pred["predicted_type"],
            "is_alert": pred["alert"],
        }
        return {"alert": alert, "attempts": 0, "errors": [], "report": None}

    def triage(state: TriageState) -> TriageState:
        a = state["alert"]
        change = calendar.active_change(a["router_id"], a["timestamp"])
        sev = severity_for(a["role"], a["predicted_type"])
        return {"severity": sev, "change_id": change.change_id if change else None,
                "alert": a | {"severity": sev}}

    def write_report(state: TriageState) -> TriageState:
        messages = list(state.get("messages") or [])
        if not messages:
            messages.append({"role": "user", "content": "Write the incident report for this alert.\nALERT: "
                                                       + json.dumps(state["alert"])})
        result = llm.complete(system, messages)
        messages.append({"role": "assistant", "content": result.text or "(empty)"})
        return {"messages": messages, "attempts": state["attempts"] + 1, "llm_backend": result.backend,
                "last_llm": {"text": result.text, "stop_reason": result.stop_reason}}

    def validate(state: TriageState) -> TriageState:
        last = state["last_llm"]
        if last["stop_reason"] == "refusal":
            return {"errors": state["errors"] + ["model refused the request"], "outcome": "refused"}
        text = last["text"].strip()
        if text.startswith("```"):  # tolerate markdown fences, nothing else
            text = text.strip("`").removeprefix("json").strip()
        try:
            report = IncidentReport.model_validate_json(text, context={"alert": state["alert"]})
            return {"report": report.model_dump()}
        except ValidationError as e:
            # Feed the exact validation errors back so the next attempt can fix them.
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'report'}: {err['msg']}"
                                 for err in e.errors(include_url=False))
            feedback = (f"Your JSON failed validation: {problems}. "
                        "Return the corrected JSON object only.")
            return {"errors": state["errors"] + [problems],
                    "messages": state["messages"] + [{"role": "user", "content": feedback}]}

    def finish(outcome: str):
        def node(state: TriageState) -> TriageState:
            update: dict[str, Any] = {"outcome": outcome}
            if sink:
                sink(outcome, state | update)
            return update
        return node

    def after_detect(state) -> str:
        return "triage" if state["alert"]["is_alert"] else "no_incident"

    def after_triage(state) -> str:
        # Planned maintenance explains session resets/flaps; it does not explain packet
        # loss or latency, so those still go through.
        if state["change_id"] and state["alert"]["predicted_type"] in ("session_down", "route_flap_storm"):
            return "suppress"
        return "log_only" if state["severity"] == "P3" else "write_report"

    def after_validate(state) -> str:
        if state.get("report"):
            return "publish"
        if state.get("outcome") == "refused" or state["attempts"] >= max_attempts:
            return "manual_review"
        return "write_report"

    g = StateGraph(TriageState)
    g.add_node("detect", detect)
    g.add_node("triage", triage)
    g.add_node("write_report", write_report)
    g.add_node("validate", validate)
    for name, outcome in [("no_incident", "no_incident"), ("suppress", "suppressed"), ("log_only", "logged"),
                          ("publish", "ticketed"), ("manual_review", "manual_review")]:
        g.add_node(name, finish(outcome))
        g.add_edge(name, END)
    g.add_edge(START, "detect")
    g.add_conditional_edges("detect", after_detect, ["triage", "no_incident"])
    g.add_conditional_edges("triage", after_triage, ["suppress", "log_only", "write_report"])
    g.add_edge("write_report", "validate")
    g.add_conditional_edges("validate", after_validate, ["publish", "manual_review", "write_report"])
    return g.compile()
