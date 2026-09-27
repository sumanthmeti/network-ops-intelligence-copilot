"""Every branch of the triage graph, with a fake detector and a scripted LLM, so the
retry / fallback behaviour is tested exactly, with no network and no model."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.change_calendar import ChangeCalendar, Window  # noqa: E402
from agent.graph import build_graph, severity_for  # noqa: E402
from agent.llm import LLMResult, OfflineTemplateLLM, ScriptedLLM  # noqa: E402
from agent.schemas import IncidentReport  # noqa: E402

TS = "2026-09-06T14:00:00+00:00"


class FakeDetector:
    def __init__(self, alert=True, ptype="session_down", score=0.99):
        self.out = {"anomaly_score": score, "alert": alert, "predicted_type": ptype if alert else "normal"}

    def predict(self, rows):
        return [self.out] * len(rows)


def row(router="R0001", role="core", ts=TS):
    return {"router_id": router, "site": "BLR", "role": role, "timestamp": ts, "latency_ms": 5.0,
            "packet_loss_pct": 0.01, "traffic_mbps": 50.0, "route_flap_count": 1, "bgp_session_state": "Idle"}


def report(**over):
    r = {"title": "BGP session down on core router R0001", "router_id": "R0001", "severity": "P1",
         "anomaly_type": "session_down", "summary": "The BGP session on R0001 dropped and traffic collapsed to near zero.",
         "probable_causes": ["peer reset"], "recommended_actions": [{"step": "Check BGP neighbor state", "owner": "noc"}],
         "customer_impact": "outage", "confidence": 0.8}
    return json.dumps(r | over)


EMPTY_CAL = ChangeCalendar([])


def run(detector, llm, r=None, calendar=EMPTY_CAL):
    sunk = []
    g = build_graph(detector, llm, calendar, sink=lambda o, s: sunk.append(o))
    state = g.invoke({"row": r or row()})
    return state, sunk


def test_no_alert_ends_without_calling_llm():
    llm = ScriptedLLM([])
    state, _ = run(FakeDetector(alert=False), llm)
    assert state["outcome"] == "no_incident" and llm.calls == []


def test_p3_is_logged_without_llm():
    llm = ScriptedLLM([])
    state, _ = run(FakeDetector(ptype="latency_spike"), llm, row(role="edge"))
    assert state["severity"] == "P3" and state["outcome"] == "logged" and llm.calls == []


def test_planned_maintenance_suppresses_session_down():
    cal = ChangeCalendar([Window("CHG00001", "R0001", pd.Timestamp("2026-09-06T13:50:00Z"), pd.Timestamp("2026-09-06T14:10:00Z"))])
    state, _ = run(FakeDetector(), ScriptedLLM([]), calendar=cal)
    assert state["outcome"] == "suppressed" and state["change_id"] == "CHG00001"


def test_maintenance_does_not_suppress_packet_loss():
    cal = ChangeCalendar([Window("CHG00001", "R0001", pd.Timestamp("2026-09-06T13:50:00Z"), pd.Timestamp("2026-09-06T14:10:00Z"))])
    llm = ScriptedLLM([report(anomaly_type="packet_loss_burst")])
    state, _ = run(FakeDetector(ptype="packet_loss_burst"), llm, calendar=cal)
    assert state["outcome"] == "ticketed"


def test_valid_first_attempt_is_ticketed():
    state, sunk = run(FakeDetector(), ScriptedLLM([report()]))
    assert state["outcome"] == "ticketed" and state["attempts"] == 1 and sunk == ["ticketed"]
    assert state["report"]["router_id"] == "R0001"


def test_invalid_then_valid_retries_with_error_feedback():
    llm = ScriptedLLM([report(severity="P3"), report()])
    state, _ = run(FakeDetector(), llm)
    assert state["outcome"] == "ticketed" and state["attempts"] == 2
    feedback = llm.calls[1][-1]["content"]
    assert "severity must be 'P1'" in feedback          # the exact error went back to the model
    assert llm.calls[1][1]["role"] == "assistant"       # and so did its own bad answer


def test_not_json_counts_as_a_failed_attempt():
    state, _ = run(FakeDetector(), ScriptedLLM(["Sure! Here is the report: ...", report()]))
    assert state["outcome"] == "ticketed" and state["attempts"] == 2


def test_markdown_fenced_json_is_accepted():
    state, _ = run(FakeDetector(), ScriptedLLM(["```json\n" + report() + "\n```"]))
    assert state["outcome"] == "ticketed" and state["attempts"] == 1


def test_gives_up_after_max_attempts_to_manual_review():
    llm = ScriptedLLM([report(router_id="R9999")] * 3)
    state, sunk = run(FakeDetector(), llm)
    assert state["outcome"] == "manual_review" and state["attempts"] == 3 and len(state["errors"]) == 3
    assert sunk == ["manual_review"]


def test_refusal_goes_straight_to_manual_review():
    state, _ = run(FakeDetector(), ScriptedLLM([LLMResult(text="", stop_reason="refusal")]))
    assert state["outcome"] == "manual_review" and state["attempts"] == 1


def test_offline_template_with_fail_first_recovers_on_retry():
    state, _ = run(FakeDetector(), OfflineTemplateLLM(fail_first=True))
    assert state["outcome"] == "ticketed" and state["attempts"] == 2


@pytest.mark.parametrize("role,atype,expected", [
    ("core", "session_down", "P1"), ("core", "latency_spike", "P2"), ("aggregation", "route_flap_storm", "P1"),
    ("aggregation", "packet_loss_burst", "P2"), ("edge", "session_down", "P2"), ("edge", "latency_spike", "P3"),
])
def test_severity_matrix(role, atype, expected):
    assert severity_for(role, atype) == expected


def test_schema_rejects_p1_without_customer_impact():
    alert = {"router_id": "R0001", "severity": "P1", "predicted_type": "session_down"}
    with pytest.raises(ValidationError, match="cannot have customer_impact"):
        IncidentReport.model_validate_json(report(customer_impact="none"), context={"alert": alert})
