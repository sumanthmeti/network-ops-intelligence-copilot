"""Pydantic contract for the LLM-written incident report.

Two layers of validation, and the difference matters:
  1. SHAPE: types, enums, lengths, required fields. Constrained decoding (structured
     outputs) can guarantee this layer at the API level.
  2. MEANING: the report must be about the router that alerted, carry the severity
     the deterministic triage assigned, and name the detected anomaly type. No JSON
     schema can express "must equal a value from the request". That needs a validator
     with context, and it is why the retry-with-feedback loop exists even with a
     model that always produces valid JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

Severity = Literal["P1", "P2", "P3"]
AnomalyType = Literal["latency_spike", "packet_loss_burst", "route_flap_storm", "session_down"]


class Action(BaseModel):
    step: str = Field(min_length=10, max_length=300, description="One concrete remediation step")
    owner: Literal["noc", "network_engineering", "vendor"]


class IncidentReport(BaseModel):
    title: str = Field(min_length=10, max_length=120)
    router_id: str = Field(pattern=r"^R\d{4}$")
    severity: Severity
    anomaly_type: AnomalyType
    summary: str = Field(min_length=40, max_length=1000, description="What happened, for the on-call engineer")
    probable_causes: list[str] = Field(min_length=1, max_length=5)
    recommended_actions: list[Action] = Field(min_length=1, max_length=6)
    customer_impact: Literal["none", "degraded", "outage"]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def matches_the_alert(self, info: ValidationInfo) -> "IncidentReport":
        """Semantic checks against the alert passed in via validation context."""
        alert = (info.context or {}).get("alert")
        if alert is None:
            return self
        problems = []
        if self.router_id != alert["router_id"]:
            problems.append(f"router_id must be {alert['router_id']!r} (the router that alerted), got {self.router_id!r}")
        if self.severity != alert["severity"]:
            problems.append(f"severity must be {alert['severity']!r} as assigned by triage, got {self.severity!r}")
        if self.anomaly_type != alert["predicted_type"]:
            problems.append(f"anomaly_type must be {alert['predicted_type']!r} as detected, got {self.anomaly_type!r}")
        if self.severity == "P1" and self.customer_impact == "none":
            problems.append("a P1 incident cannot have customer_impact 'none'")
        if problems:
            raise ValueError("; ".join(problems))
        return self
