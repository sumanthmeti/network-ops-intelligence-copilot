"""LLM backends behind one tiny interface, so the graph doesn't care which is used.

    BedrockClaude     production: Claude on Amazon Bedrock (Anthropic SDK, Mantle client)
    ScriptedLLM       tests: replays exact responses, so the retry path is deterministic
    OfflineTemplateLLM  demos without LLM access: builds a report from the alert facts.
                      NOT an LLM. Clearly labeled everywhere it is used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    stop_reason: str = "end_turn"   # "refusal" is routed straight to manual review
    backend: str = ""


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict]) -> LLMResult: ...


class BedrockClaude:
    """Claude via Bedrock's Messages-API ("Mantle") endpoint. Credentials come from the
    normal AWS chain (~/.aws/credentials, env vars, or an IAM role in the cluster)."""

    def __init__(self, model: str | None = None, region: str | None = None):
        from anthropic import AnthropicBedrockMantle
        self.model = model or os.environ.get("BEDROCK_MODEL", "anthropic.claude-opus-5")
        self.client = AnthropicBedrockMantle(aws_region=region or os.environ.get("AWS_REGION", "us-east-1"))

    def complete(self, system: str, messages: list[dict]) -> LLMResult:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=messages,
            # A short, well-specified summarisation task: low effort keeps cost and
            # latency down without hurting quality on work like this.
            output_config={"effort": "low"},
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResult(text=text, stop_reason=r.stop_reason, backend=f"bedrock:{self.model}")


class ScriptedLLM:
    """Returns pre-written responses in order and records every prompt it received."""

    def __init__(self, responses: list[str | LLMResult]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict]) -> LLMResult:
        self.calls.append([dict(m) for m in messages])
        r = self.responses.pop(0)
        return r if isinstance(r, LLMResult) else LLMResult(text=r, backend="scripted")


ACTIONS = {
    "session_down": [("Check BGP neighbor state and last reset reason on the router", "noc"),
                     ("Verify the peer's interface and physical link status", "network_engineering")],
    "route_flap_storm": [("Identify the flapping prefixes and the originating peer", "noc"),
                         ("Apply route dampening or shut the unstable peer if flaps continue", "network_engineering")],
    "packet_loss_burst": [("Check interface error counters and drops on the affected links", "noc"),
                          ("Inspect optics / cabling on interfaces with rising CRC errors", "network_engineering")],
    "latency_spike": [("Compare latency against link utilisation to rule out congestion", "noc"),
                      ("Check for suboptimal routing or a recent path change", "network_engineering")],
}


class OfflineTemplateLLM:
    """Stand-in for demos when no LLM is reachable (e.g. AWS Free plan blocks Bedrock).
    Deterministic text built from the alert. With fail_first=True the first attempt
    for each alert is deliberately wrong (severity mismatch), to demo the retry loop."""

    def __init__(self, fail_first: bool = False):
        self.fail_first = fail_first

    def complete(self, system: str, messages: list[dict]) -> LLMResult:
        alert = json.loads(messages[0]["content"].split("ALERT:", 1)[1])
        is_retry = len(messages) > 1
        severity = alert["severity"]
        if self.fail_first and not is_retry:
            severity = "P3" if severity != "P3" else "P2"
        t = alert["predicted_type"]
        report = {
            "title": f"{t.replace('_', ' ').title()} on {alert['role']} router {alert['router_id']} ({alert['site']})",
            "router_id": alert["router_id"],
            "severity": severity,
            "anomaly_type": t,
            "summary": (f"Router {alert['router_id']} ({alert['role']}, {alert['site']}) raised a {t} alert at "
                        f"{alert['timestamp']} with anomaly score {alert['anomaly_score']:.3f}. Latency "
                        f"{alert['latency_ms']:.1f} ms, packet loss {alert['packet_loss_pct']:.2f}%, BGP state "
                        f"{alert['bgp_session_state']}, {alert['route_flap_count']} route flaps in the last sample."),
            "probable_causes": [f"Detector classified the pattern as {t}"],
            "recommended_actions": [{"step": s, "owner": o} for s, o in ACTIONS[t]],
            "customer_impact": "outage" if t == "session_down" else "degraded",
            "confidence": round(min(alert["anomaly_score"], 0.99), 2),
        }
        return LLMResult(text=json.dumps(report), backend="offline-template (not an LLM)")
