"""Run the triage agent over real alerts from the held-out test days.

    python agent/run_batch.py                      # offline template "LLM", 1st attempt deliberately wrong
    python agent/run_batch.py --llm bedrock        # Claude on Bedrock (needs Bedrock access)
    python agent/run_batch.py --limit 200

Writes artifacts/queues/{ticketed,manual_review,suppressed,logged}.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.change_calendar import DEFAULT_PATH, ChangeCalendar  # noqa: E402
from agent.detector import LocalDetector  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.llm import BedrockClaude, OfflineTemplateLLM  # noqa: E402

QUEUES = ROOT / "artifacts" / "queues"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["offline", "bedrock"], default="offline")
    p.add_argument("--limit", type=int, default=150)
    a = p.parse_args()

    if not DEFAULT_PATH.exists():
        ChangeCalendar.build_from_telemetry(pd.read_parquet(ROOT / "data/raw/telemetry.parquet"))
    calendar = ChangeCalendar.load()
    detector = LocalDetector()
    llm = BedrockClaude() if a.llm == "bedrock" else OfflineTemplateLLM(fail_first=True)

    QUEUES.mkdir(parents=True, exist_ok=True)
    for f in QUEUES.glob("*.jsonl"):
        f.unlink()

    def sink(outcome, state):
        with open(QUEUES / f"{outcome}.jsonl", "a") as fh:
            fh.write(json.dumps({k: state.get(k) for k in
                                 ["alert", "severity", "change_id", "attempts", "errors", "report", "llm_backend"]},
                                default=str) + "\n")

    graph = build_graph(detector, llm, calendar, sink=sink)

    # Rows the detector alerts on, plus every maintenance row (to exercise suppression).
    test = pd.read_parquet(ROOT / "artifacts/output/test_predictions.parquet")
    rows = pd.concat([test[test["alert"]].head(a.limit), test[test["benign_event"] == "maintenance"]])
    rows = rows.drop_duplicates(["router_id", "timestamp"])

    outcomes, attempts = Counter(), Counter()
    for _, r in rows.iterrows():
        state = graph.invoke({"row": r.drop(["anomaly_score", "alert", "predicted_type"]).to_dict()})
        outcomes[state["outcome"]] += 1
        if state["outcome"] in ("ticketed", "manual_review"):
            attempts[state["attempts"]] += 1

    print(f"LLM backend: {llm.__class__.__name__}")
    print(f"Processed {len(rows)} rows -> {dict(outcomes)}")
    print(f"LLM attempts per report: {dict(sorted(attempts.items()))}")
    sample = QUEUES / "ticketed.jsonl"
    if sample.exists():
        first = json.loads(sample.read_text().splitlines()[0])
        print("\nExample ticket (errors from the failed first attempt shown too):")
        print(json.dumps({k: first[k] for k in ["severity", "attempts", "errors", "report"]}, indent=2))


if __name__ == "__main__":
    main()
