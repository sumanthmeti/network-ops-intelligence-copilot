"""Change calendar: planned maintenance windows, as a change-management system would export.

The ML detector can't know that an engineer reset a BGP session on purpose at 02:00.
The agent checks this calendar before paging anyone. This is the context step that
turns "anomaly detected" into "incident worth waking someone for".

Built here from the simulator's labeled maintenance events (standing in for a real
ServiceNow / Jira change export), with a small buffer around each window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data/raw/change_calendar.json"


@dataclass(frozen=True)
class Window:
    change_id: str
    router_id: str
    start: pd.Timestamp
    end: pd.Timestamp


class ChangeCalendar:
    def __init__(self, windows: list[Window]):
        self.by_router: dict[str, list[Window]] = {}
        for w in windows:
            self.by_router.setdefault(w.router_id, []).append(w)

    def active_change(self, router_id: str, ts) -> Window | None:
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        return next((w for w in self.by_router.get(router_id, []) if w.start <= ts <= w.end), None)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "ChangeCalendar":
        raw = json.loads(Path(path).read_text())
        return cls([Window(w["change_id"], w["router_id"], pd.Timestamp(w["start"]), pd.Timestamp(w["end"]))
                    for w in raw])

    @staticmethod
    def build_from_telemetry(telemetry: pd.DataFrame, path: str | Path = DEFAULT_PATH,
                             buffer: str = "15min") -> Path:
        m = telemetry[telemetry["benign_event"] == "maintenance"].sort_values(["router_id", "timestamp"])
        # Consecutive maintenance samples on the same router form one change window.
        new_window = (m["router_id"] != m["router_id"].shift()) | \
                     (m["timestamp"].diff() > pd.Timedelta("10min"))
        m = m.assign(win=new_window.cumsum())
        pad = pd.Timedelta(buffer)
        windows = [
            {"change_id": f"CHG{int(i):05d}", "router_id": g["router_id"].iloc[0],
             "start": (g["timestamp"].min() - pad).isoformat(), "end": (g["timestamp"].max() + pad).isoformat()}
            for i, g in m.groupby("win")
        ]
        Path(path).write_text(json.dumps(windows, indent=1))
        return Path(path)
