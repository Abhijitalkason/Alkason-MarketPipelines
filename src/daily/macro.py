"""Macro / event calendar (the post-event-drift core; PIT-critical).

data/reference/macro_calendar.csv schema:
    event_date,event_type,value,consensus,release_ts
event_date is when the event happens; release_ts is when its outcome became public
(IST). Features (days_to/since_macro, surprise) filter on release_ts <= decision_ts,
so a future event's DATE may be known (scheduled ahead) while its VALUE is not until
release_ts — exactly the asymmetry the post-event-drift hypothesis needs.

There is no single free API for the Indian macro calendar. build_calendar() creates
the file with the canonical header (and any built-in known schedule) so it can be
hand/semi-automatically maintained; it never invents values. Earnings dates live in
the existing events.csv (event_type='results'), read directly by features._macro_row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.daily import daily_path

logger = logging.getLogger(__name__)

CALENDAR_COLUMNS = ["event_date", "event_type", "value", "consensus", "release_ts"]


def _calendar_path() -> Path:
    return daily_path("reference_path") / "macro_calendar.csv"


def build_calendar() -> dict:
    """Ensure macro_calendar.csv exists with the canonical schema. Returns status;
    populating values is a maintained process (RBI/MOSPI/Fed/BLS releases), not an
    automatic scrape — each row carries its real release_ts so PIT holds."""
    path = _calendar_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=CALENDAR_COLUMNS).to_csv(path, index=False)
        logger.info("macro_calendar.csv created (empty, canonical schema) → %s", path)
        return {"created": True, "path": str(path), "rows": 0, "columns": CALENDAR_COLUMNS}
    df = pd.read_csv(path, comment="#")
    missing = [c for c in CALENDAR_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"macro_calendar.csv missing columns {missing} — expected {CALENDAR_COLUMNS}")
    return {"created": False, "path": str(path), "rows": len(df), "columns": list(df.columns)}


def add_event(event_date: str, event_type: str, release_ts: str,
              value: float | None = None, consensus: float | None = None) -> int:
    """Append one PIT macro event. release_ts is mandatory (the public instant)."""
    path = _calendar_path()
    build_calendar()
    df = pd.read_csv(path, comment="#") if path.exists() else pd.DataFrame(columns=CALENDAR_COLUMNS)
    row = {"event_date": event_date, "event_type": event_type, "value": value,
           "consensus": consensus, "release_ts": release_ts}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).drop_duplicates(
        subset=["event_date", "event_type"], keep="last")
    df.to_csv(path, index=False)
    return len(df)
