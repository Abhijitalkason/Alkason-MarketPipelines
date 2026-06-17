"""Hash-chained append-only decision journal (v3_supplementry §12.1, H-11).

One JSONL file per day: reports/v3/journal/<date>.jsonl. Each line carries the
sha256 of the previous line, so tampering or truncation is detectable. This is
the SEBI-audit substrate and the train/serve-skew evidence base.

Event types: screen, gate_eval, fill, exit, aci_update, train, promote,
rollback, halt, drift, reconcile, explain, recalibrate, session_close.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

from src.intraday import ROOT, load_config, now_ist

logger = logging.getLogger(__name__)

EVENT_TYPES = {
    "screen", "gate_eval", "fill", "exit", "aci_update", "train", "promote",
    "rollback", "halt", "drift", "reconcile", "explain", "recalibrate",
    "session_close",
}


class JournalError(RuntimeError):
    pass


def _journal_dir() -> Path:
    p = Path(load_config()["paths"]["journal"])
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(day: date) -> Path:
    return _journal_dir() / f"{day.isoformat()}.jsonl"


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def _last_line(p: Path) -> str | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    lines = p.read_text().splitlines()
    return lines[-1] if lines else None


def journal_write(event_type: str, payload: dict, day: date | None = None) -> dict:
    """Append one chained event line. Returns the record written."""
    if event_type not in EVENT_TYPES:
        raise JournalError(f"unknown journal event type: {event_type}")
    day = day or now_ist().date()
    p = _path(day)
    prev = _last_line(p)
    record = {
        "ts": now_ist().isoformat(timespec="seconds"),
        "type": event_type,
        "payload": payload,
        "prev_sha": _sha(prev) if prev is not None else "genesis",
    }
    line = json.dumps(record, default=str, sort_keys=True)
    with open(p, "a") as f:  # append-only by construction
        f.write(line + "\n")
    return record


def seal_day(day: date | None = None) -> dict:
    """Write the daily-close record carrying the line count — seals the chain."""
    day = day or now_ist().date()
    p = _path(day)
    n = len(p.read_text().splitlines()) if p.exists() else 0
    return journal_write("session_close", {"lines_before_seal": n}, day=day)


def journal_verify(day: date) -> bool:
    """Recompute the hash chain for one day's journal. Used by gates.py."""
    p = _path(day)
    if not p.exists():
        return False
    prev: str | None = None
    for i, line in enumerate(p.read_text().splitlines()):
        rec = json.loads(line)
        expected = _sha(prev) if prev is not None else "genesis"
        if rec.get("prev_sha") != expected:
            logger.error("journal %s: chain broken at line %d", day, i + 1)
            return False
        prev = line
    return True


def read_journal(day: date, event_type: str | None = None) -> list[dict]:
    """All events for a day, optionally filtered by type."""
    p = _path(day)
    if not p.exists():
        return []
    out = [json.loads(line) for line in p.read_text().splitlines()]
    if event_type:
        out = [r for r in out if r["type"] == event_type]
    return out


def journal_days() -> list[date]:
    """Dates that have a journal file, ascending."""
    return sorted(date.fromisoformat(f.stem) for f in _journal_dir().glob("*.jsonl"))
