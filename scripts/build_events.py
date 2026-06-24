"""Build data/reference/events.csv — the blackout calendar (PLAN_v3 §13, §11).

Populates MARKET-WIDE macro events (RBI MPC, US FOMC, Union Budget) from the
verified date lists below, then OPTIONALLY appends per-symbol earnings (results)
dates fetched from a live source. Re-runnable; rewrites the file from scratch.

Sources (verified):
  - FOMC decision dates: federalreserve.gov/monetarypolicy/fomccalendars.htm
  - Union Budget: Feb 1 each year (Govt of India)
  - RBI MPC announcement (3rd-day) dates: RBI press releases (bi-monthly)

Per-symbol results dates are NOT fabricated. Run with --earnings to fetch them
from NSE; without it, only the (authoritative) market-wide calendar is written.
A symbol with no results row simply gets no earnings blackout — same as today,
but now explicit and extendable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intraday import ROOT  # noqa: E402

# ── verified market-wide macro events (2023-06 … 2026-06 backtest span) ──
FOMC = [  # FOMC rate-decision day (2nd day of meeting)
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]
RBI = [  # RBI MPC resolution-announcement day (3rd day of meeting)
    "2023-08-10", "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-01", "2025-12-05",
    "2026-02-06", "2026-04-08", "2026-06-06",
]
BUDGET = ["2024-02-01", "2025-02-01", "2026-02-01"]


def build(with_earnings: bool = False) -> Path:
    rows = ["# Event blackout calendar — consumed by src/intraday/risk.py (parsed with comment='#').",
            "# type ∈ {expiry, results, rbi, fed, budget}. symbol empty = market-wide.",
            "# Weekly index expiry (Thursdays) is auto-flagged by risk.py from config.",
            "# Market-wide macro dates verified from FOMC calendar / RBI press releases / Union Budget.",
            "# Per-symbol results dates: run `python scripts/build_events.py --earnings` to populate.",
            "date,type,symbol"]
    for d in BUDGET:
        rows.append(f"{d},budget,")
    for d in RBI:
        rows.append(f"{d},rbi,")
    for d in FOMC:
        rows.append(f"{d},fed,")

    if with_earnings:
        from scripts.fetch_results_dates import fetch_all_results  # local, best-effort
        for sym, d in fetch_all_results():
            rows.append(f"{d},results,{sym}")

    out = ROOT / "data" / "reference" / "events.csv"
    out.write_text("\n".join(rows) + "\n")
    n = len(rows) - 6
    print(f"wrote {out}  ({n} events: {len(BUDGET)} budget, {len(RBI)} rbi, {len(FOMC)} fed"
          + (", + earnings" if with_earnings else "") + ")")
    return out


if __name__ == "__main__":
    build(with_earnings="--earnings" in sys.argv)
