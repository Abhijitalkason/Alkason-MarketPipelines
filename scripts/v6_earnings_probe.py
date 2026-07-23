"""PLAN_v6 §4.6b — earnings-calendar PIT probe (F9).

Gate 3's event blackout needs results dates *as known before the event*.
Two NSE lanes are probed on a 20-symbol sample of the PIT universe,
2023-07 → 2026-06:

  announcements — corporate-announcements filtered to financial results
      (post-event timestamps: when results were FILED — establishes archive
      depth/coverage, but is not by itself PIT);
  board_meetings — corporate-board-meetings intimations (announced days or
      weeks IN ADVANCE — the true PIT source for a blackout).

If PIT coverage < 90% of universe-quarters, the blackout falls back to the
conservative scheduled-month window (plan §4.6b) — decided by the report,
not here.

Run: python -m scripts.v6_earnings_probe
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from datetime import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from src.intraday import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("v6.earnings")

NSE_API = "https://www.nseindia.com/api"
SEED = 42
N_SYMBOLS = 20
FRM, TO = date(2023, 7, 1), date(2026, 6, 30)
OUT = ROOT / "reports" / "v6" / "earnings_probe.json"


def _sample_symbols() -> list[str]:
    uni = pd.read_parquet(ROOT / "data" / "processed" / "v6_universe.parquet")
    rng = random.Random(SEED)
    return sorted(rng.sample(sorted(set(uni["symbol"])), N_SYMBOLS))


def _board_meetings_for(s, symbol: str, frm: date, to: date) -> list[dict]:
    url = (f"{NSE_API}/corporate-board-meetings?index=equities&symbol={symbol}"
           f"&from_date={frm.strftime('%d-%m-%Y')}&to_date={to.strftime('%d-%m-%Y')}")
    resp = s.get(url, timeout=30)
    resp.raise_for_status()
    out = []
    for row in resp.json():
        purpose = (row.get("bm_purpose") or row.get("purpose") or "").lower()
        if "result" not in purpose:
            continue
        meeting = row.get("bm_date") or row.get("meetingdate")
        announced = row.get("bm_timestamp") or row.get("attachmentdt") or row.get("dt")
        try:
            m = pd.to_datetime(str(meeting), dayfirst=True).date()
            a = pd.to_datetime(str(announced), dayfirst=True).date() if announced else None
        except Exception:  # noqa: BLE001
            continue
        out.append({"meeting": m.isoformat(),
                    "announced": a.isoformat() if a else None,
                    "lead_days": (m - a).days if a else None})
    return out


def main() -> None:
    from scripts.fetch_results_dates import _results_for
    from src.intraday.bhavcopy import _session

    s = _session()
    symbols = _sample_symbols()
    logger.info("probing %d symbols: %s", len(symbols), symbols)
    expected_quarters = 12  # 2023Q3..2026Q2

    per_symbol = {}
    for sym in symbols:
        filings, meetings = [], []
        cur = FRM
        while cur < TO:
            end = min(cur + timedelta(days=180), TO)
            try:
                filings += _results_for(s, sym, cur, end)
            except Exception as e:  # noqa: BLE001
                logger.warning("announcements %s %s..%s: %s", sym, cur, end, e)
            try:
                meetings += _board_meetings_for(s, sym, cur, end)
            except Exception as e:  # noqa: BLE001
                logger.warning("board-meetings %s %s..%s: %s", sym, cur, end, e)
            cur = end + timedelta(days=1)
            time.sleep(1)
        q_filed = {pd.Period(d, freq="Q") for d in filings}
        q_meet = {pd.Period(m["meeting"], freq="Q") for m in meetings}
        leads = [m["lead_days"] for m in meetings if m["lead_days"] is not None]
        per_symbol[sym] = {
            "filings": len(filings), "filed_quarters": len(q_filed),
            "meetings": len(meetings), "meeting_quarters": len(q_meet),
            "median_lead_days": float(pd.Series(leads).median()) if leads else None,
        }
        logger.info("%s: %d filings (%d qtrs), %d meetings (%d qtrs)",
                    sym, len(filings), len(q_filed), len(meetings), len(q_meet))

    df = pd.DataFrame(per_symbol).T
    summary = {
        "generated": dt.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "symbols_probed": len(symbols),
        "expected_quarters_per_symbol": expected_quarters,
        "announcement_coverage": round(float(df["filed_quarters"].sum()
                                             / (len(symbols) * expected_quarters)), 4),
        "board_meeting_pit_coverage": round(float(df["meeting_quarters"].sum()
                                                  / (len(symbols) * expected_quarters)), 4),
        "median_lead_days": (float(df["median_lead_days"].dropna().median())
                             if df["median_lead_days"].notna().any() else None),
        "per_symbol": per_symbol,
        "verdict_rule": "board-meeting PIT coverage >= 0.90 => PIT blackout usable; "
                        "else conservative scheduled-month window",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    logger.info("summary: ann=%.0f%% pit=%.0f%% -> %s",
                summary["announcement_coverage"] * 100,
                summary["board_meeting_pit_coverage"] * 100, OUT)


if __name__ == "__main__":
    main()
