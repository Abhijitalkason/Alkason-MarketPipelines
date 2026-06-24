"""Best-effort fetch of per-symbol quarterly-results dates from NSE corporate
announcements, for the events.csv blackout calendar (PLAN_v3 §11).

Real source only — no fabrication. Reuses bhavcopy._session() (warmed NSE
cookies + browser UA). Filters announcements whose subject mentions a financial
result. Returns (symbol, YYYY-MM-DD) pairs over the backfill span. Symbols that
fail or return nothing are skipped and logged (no blackout for them, explicit).
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.intraday import ROOT, load_config, load_universe, today_ist  # noqa: E402

logger = logging.getLogger(__name__)
NSE_API = "https://www.nseindia.com/api"


def _results_for(s, symbol: str, frm: date, to: date) -> list[str]:
    url = (f"{NSE_API}/corporate-announcements?index=equities&symbol={symbol}"
           f"&from_date={frm.strftime('%d-%m-%Y')}&to_date={to.strftime('%d-%m-%Y')}")
    resp = s.get(url, timeout=30)
    resp.raise_for_status()
    out = []
    for row in resp.json():
        # NSE category for quarterly results is desc='Financial Result Updates'
        if "financial result" in (row.get("desc") or "").lower():
            stamp = row.get("an_dt") or row.get("dt") or ""  # e.g. '25-Apr-2025 20:10:43'
            try:
                out.append(pd.to_datetime(str(stamp), dayfirst=True).date().isoformat())
            except Exception:  # noqa: BLE001
                continue
    return sorted(set(out))


def fetch_all_results() -> list[tuple[str, str]]:
    from src.intraday.bhavcopy import _session

    cfg = load_config()
    years = cfg["data"]["backfill_years"]
    to = today_ist()
    frm = to - timedelta(days=int(years * 365.25))
    syms = load_universe(as_of=to)["symbol"].tolist()
    s = _session()
    pairs, failed = [], []
    # NSE caps each query window; walk in 6-month slices
    for sym in syms:
        got = []
        cur = frm
        while cur < to:
            chunk_end = min(cur + timedelta(days=180), to)
            try:
                got += _results_for(s, sym, cur, chunk_end)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s %s..%s: %s", sym, cur, chunk_end, e)
            cur = chunk_end + timedelta(days=1)
        if got:
            pairs += [(sym, d) for d in sorted(set(got))]
        else:
            failed.append(sym)
    logger.info("results dates: %d pairs for %d symbols; %d symbols returned none (%s)",
                len(pairs), len(syms) - len(failed), len(failed), failed)
    return pairs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for sym, d in fetch_all_results():
        print(f"{d},results,{sym}")
