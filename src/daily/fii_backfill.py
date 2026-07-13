"""FII/DII history backfill — close the known gap (data/bhavcopy/fii_dii.parquet
currently holds ~1 row, so the fii_5d_z feature is _present=0 for v1).

The NSE live endpoint (reused via intraday.bhavcopy.fetch_fii_dii) returns only a
short recent window; it cannot reconstruct multi-year history. backfill() pulls
whatever that endpoint offers and merges it (append-only, keyed by date) so the
series grows forward, and reports coverage. Full multi-year history needs a manual
source (NSE provisional cash-market archive / a data vendor) dropped into the same
schema [date, fii_net_cr, dii_net_cr, capture_ts]; until then features keep
_present=0 and the model trains on market + F&O alone (never on imputed flow).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.daily import daily_path

logger = logging.getLogger(__name__)


def _path():
    return daily_path("bhavcopy_path") / "fii_dii.parquet"


def coverage() -> dict:
    p = _path()
    if not p.exists():
        return {"exists": False, "rows": 0}
    df = pd.read_parquet(p)
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.Series([], dtype="datetime64[ns]")
    return {"exists": True, "rows": len(df),
            "first": str(dates.min().date()) if len(dates) else None,
            "last": str(dates.max().date()) if len(dates) else None}


def backfill() -> dict:
    """Append the latest fetchable FII/DII rows to the history. Network failures
    surface as a status (not a crash) so a scheduled run degrades gracefully."""
    before = coverage()
    try:
        from src.intraday.bhavcopy import fetch_fii_dii
        fetch_fii_dii()                       # append-only into fii_dii.parquet
        status = "fetched"
    except Exception as e:  # noqa: BLE001
        logger.warning("fii backfill fetch failed: %s", e)
        status = f"fetch_failed: {e}"
    after = coverage()
    enough = after["rows"] >= 25              # ~enough for a 20-day z with 5-day window
    return {"status": status, "before": before, "after": after,
            "feature_usable": enough,
            "note": None if enough else "history still too short → fii_5d_z stays _present=0; "
                                        "load a multi-year source into the same schema"}
