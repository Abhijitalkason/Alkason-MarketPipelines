"""Forward news collector (Track B). Pulls current local+global headlines for the
PIT universe + market-wide and appends them with release_ts = first-seen IST.

Source is pluggable: set DAILY_NEWS_SOURCE (e.g. an RSS/news-API adapter). With no
source configured, collect() reports that and writes nothing — it never invents or
backfills news (historical = lookahead). Run daily (or hourly) on a scheduler so
the archive grows; it gates nothing in Milestone 1/2.
"""

from __future__ import annotations

import logging
import os

from src.daily import load_universe, today_ist
from src.daily.sentiment.archive import append_raw

logger = logging.getLogger(__name__)


def _fetch_headlines(symbols: list[str]) -> list[dict]:
    """Adapter seam. Implement against your chosen news source; must return dicts
    with symbol, headline, source, url. release_ts is stamped at append time."""
    source = os.environ.get("DAILY_NEWS_SOURCE", "")
    if not source:
        return []
    # Example wiring point — a concrete adapter (RSS/newsapi/GDELT) plugs in here.
    raise NotImplementedError(
        f"DAILY_NEWS_SOURCE={source!r} set but no adapter implemented — add one in "
        f"sentiment/collector._fetch_headlines"
    )


def collect(day=None) -> dict:
    day = day or today_ist()
    symbols = load_universe(as_of=day)["symbol"].tolist()
    if not os.environ.get("DAILY_NEWS_SOURCE"):
        return {"status": "no_source_configured", "appended": 0,
                "note": "set DAILY_NEWS_SOURCE and implement the adapter; the archive accrues "
                        "FORWARD only — never backfill historical news (lookahead)"}
    headlines = _fetch_headlines(symbols)
    n = append_raw(headlines)
    logger.info("sentiment collect %s: appended %d headlines", day, n)
    return {"status": "ok", "appended": n, "date": str(day)}
