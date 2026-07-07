"""Pre-open news sentiment channel (PLAN_v4 §8) — forward-only.

Captures financial headlines published BEFORE 09:15 IST, scores them with
FinBERT, and aggregates to market / sector / stock sentiment. This is the honest
place for "news moves the market": pre-open, where an overnight tariff headline
or an inflation print sets the day's tone — not live intra-window, where retail
feeds lag the repricing.

Honesty clause (PLAN_v4 §8): historical Indian headlines with trustworthy
pre-09:15 published timestamps are not freely recoverable at scale, so this
channel is NOT backfilled. It accrues forward: features enter training with
present=0 historically and only earn a real value on days the capture ran. Its
acceptance is the paper stage (news-present vs news-absent), never the decision
backtest.

Storage — data/regime/news/<YYYY-MM-DD>.parquet:
  [published_ts, capture_ts, source, url, title, matched_symbols,
   matched_sectors, sent_score]
which fixes the legacy data/news/*.json defect (no per-headline published_ts).

Deps (feedparser, transformers/torch) are imported lazily; absent/offline, the
capture is a no-op and readers return present=0. Nothing here is import-time fatal.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday import ROOT, data_path, load_config, load_universe, now_ist, point_in_time

logger = logging.getLogger(__name__)

# RSS sources (PLAN_v4 §8). Market/macro feeds drive news_sent_market; all feeds
# feed symbol/sector matching.
RSS_FEEDS = {
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "reuters_india": "https://www.reutersagency.com/feed/?best-regions=india&post_type=best",
}

# GDELT DOC 2.0 — the secondary net (PLAN_v4 §8): catches overnight macro/India
# headlines the RSS feeds miss. Article `seendate` is a real UTC capture stamp,
# so the PIT discipline holds. Window: prior session close → today's 09:15 IST.
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = ('(Nifty OR Sensex OR NSE OR RBI OR "Indian stock market" '
               'OR "Indian shares") sourcecountry:IN')
GDELT_MAX_RECORDS = 75
_IST_OFFSET = timedelta(hours=5, minutes=30)

NEWS_COLUMNS = ["published_ts", "capture_ts", "source", "url", "title",
                "matched_symbols", "matched_sectors", "sent_score"]

NEWS_FEATURES = ["news_sent_market", "news_sent_sector", "news_sent_stock", "news_count_stock"]

# only headlines published strictly before this local time count for a given day
PREOPEN_CUTOFF = time(9, 15)


def _news_dir() -> Path:
    # data_path() resolves against the LIVE ROOT (see regime_data._global_dir).
    return data_path("regime_path") / "news"


def _day_file(day: date) -> Path:
    return _news_dir() / f"{day.isoformat()}.parquet"


# ── alias map (symbol/sector matching) ────────────────────────────────

def _alias_map(day: date) -> tuple[dict[str, str], dict[str, str]]:
    """Return (name_token → symbol, symbol → sector) for matching headlines.

    Built from the PIT universe plus an optional data/reference/symbol_aliases.csv
    (symbol,alias) for common colloquial names (e.g. 'HDFC Bank' → HDFCBANK)."""
    uni = load_universe(as_of=day)
    tok2sym: dict[str, str] = {}
    sym2sector: dict[str, str] = {}
    for _, r in uni.iterrows():
        sym = str(r["symbol"])
        sym2sector[sym] = str(r.get("sector", "UNKNOWN"))
        tok2sym[sym.lower()] = sym
        name = str(r.get("name", "")).lower()
        if name:
            tok2sym[name] = sym
            # first significant word (e.g. "reliance" from "Reliance Industries")
            first = name.split()[0]
            if len(first) >= 4:
                tok2sym.setdefault(first, sym)
    ap = ROOT / load_config()["data"]["reference_path"] / "symbol_aliases.csv"
    if ap.exists():
        adf = pd.read_csv(ap, comment="#")
        for _, r in adf.iterrows():
            tok2sym[str(r["alias"]).lower()] = str(r["symbol"])
    return tok2sym, sym2sector


# ── scoring ────────────────────────────────────────────────────────────

_SCORER = {"pipe": None, "tried": False}


def _finbert():
    """Lazy FinBERT sentiment pipeline. Cached; returns None if transformers/torch
    are unavailable so capture degrades to a no-op (PLAN_v4 §8)."""
    if _SCORER["tried"]:
        return _SCORER["pipe"]
    _SCORER["tried"] = True
    try:
        from transformers import pipeline  # type: ignore
        _SCORER["pipe"] = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                                   truncation=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("FinBERT unavailable (%s) — news scoring disabled", e)
        _SCORER["pipe"] = None
    return _SCORER["pipe"]


def score_text(title: str) -> float:
    """Signed sentiment in [-1, 1] (positive − negative). 0.0 when no scorer."""
    pipe = _finbert()
    if pipe is None:
        return 0.0
    try:
        out = pipe(title)[0]
        label = out["label"].lower()
        score = float(out["score"])
        if label.startswith("pos"):
            return score
        if label.startswith("neg"):
            return -score
        return 0.0
    except Exception as e:  # noqa: BLE001
        logger.warning("news scoring failed: %s", e)
        return 0.0


# ── capture ──────────────────────────────────────────────────────────

def _feedparser():
    try:
        import feedparser  # type: ignore
        return feedparser
    except Exception as e:  # noqa: BLE001
        logger.warning("feedparser unavailable (%s) — news capture is a no-op", e)
        return None


def capture(day: date | None = None) -> int:
    """Pull RSS headlines (primary) + GDELT DOC 2.0 (secondary net), keep those
    published before 09:15 IST for `day`, match symbols/sectors, score, and append
    to the day file. Idempotent by url. Run at ~08:00 and ~09:10 IST (PLAN_v4 §8).
    Each source degrades independently to a no-op without its dependency/network."""
    day = day or now_ist().date()
    cap = now_ist().isoformat(timespec="seconds")
    tok2sym, sym2sector = _alias_map(day)
    cutoff = datetime.combine(day, PREOPEN_CUTOFF)
    recs: list[dict] = []

    def _add(pub: datetime, title: str, url: str, source: str) -> None:
        syms, sectors = _match(title, tok2sym, sym2sector)
        recs.append({
            "published_ts": pub.isoformat(timespec="seconds"), "capture_ts": cap,
            "source": source, "url": url, "title": title,
            "matched_symbols": ",".join(sorted(syms)),
            "matched_sectors": ",".join(sorted(sectors)),
            "sent_score": score_text(title),
        })

    fp = _feedparser()
    if fp is not None:
        for source, url in RSS_FEEDS.items():
            try:
                feed = fp.parse(url)
            except Exception as e:  # noqa: BLE001
                logger.warning("news feed %s failed: %s", source, e)
                continue
            for entry in getattr(feed, "entries", []):
                pub = _entry_ts(entry)
                if pub is None or pub >= cutoff or pub.date() != day:
                    continue
                title = str(getattr(entry, "title", "")).strip()
                if title:
                    _add(pub, title, str(getattr(entry, "link", "")), source)

    # GDELT secondary net — overnight window reaches back to the prior close, so
    # late-evening headlines (post-RSS-cutoff macro news) are still captured.
    for pub, title, url in _gdelt_headlines(day):
        if pub < cutoff:
            _add(pub, title, url, "gdelt")
    return _write(day, recs)


def _entry_ts(entry) -> datetime | None:
    import calendar
    for attr in ("published_parsed", "updated_parsed"):
        st = getattr(entry, attr, None)
        if st:
            # feed times are UTC struct_time; convert to IST-naive to match now_ist()
            utc = datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
            return utc.replace(tzinfo=None) + _IST_OFFSET
    return None


def _gdelt_headlines(day: date) -> list[tuple[datetime, str, str]]:
    """Secondary net (PLAN_v4 §8): GDELT DOC 2.0 India-market headlines in the
    window [prior day 15:30 IST → today 09:15 IST]. Returns (published_ist,
    title, url) tuples; empty list on any failure (network/API) — fail-open,
    same contract as the RSS path."""
    try:
        import requests
    except Exception as e:  # noqa: BLE001
        logger.warning("requests unavailable (%s) — GDELT capture skipped", e)
        return []
    fmt = "%Y%m%d%H%M%S"
    start_utc = datetime.combine(day - timedelta(days=1), time(15, 30)) - _IST_OFFSET
    end_utc = datetime.combine(day, PREOPEN_CUTOFF) - _IST_OFFSET
    try:
        r = requests.get(GDELT_DOC_API, timeout=20, params={
            "query": GDELT_QUERY, "mode": "artlist", "format": "json",
            "maxrecords": GDELT_MAX_RECORDS, "sort": "datedesc",
            "startdatetime": start_utc.strftime(fmt),
            "enddatetime": end_utc.strftime(fmt),
        })
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except Exception as e:  # noqa: BLE001 — secondary source, never gating
        logger.warning("GDELT pull failed for %s: %s", day, e)
        return []
    out = []
    for a in articles:
        try:
            pub_utc = datetime.strptime(str(a.get("seendate", "")), "%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
        title = str(a.get("title", "")).strip()
        if title:
            out.append((pub_utc + _IST_OFFSET, title, str(a.get("url", ""))))
    return out


def _match(title: str, tok2sym: dict[str, str], sym2sector: dict[str, str]) -> tuple[set, set]:
    low = title.lower()
    syms = {sym for tok, sym in tok2sym.items() if tok in low}
    sectors = {sym2sector.get(s, "UNKNOWN") for s in syms}
    return syms, sectors


def _write(day: date, recs: list[dict]) -> int:
    if not recs:
        return 0
    _news_dir().mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs, columns=NEWS_COLUMNS)
    path = _day_file(day)
    if path.exists():
        prev = pd.read_parquet(path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["url", "title"], keep="first")
    df.to_parquet(path, index=False)
    logger.info("news capture %s: %d headlines", day, len(df))
    return len(df)


# ── read / feature computation ─────────────────────────────────────────

def news_features(symbol: str, sector: str, day: date,
                  decision_ts: pd.Timestamp | None = None) -> dict:
    """News sentiment features for (symbol, sector, day), each with a _present
    flag. PIT: only headlines published before 09:15 AND captured ≤ decision_ts.

    Historically (no capture ran) every flag is 0 — the forward-only contract."""
    if decision_ts is None:
        decision_ts = pd.Timestamp(datetime.combine(day, time(9, 30)))
    feats = {k: np.nan for k in NEWS_FEATURES}
    for k in NEWS_FEATURES:
        feats[k + "_present"] = 0.0

    path = _day_file(day)
    if not path.exists():
        return feats
    df = pd.read_parquet(path)
    if df.empty:
        return feats
    df = point_in_time(df, day, decision_ts, date_col="__none__")  # capture_ts ≤ decision
    pub = pd.to_datetime(df["published_ts"])
    df = df[pub < pd.Timestamp(datetime.combine(day, PREOPEN_CUTOFF))]
    if df.empty:
        return feats

    feats["news_sent_market"] = float(df["sent_score"].mean())
    feats["news_sent_market_present"] = 1.0

    sec_rows = df[df["matched_sectors"].fillna("").str.split(",").apply(lambda xs: sector in xs)]
    if not sec_rows.empty:
        feats["news_sent_sector"] = float(sec_rows["sent_score"].mean())
        feats["news_sent_sector_present"] = 1.0

    sym_rows = df[df["matched_symbols"].fillna("").str.split(",").apply(lambda xs: symbol in xs)]
    feats["news_count_stock"] = float(len(sym_rows))
    feats["news_count_stock_present"] = 1.0
    if not sym_rows.empty:
        feats["news_sent_stock"] = float(sym_rows["sent_score"].mean())
        feats["news_sent_stock_present"] = 1.0
    return feats
