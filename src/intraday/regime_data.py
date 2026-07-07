"""Pre-open global/macro regime data layer (PLAN_v4 §5).

The realistic intraday use of "news": not live intra-window headlines, but the
market-direction context that is already set BEFORE 09:15 — overnight US and
Asian moves, USD/INR, crude, and the India VIX regime. These are the features
that let the model lean with or against the day's tone and let the risk layer
stand down on hostile mornings.

Point-in-time discipline (identical to the flow channel, B-4/H-8):
  - prior-day closes (US/Asia full session, USDINR, crude, NSEI, India VIX) are
    content-dated strictly before `day` → validly backfillable and always PIT-safe;
  - same-morning snapshots (Asia partial session, GIFT-Nifty pre-open gap) are
    forward-only, carry a real `capture_ts`, and are read through point_in_time()
    so a backtest never sees a snapshot stamped after 09:30.

Storage — data/regime/global/<YYYY-MM-DD>.parquet, LONG format:
  [date, instrument, field, value, source, capture_ts]
one row per (content date, instrument, field). Returns are computed on read, so
the store holds only observed levels (never derived columns without their source).

Network/deps: yfinance is imported lazily. With it absent or offline, fetch/
capture log a warning and write nothing; readers then return present=0 for every
feature and the model imputes — exactly the missingness contract the flow channel
already uses. Nothing here is import-time fatal.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday import data_path, now_ist, point_in_time

logger = logging.getLogger(__name__)

# logical instrument → (yfinance ticker, kind). kind drives PIT treatment:
#   "prior_close"  — prior-session close, backfillable, always PIT-safe
#   "morning_snap" — same-day pre-open snapshot, forward-only, capture-stamped
INSTRUMENTS: dict[str, tuple[str, str]] = {
    "spx": ("^GSPC", "prior_close"),
    "ndx": ("^IXIC", "prior_close"),
    "nikkei": ("^N225", "prior_close"),
    "hangseng": ("^HSI", "prior_close"),
    "usdinr": ("USDINR=X", "prior_close"),
    "crude": ("BZ=F", "prior_close"),
    "nsei": ("^NSEI", "prior_close"),
    "india_vix": ("^INDIAVIX", "prior_close"),
}

GLOBAL_COLUMNS = ["date", "instrument", "field", "value", "source", "capture_ts"]


def _global_dir() -> Path:
    # data_path() resolves against the LIVE ROOT (updated by tests' monkeypatch),
    # unlike a module-level ROOT import which binds once at import time.
    return data_path("regime_path") / "global"


def _day_file(day: date) -> Path:
    return _global_dir() / f"{day.isoformat()}.parquet"


# ── fetch / capture ───────────────────────────────────────────────────

def _yf():
    """Lazy yfinance import. Returns None (with a warning) when unavailable so
    callers degrade to present=0 rather than crashing (PLAN_v4 §5 honesty clause)."""
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception as e:  # noqa: BLE001
        logger.warning("yfinance unavailable (%s) — regime global fetch is a no-op", e)
        return None


def _write_day(day: date, records: list[dict]) -> None:
    if not records:
        return
    _global_dir().mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=GLOBAL_COLUMNS)
    path = _day_file(day)
    if path.exists():  # merge, official/newer capture wins per (instrument, field)
        prev = pd.read_parquet(path)
        df = (pd.concat([prev, df], ignore_index=True)
              .sort_values("capture_ts")
              .drop_duplicates(subset=["instrument", "field"], keep="last"))
    df.to_parquet(path, index=False)


def backfill(start: date, end: date) -> int:
    """Backfill prior-close instruments over [start, end] (PLAN_v4 §5).

    Only 'prior_close' instruments are historically valid — their content date is
    strictly before the trading day that would consume them, so no lookahead. The
    close observed for calendar date D is written under date D; readers select
    rows with content date < the decision day.
    """
    yf = _yf()
    if yf is None:
        return 0
    written = 0
    cap = now_ist().isoformat(timespec="seconds")
    for name, (ticker, kind) in INSTRUMENTS.items():
        if kind != "prior_close":
            continue
        try:
            hist = yf.Ticker(ticker).history(start=start.isoformat(),
                                             end=(end + timedelta(days=1)).isoformat())
        except Exception as e:  # noqa: BLE001
            logger.warning("regime backfill %s (%s) failed: %s", name, ticker, e)
            continue
        if hist is None or hist.empty:
            continue
        by_day: dict[date, list[dict]] = {}
        for ts, row in hist.iterrows():
            d = pd.Timestamp(ts).date()
            close = float(row["Close"])
            if not np.isfinite(close):
                continue
            by_day.setdefault(d, []).append(
                {"date": d, "instrument": name, "field": "close",
                 "value": close, "source": f"yfinance:{ticker}", "capture_ts": cap})
        for d, recs in by_day.items():
            _write_day(d, recs)
            written += len(recs)
    logger.info("regime backfill %s→%s: wrote %d rows", start, end, written)
    return written


def capture_today(day: date | None = None) -> int:
    """Capture the pre-open snapshot for `day` (PLAN_v4 §5, run 08:45–09:10 IST).

    Prior-close instruments get a fresh capture; morning-snapshot instruments
    (Asia partial, GIFT gap) accrue forward only. Everything is stamped with a
    real capture_ts so point_in_time() can enforce ≤ 09:30 at replay time."""
    day = day or now_ist().date()
    yf = _yf()
    if yf is None:
        return 0
    cap = now_ist().isoformat(timespec="seconds")
    records: list[dict] = []
    for name, (ticker, kind) in INSTRUMENTS.items():
        try:
            hist = yf.Ticker(ticker).history(period="7d")
        except Exception as e:  # noqa: BLE001
            logger.warning("regime capture %s (%s) failed: %s", name, ticker, e)
            continue
        if hist is None or hist.empty:
            continue
        last_ts = pd.Timestamp(hist.index[-1]).date()
        close = float(hist["Close"].iloc[-1])
        # the observed level's content date is the session it belongs to
        records.append({"date": last_ts, "instrument": name, "field": "close",
                        "value": close, "source": f"yfinance:{ticker}", "capture_ts": cap})
    _capture_gift_gap(day, cap, records)
    # group by content date so each level lands in its own day-file
    written = 0
    for d in sorted({r["date"] for r in records}):
        _write_day(d, [r for r in records if r["date"] == d])
        written += sum(1 for r in records if r["date"] == d)
    logger.info("regime capture %s: wrote %d rows", day, written)
    return written


def _capture_gift_gap(day: date, cap: str, records: list[dict]) -> None:
    """Best-effort GIFT-Nifty pre-open level (PLAN_v4 §5). No reliable free API;
    if it can't be obtained the feature simply stays present=0 forever — which is
    exactly what the missingness policy is built to handle. Stamped same-day so
    point_in_time() treats it as a morning snapshot (capture_ts ≤ 09:30)."""
    # Placeholder for a future NSE-IX scrape. Intentionally a no-op today so the
    # channel is wired end-to-end (feature exists, present flag flips when data
    # arrives) without inventing a fragile source now.
    return


# ── read / feature computation ─────────────────────────────────────────

def _load_window(day: date, lookback_days: int = 15) -> pd.DataFrame:
    """All stored global rows over the trailing window up to and including `day`."""
    frames = []
    for d in pd.date_range(end=day, periods=lookback_days + 5).date:
        p = _day_file(d)
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=GLOBAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _series(df: pd.DataFrame, instrument: str) -> pd.Series:
    sub = df[(df["instrument"] == instrument) & (df["field"] == "close")].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["date"] = pd.to_datetime(sub["date"]).dt.date
    sub = sub.sort_values("date").drop_duplicates("date", keep="last")
    return pd.Series(sub["value"].to_numpy(), index=sub["date"])


def regime_features(day: date, decision_ts: pd.Timestamp | None = None) -> dict:
    """Market-level regime features for `day`, each with a _present flag.

    Prior-close instruments: strictly-prior content dates (always PIT-safe).
    Morning snapshots (GIFT gap): filtered via point_in_time on capture_ts.
    Returns a dict keyed by the feature names in REGIME_MARKET_FEATURES + flags;
    keys are always present so the schema is stable — only the values/flags vary.
    """
    if decision_ts is None:
        decision_ts = pd.Timestamp(datetime.combine(day, time(9, 30)))
    feats = {k: np.nan for k in REGIME_MARKET_FEATURES}
    for k in REGIME_MARKET_FEATURES:
        feats[k + "_present"] = 0.0

    win = _load_window(day)
    if win.empty:
        return feats
    win = point_in_time(win, day, decision_ts, date_col="date")
    if win.empty:
        return feats

    def _ret_1d(instr: str) -> None:
        s = _series(win, instr)
        prior = s[s.index < day]
        if len(prior) >= 2:
            r = prior.iloc[-1] / prior.iloc[-2] - 1.0
            if np.isfinite(r):
                feats[f"{instr}_ret_1d"] = float(r)
                feats[f"{instr}_ret_1d_present"] = 1.0

    _ret_1d("spx"); _ret_1d("ndx"); _ret_1d("usdinr"); _ret_1d("crude")

    # Asia: prior-close return of Nikkei as the representative session move
    s_asia = _series(win, "nikkei")
    prior_asia = s_asia[s_asia.index < day]
    if len(prior_asia) >= 2:
        r = prior_asia.iloc[-1] / prior_asia.iloc[-2] - 1.0
        if np.isfinite(r):
            feats["asia_ret_1d"] = float(r)
            feats["asia_ret_1d_present"] = 1.0

    # India VIX: prior close level + 5-day change (regime, not direction)
    s_vix = _series(win, "india_vix")
    prior_vix = s_vix[s_vix.index < day]
    if len(prior_vix) >= 1:
        feats["india_vix_prev"] = float(prior_vix.iloc[-1])
        feats["india_vix_prev_present"] = 1.0
    if len(prior_vix) >= 6:
        chg = prior_vix.iloc[-1] / prior_vix.iloc[-6] - 1.0
        if np.isfinite(chg):
            feats["india_vix_chg_5d"] = float(chg)
            feats["india_vix_chg_5d_present"] = 1.0

    # GIFT gap: same-morning snapshot vs prior NSEI close (forward-only)
    s_gift = _series(win, "gift")
    s_nsei = _series(win, "nsei")
    prior_nsei = s_nsei[s_nsei.index < day]
    gift_today = s_gift[s_gift.index >= day]
    if not gift_today.empty and len(prior_nsei) >= 1 and prior_nsei.iloc[-1] > 0:
        gap = gift_today.iloc[-1] / prior_nsei.iloc[-1] - 1.0
        if np.isfinite(gap):
            feats["gift_gap_pct"] = float(gap)
            feats["gift_gap_pct_present"] = 1.0
    return feats


# Market-level regime features (constant across symbols on a given day). The
# per-symbol/sector news features live in news_regime.py and are merged in
# features.py; kept separate so the backfillable global channel and the
# forward-only news channel can be reasoned about independently (PLAN_v4 §5/§6).
REGIME_MARKET_FEATURES = [
    "spx_ret_1d", "ndx_ret_1d", "asia_ret_1d", "usdinr_ret_1d", "crude_ret_1d",
    "india_vix_prev", "india_vix_chg_5d", "gift_gap_pct",
]
