"""Broker API client — historical 1-min backfill, live quotes, post-close
reconciliation (PLAN_v3 §6; v3_supplementry §3.6).

Provider: Upstox (config data.provider; any other value fails loudly — no
silently-wrong provider). Requires UPSTOX_ACCESS_TOKEN in environment / .env.

Bar storage: monthly parquet data/bars/<SYMBOL>/<YYYY-MM>.parquet with columns
[ts, open, high, low, close, volume, capture_ts, provisional]. Timestamps are
tz-naive IST. Official candles always replace provisional (poll-built) rows
for the same ts; among official rows the first capture wins (point-in-time).

Failures raise — no silent degradation (v1 post-mortem #2/#7).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.intraday import ROOT, load_config, now_ist, today_ist

logger = logging.getLogger(__name__)

UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

RECONCILE_ALERT_BPS = 5.0  # mean |provisional − official| close above this → alert file


class DataFeedError(RuntimeError):
    pass


def make_feed(**kwargs) -> "UpstoxFeed":
    """Provider factory — consumes config data.provider (no hardcoded client)."""
    provider = load_config()["data"]["provider"]
    if provider == "upstox":
        return UpstoxFeed(**kwargs)
    raise DataFeedError(f"data.provider={provider!r} is not implemented (only 'upstox')")


class UpstoxFeed:
    """Historical + live market data via Upstox API v2."""

    def __init__(self, request_pause: float = 0.30):
        token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
        if not token:
            raise DataFeedError(
                "UPSTOX_ACCESS_TOKEN not set. Create an app at developer.upstox.com, "
                "complete the OAuth login once, and put the access token in .env"
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})
        self.request_pause = request_pause
        self.cfg = load_config()
        self._instrument_map: dict[str, str] | None = None

    # ── auth ──────────────────────────────────────────────────────────

    def auth_ping(self) -> None:
        """Cheap authenticated request. Run at 08:55 so a stale token fails
        BEFORE market open, not at the first poll (G.5). See RUNBOOK.md for
        the daily token-refresh procedure."""
        resp = self.session.get(f"{UPSTOX_BASE}/user/profile", timeout=15)
        if resp.status_code != 200:
            raise DataFeedError(
                f"Upstox auth ping failed (HTTP {resp.status_code}) — refresh "
                f"UPSTOX_ACCESS_TOKEN before the session (see RUNBOOK.md)"
            )

    # ── instrument master ─────────────────────────────────────────────

    def _load_instruments(self) -> dict[str, str]:
        """Map NSE equity trading symbol → Upstox instrument_key. Cached on disk."""
        if self._instrument_map is not None:
            return self._instrument_map
        cache = ROOT / self.cfg["data"]["reference_path"] / "upstox_instruments.json"
        if cache.exists() and cache.stat().st_mtime > time.time() - 7 * 86400:
            self._instrument_map = json.loads(cache.read_text())
            return self._instrument_map

        logger.info("Downloading Upstox NSE instrument master")
        resp = requests.get(INSTRUMENTS_URL, timeout=60)
        resp.raise_for_status()
        rows = json.loads(gzip.decompress(resp.content))
        mapping = {
            r["trading_symbol"]: r["instrument_key"]
            for r in rows
            if r.get("segment") == "NSE_EQ" and r.get("instrument_type") == "EQ"
        }
        if not mapping:
            raise DataFeedError("Instrument master parse produced 0 NSE_EQ symbols")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(mapping))
        self._instrument_map = mapping
        return mapping

    def instrument_key(self, symbol: str) -> str:
        m = self._load_instruments()
        if symbol not in m:
            raise DataFeedError(f"{symbol}: not found in Upstox NSE_EQ instrument master")
        return m[symbol]

    # ── historical candles ────────────────────────────────────────────

    def _get(self, url: str) -> dict:
        for attempt in range(4):
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 429:  # rate limited — back off and retry
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") != "success":
                raise DataFeedError(f"Upstox error for {url}: {payload}")
            time.sleep(self.request_pause)
            return payload
        raise DataFeedError(f"Rate-limited 4x on {url}")

    def historical_1min(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Official 1-minute candles for [start, end] (inclusive), chunked monthly."""
        key = self.instrument_key(symbol)
        frames = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=28), end)
            url = f"{UPSTOX_BASE}/historical-candle/{key}/1minute/{chunk_end}/{chunk_start}"
            payload = self._get(url)
            candles = payload.get("data", {}).get("candles", [])
            if candles:
                df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
                frames.append(df.drop(columns=["oi"]))
            chunk_start = chunk_end + timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        out = pd.concat(frames, ignore_index=True)
        out["ts"] = pd.to_datetime(out["ts"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        out = out.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        return out

    # ── live quotes ───────────────────────────────────────────────────

    def quotes(self, symbols: list[str]) -> pd.DataFrame:
        """Full quotes (LTP, volume, 5-level depth) for up to 50 symbols per call."""
        rows = []
        for i in range(0, len(symbols), 50):
            batch = symbols[i : i + 50]
            keys = ",".join(self.instrument_key(s) for s in batch)
            payload = self._get(f"{UPSTOX_BASE}/market-quote/quotes?instrument_key={keys}")
            now = now_ist()
            for name, q in payload.get("data", {}).items():
                depth = q.get("depth", {})
                rows.append({
                    "symbol": q.get("symbol", name),
                    "ltp": q.get("last_price"),
                    "volume": q.get("volume"),
                    "bid": (depth.get("buy") or [{}])[0].get("price"),
                    "ask": (depth.get("sell") or [{}])[0].get("price"),
                    "depth_buy": json.dumps(depth.get("buy", [])),
                    "depth_sell": json.dumps(depth.get("sell", [])),
                    "capture_ts": now,
                })
        return pd.DataFrame(rows)


# ── bar persistence ───────────────────────────────────────────────────

def bars_dir(symbol: str) -> Path:
    base = Path(load_config()["data"]["bars_path"])
    base = base if base.is_absolute() else ROOT / base
    d = base / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_monthly(symbol: str, df: pd.DataFrame, provisional: bool = False) -> list[Path]:
    """Persist 1-min bars into monthly parquet files.

    Merge rules (v3_supplementry §3.6):
      - official rows (provisional=False) always replace provisional rows for
        the same ts;
      - among official rows, the FIRST capture wins (point-in-time);
      - provisional rows never overwrite anything.
    """
    written = []
    df = df.copy()
    if "capture_ts" not in df.columns:
        df["capture_ts"] = now_ist()
    df["provisional"] = provisional
    for period, chunk in df.groupby(df["ts"].dt.to_period("M")):
        path = bars_dir(symbol) / f"{period}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            if "provisional" not in old.columns:
                old["provisional"] = False
            merged = pd.concat([old, chunk], ignore_index=True)
            # official first within each ts group, earliest capture first among equals
            merged = merged.sort_values(["ts", "provisional", "capture_ts"])
            chunk = merged.drop_duplicates(subset="ts", keep="first").sort_values("ts")
        chunk.to_parquet(path, index=False)
        written.append(path)
    return written


# ── backfill orchestration ────────────────────────────────────────────

def backfill(symbols: list[str] | None = None, years: int | None = None) -> None:
    """Backfill official 1-min history for the PIT universe (PLAN_v3 Step 2).

    After each symbol's save, cross-checks the 1-min aggregate against the
    official bhavcopy close on 5 random session dates (B-9 — first caller of
    the cross-check control). Any failure joins the loud final raise.
    """
    from src.intraday import load_universe
    from src.intraday.bars import cross_check_bhavcopy, load_1min

    cfg = load_config()
    feed = make_feed()
    symbols = symbols or load_universe(as_of=today_ist())["symbol"].tolist()
    years = years or cfg["data"]["backfill_years"]
    start = today_ist() - timedelta(days=int(years * 365.25))
    failures = []
    for sym in symbols:
        try:
            df = feed.historical_1min(sym, start, today_ist())
            if df.empty:
                raise DataFeedError("0 candles returned")
            save_monthly(sym, df, provisional=False)
            logger.info("%s: %d 1-min bars (%s → %s)", sym, len(df), df.ts.min(), df.ts.max())
            _spot_check(sym, df, cross_check_bhavcopy, load_1min)
        except Exception as e:  # noqa: BLE001 — collect, then fail loudly at the end
            logger.error("%s backfill FAILED: %s", sym, e)
            failures.append(f"{sym}: {e}")
    if failures:
        raise DataFeedError(f"Backfill failed for {len(failures)} symbols: {failures}")


def _spot_check(symbol: str, df: pd.DataFrame, cross_check, load_1min) -> None:
    """Cross-check 5 random downloaded session dates against bhavcopy.
    Days without a bhavcopy file on disk are skipped (the bhavcopy backfill
    has its own raise-on-miss); days WITH a file must match."""
    days = sorted(set(df["ts"].dt.date))
    if not days:
        return
    rng = random.Random(symbol)  # deterministic per symbol
    sample = rng.sample(days, min(5, len(days)))
    bdir = Path(load_config()["data"]["bhavcopy_path"])
    bdir = bdir if bdir.is_absolute() else ROOT / bdir
    checked = 0
    for day in sample:
        if not (bdir / f"eq_{day.isoformat()}.parquet").exists():
            continue
        stored = load_1min(symbol, day, day, adjusted=False)
        cross_check(symbol, day, stored)  # raises BarValidationError on mismatch
        checked += 1
    logger.info("%s: bhavcopy cross-check passed on %d/%d sampled days", symbol, checked, len(sample))


# ── post-close reconciliation (H-3) ───────────────────────────────────

def reconcile_day(symbol: str, day: date, feed: UpstoxFeed | None = None) -> float:
    """Replace the day's poll-built provisional bars with official candles,
    then bhavcopy cross-check. Returns mean |provisional − official| close in
    bps (the live skew detector, PLAN_v3 §14). Called by the recorder
    post-close for every recorded symbol."""
    from src.intraday.bars import cross_check_bhavcopy, load_1min
    from src.intraday.journal import journal_write

    feed = feed or make_feed()
    official = feed.historical_1min(symbol, day, day)
    if official.empty:
        raise DataFeedError(f"{symbol} {day}: no official candles available for reconcile")

    # measure provisional skew before overwriting
    skew_bps = float("nan")
    try:
        prov = load_1min(symbol, day, day, adjusted=False, include_provisional=True)
        prov = prov[prov["provisional"].astype(bool)]
        if not prov.empty:
            joined = prov[["close"]].join(
                official.set_index("ts")[["close"]], rsuffix="_official", how="inner"
            ).dropna()
            if not joined.empty:
                skew_bps = float(
                    ((joined["close"] - joined["close_official"]).abs()
                     / joined["close_official"]).mean() * 1e4
                )
    except Exception as e:  # noqa: BLE001 — skew measurement is diagnostic, not gating
        logger.warning("%s %s: provisional skew not measurable (%s)", symbol, day, e)

    save_monthly(symbol, official, provisional=False)  # official replaces provisional
    stored = load_1min(symbol, day, day, adjusted=False)
    bdir = Path(load_config()["data"]["bhavcopy_path"])
    bdir = bdir if bdir.is_absolute() else ROOT / bdir
    if (bdir / f"eq_{day.isoformat()}.parquet").exists():
        cross_check_bhavcopy(symbol, day, stored)

    journal_write("reconcile", {"symbol": symbol, "day": day.isoformat(),
                                "provisional_skew_bps": skew_bps}, day=day)
    if skew_bps == skew_bps and skew_bps > RECONCILE_ALERT_BPS:  # NaN-safe compare
        alerts = Path(load_config()["paths"]["alerts"])
        alerts = alerts if alerts.is_absolute() else ROOT / alerts
        alerts.mkdir(parents=True, exist_ok=True)
        alert_file = alerts / f"{day.isoformat()}_reconcile.json"
        existing = json.loads(alert_file.read_text()) if alert_file.exists() else []
        existing.append({"symbol": symbol, "skew_bps": skew_bps, "ts": now_ist().isoformat()})
        alert_file.write_text(json.dumps(existing, indent=2))
        logger.warning("%s %s: provisional skew %.1f bps > %.1f — alert written",
                       symbol, day, skew_bps, RECONCILE_ALERT_BPS)
    return skew_bps
