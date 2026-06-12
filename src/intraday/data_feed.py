"""Broker API client — historical 1-min backfill + live quotes (PLAN_v3 Section 6).

Provider: Upstox (free). Requires UPSTOX_ACCESS_TOKEN in environment / .env.
All bars stored as monthly parquet: data/bars/<SYMBOL>/<YYYY-MM>.parquet with
columns [ts, open, high, low, close, volume, capture_ts]. Timestamps are
tz-naive IST. Failures raise — no silent degradation (v1 post-mortem #2/#7).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)

UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"


class DataFeedError(RuntimeError):
    pass


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
        """1-minute candles for [start, end] (inclusive), chunked monthly."""
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
            now = datetime.now()
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


# ── backfill orchestration ────────────────────────────────────────────

def bars_dir(symbol: str) -> Path:
    cfg = load_config()
    d = ROOT / cfg["data"]["bars_path"] / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_monthly(symbol: str, df: pd.DataFrame) -> list[Path]:
    """Persist 1-min bars into monthly parquet files (merge with existing)."""
    written = []
    df = df.copy()
    df["capture_ts"] = df.get("capture_ts", pd.Timestamp.now())
    for period, chunk in df.groupby(df["ts"].dt.to_period("M")):
        path = bars_dir(symbol) / f"{period}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            chunk = (
                pd.concat([old, chunk], ignore_index=True)
                .drop_duplicates(subset="ts", keep="first")  # keep first capture — point-in-time
                .sort_values("ts")
            )
        chunk.to_parquet(path, index=False)
        written.append(path)
    return written


def backfill(symbols: list[str] | None = None, years: int | None = None) -> None:
    """Backfill 1-min history for the universe (PLAN_v3 Step 2 — starts day one)."""
    from src.intraday import load_universe

    cfg = load_config()
    feed = UpstoxFeed()
    symbols = symbols or load_universe()["symbol"].tolist()
    years = years or cfg["data"]["backfill_years"]
    start = date.today() - timedelta(days=int(years * 365.25))
    failures = []
    for sym in symbols:
        try:
            df = feed.historical_1min(sym, start, date.today())
            if df.empty:
                raise DataFeedError("0 candles returned")
            save_monthly(sym, df)
            logger.info("%s: %d 1-min bars (%s → %s)", sym, len(df), df.ts.min(), df.ts.max())
        except Exception as e:  # noqa: BLE001 — collect, then fail loudly at the end
            logger.error("%s backfill FAILED: %s", sym, e)
            failures.append(sym)
    if failures:
        raise DataFeedError(f"Backfill failed for {len(failures)} symbols: {failures}")
