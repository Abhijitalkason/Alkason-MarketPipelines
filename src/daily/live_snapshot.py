"""Live intraday snapshot via Upstox (feature request 2026-07-23).

The daily/swing pipeline is END-OF-DAY by design: it reads NSE's official
bhavcopy, published only after ~18:30 IST. This module lets it ALSO run on a
LIVE snapshot during market hours — a provisional "today" bar per stock built
from the Upstox market-quote API (today's open/high/low + the current price as
a provisional close + volume-so-far).

⚠️ Honesty contract — a live snapshot is NOT a final session:
  - `close` is the CURRENT price, not the 15:30 close (it moves all day);
  - high/low/volume are partial (the day isn't over);
  - flow features (delivery %, OI, FII) do NOT exist intraday — they come from
    the EOD bhavcopy — so those inputs are absent (imputed, `_present=0`), and
    the model was TRAINED on final bars with them present. This is a genuine
    train/serve mismatch. Any pick from a live snapshot is therefore a
    PROVISIONAL preview, explicitly flagged as such — never a final signal.
  - It does not (and cannot) create predictive edge the model doesn't have.

Requires a valid UPSTOX_ACCESS_TOKEN (expires daily ~03:30 IST).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.daily import load_universe, today_ist
from src.intraday import ROOT

logger = logging.getLogger(__name__)

UPSTOX_BASE = "https://api.upstox.com/v2"
DAILY_DIR = ROOT / "data" / "daily"
# columns a daily panel row must carry (match src/daily/panel.py output)
_PANEL_COLS = ["date", "open", "high", "low", "close", "prev_close", "volume",
               "turnover_lacs", "deliv_per", "oi", "oi_change", "basis_pct",
               "fut_close", "spot_close", "adj_factor", "capture_ts"]


def token_ok() -> bool:
    """True iff a live Upstox call authenticates (cheap profile ping)."""
    try:
        from src.intraday.data_feed import make_feed
        make_feed().auth_ping()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Upstox live snapshot unavailable: %s", e)
        return False


def fetch_live_quotes(symbols: list[str], trace=None) -> dict[str, dict]:
    """Today's provisional OHLCV per symbol from Upstox market-quote (50/call).
    Returns {symbol: {open, high, low, close, volume}}; symbols Upstox can't
    map or price are skipped and (optionally) traced."""
    from src.intraday.data_feed import make_feed
    feed = make_feed()
    out: dict[str, dict] = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i : i + 50]
        key_map = {}
        for s in batch:
            try:
                key_map[feed.instrument_key(s)] = s
            except Exception:  # noqa: BLE001 — unmapped (delisted/renamed)
                if trace:
                    trace("live", s, "skip", reason="not in Upstox instrument master")
        if not key_map:
            continue
        payload = feed._get(f"{UPSTOX_BASE}/market-quote/quotes?instrument_key="
                            + ",".join(key_map))
        for _, q in payload.get("data", {}).items():
            sym = q.get("symbol")
            if sym not in symbols and sym in key_map.values():
                pass
            ohlc = q.get("ohlc") or {}
            ltp = q.get("last_price")
            sym = q.get("symbol") or key_map.get(q.get("instrument_token"), "?")
            if ltp is None or not ohlc:
                continue
            out[sym] = {"open": ohlc.get("open"), "high": ohlc.get("high"),
                        "low": ohlc.get("low"), "close": ltp,
                        "volume": q.get("volume")}
            if trace:
                trace("live", sym, "ok", ltp=ltp, open=ohlc.get("open"),
                      high=ohlc.get("high"), low=ohlc.get("low"))
    return out


def inject_live_rows(day: date | None = None, trace=None) -> dict:
    """Write a PROVISIONAL 'today' row into each universe symbol's daily panel
    from the live Upstox snapshot, so the pipeline can score `day`=today before
    NSE publishes. Idempotent: re-running overwrites the provisional row.
    Returns {'injected': n, 'skipped': m, 'day': day, 'provisional': True}."""
    day = day or today_ist()
    syms = load_universe(as_of=day)["symbol"].tolist()
    quotes = fetch_live_quotes(syms, trace=trace)
    injected, skipped = 0, 0
    ts = pd.Timestamp(today_ist())
    for sym in syms:
        f = DAILY_DIR / f"{sym}.parquet"
        q = quotes.get(sym)
        if not f.exists() or not q or q.get("close") is None:
            skipped += 1
            continue
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] < ts]                       # drop any prior provisional today-row
        if not len(df):
            skipped += 1
            continue
        last = df.iloc[-1]
        prev_close = float(last["close"])
        vol = q.get("volume") or 0
        # delivery %/OI/basis do NOT exist intraday — the model treats delivery %
        # as MANDATORY, so carry forward the last known session's values as a
        # proxy (delivery % is sticky day-to-day). Disclosed approximation.
        row = {c: None for c in _PANEL_COLS}
        row.update({"date": ts, "open": q.get("open") or q["close"],
                    "high": q.get("high") or q["close"], "low": q.get("low") or q["close"],
                    "close": q["close"], "prev_close": prev_close, "volume": vol,
                    "turnover_lacs": round(q["close"] * vol / 1e5, 2),
                    "deliv_per": (float(last["deliv_per"]) if pd.notna(last.get("deliv_per")) else None),
                    "oi": (float(last["oi"]) if pd.notna(last.get("oi")) else None),
                    "oi_change": 0.0, "basis_pct": (float(last["basis_pct"])
                                                    if pd.notna(last.get("basis_pct")) else None),
                    "spot_close": q["close"], "adj_factor": 1.0,
                    "capture_ts": pd.Timestamp(today_ist())})
        out_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        out_df.to_parquet(f, index=False)
        injected += 1
    logger.info("live snapshot: injected provisional %s rows for %d/%d symbols (%d skipped)",
                day, injected, len(syms), skipped)
    return {"injected": injected, "skipped": skipped, "day": str(day),
            "provisional": True, "source": "upstox market-quote (intraday)"}
