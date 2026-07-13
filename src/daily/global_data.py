"""Global / overnight / commodity / FX daily series (cross-market transmission).

Each series → data/global/<name>.parquet with columns [date, value, release_ts,
capture_ts]. release_ts is the IST instant the value became PUBLIC — the PIT crux:
a US close dated D is only knowable ~02:30 IST on D+1, an Asian close ~13:00 IST
on D, USD/INR ~17:00 IST on D. Features filter on release_ts <= decision_ts, so a
mis-stamped release_ts is the one way lookahead sneaks in here — hence the explicit
per-series offset table below.

Source: yfinance (free). NSE prices stay from the bhavcopy panel — yfinance's NSE
adjustment bug is intraday-v1's; we use it ONLY for global instruments.

File names MUST match the series keys read in features._global_row.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from src.daily import daily_path, now_ist

logger = logging.getLogger(__name__)

# series_name → (yfinance ticker, release offset HOURS from the value-date midnight IST)
# US session closes become public ~02:30 IST next day → 24+2.5 = 26.5h.
SINGLE_SERIES = {
    "spx": ("^GSPC", 26.5), "ndx": ("^IXIC", 26.5), "dxy": ("DX-Y.NYB", 26.5),
    "us10y": ("^TNX", 26.5), "crude": ("CL=F", 26.5), "gold": ("GC=F", 26.5),
    "india_vix": ("^INDIAVIX", 16.0), "usdinr": ("INR=X", 17.0),
}
# Asian indices close in IST early afternoon (~13:00 same day) → composite 'asia'.
ASIA_TICKERS = {"nikkei": "^N225", "hangseng": "^HSI", "shanghai": "000001.SS", "kospi": "^KS11"}
ASIA_RELEASE_HOURS = 13.0


class GlobalDataError(RuntimeError):
    pass


def _yf():
    try:
        import yfinance as yf
    except ImportError as e:  # noqa: F841
        raise GlobalDataError(
            "yfinance not installed — `pip install yfinance` (it is in requirements for the "
            "daily system's global block; the on-disk market/F&O blocks work without it)"
        )
    return yf


def _release_ts(d: pd.Timestamp, hours: float) -> pd.Timestamp:
    return pd.Timestamp(d.date()) + timedelta(hours=hours)


def _write_series(name: str, df: pd.DataFrame) -> int:
    out_dir = daily_path("global_path")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    if path.exists():                         # append-only, keyed by date
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return len(df)


def fetch_series(name: str, ticker: str, release_hours: float,
                 start: date, end: date) -> int:
    yf = _yf()
    hist = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                       progress=False, auto_adjust=True)
    if hist is None or hist.empty:
        raise GlobalDataError(f"{name}: yfinance returned no data for {ticker}")
    close = hist["Close"]
    if hasattr(close, "columns"):             # multiindex → single column
        close = close.iloc[:, 0]
    idx = pd.to_datetime(close.index)
    df = pd.DataFrame({
        "date": idx, "value": close.to_numpy(dtype=float),
        "release_ts": [_release_ts(t, release_hours) for t in idx],
        "capture_ts": now_ist(),
    }).dropna(subset=["value"])
    n = _write_series(name, df)
    logger.info("global %s (%s): %d rows", name, ticker, len(df))
    return n


def fetch_asia(start: date, end: date) -> int:
    """Equal-weight Asia composite (mean of each index normalized to its first
    available close), released ~13:00 IST on the value date."""
    yf = _yf()
    levels = []
    for nm, tk in ASIA_TICKERS.items():
        try:
            h = yf.download(tk, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                            progress=False, auto_adjust=True)["Close"]
            if hasattr(h, "columns"):
                h = h.iloc[:, 0]
            levels.append((h / h.dropna().iloc[0]).rename(nm))
        except Exception as e:  # noqa: BLE001
            logger.warning("asia %s skipped (%s)", nm, e)
    if not levels:
        raise GlobalDataError("asia composite: no constituent indices fetched")
    comp = pd.concat(levels, axis=1).mean(axis=1).dropna()
    idx = pd.to_datetime(comp.index)
    df = pd.DataFrame({"date": idx, "value": comp.to_numpy(dtype=float),
                       "release_ts": [_release_ts(t, ASIA_RELEASE_HOURS) for t in idx],
                       "capture_ts": now_ist()})
    return _write_series("asia", df)


def fetch_all(start: date | None = None, end: date | None = None) -> dict:
    """Fetch every global series. GIFT Nifty has no reliable free feed → omitted
    (a documented gap; SGX/NSE-IX integration is a follow-up)."""
    end = end or now_ist().date()
    start = start or (end - timedelta(days=365 * 3))
    written = {}
    for name, (ticker, hours) in SINGLE_SERIES.items():
        try:
            written[name] = fetch_series(name, ticker, hours, start, end)
        except GlobalDataError as e:
            logger.warning("%s", e)
            written[name] = f"ERROR: {e}"
    try:
        written["asia"] = fetch_asia(start, end)
    except GlobalDataError as e:
        written["asia"] = f"ERROR: {e}"
    written["gift_nifty"] = "OMITTED: no free feed (follow-up)"
    return written
