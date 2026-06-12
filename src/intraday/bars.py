"""Bar store — load 1-min parquet, resample to 5/15-min, ATR(2h), validation
(PLAN_v3 Section 6.2).

Derived bars are always computed on read from the 1-min source, never stored.
Validation failures RAISE (no drop-and-continue — v1 post-mortem #7).
"""

from __future__ import annotations

import logging
from datetime import date, time

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)


class BarValidationError(ValueError):
    pass


def _bars_path(symbol: str):
    return ROOT / load_config()["data"]["bars_path"] / symbol


def load_1min(symbol: str, start: date, end: date) -> pd.DataFrame:
    """1-min bars for [start, end], session-filtered, ts-indexed, validated."""
    cfg = load_config()
    months = pd.period_range(start=start, end=end, freq="M")
    frames = []
    for m in months:
        p = _bars_path(symbol) / f"{m}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise BarValidationError(f"{symbol}: no 1-min bars on disk for {start}..{end}")
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[(df.ts.dt.date >= start) & (df.ts.dt.date <= end)]
    open_t = time.fromisoformat(cfg["data"]["session_open"])
    close_t = time.fromisoformat(cfg["data"]["session_close"])
    df = df[(df.ts.dt.time >= open_t) & (df.ts.dt.time < close_t)]
    df = df.drop_duplicates(subset="ts").sort_values("ts").set_index("ts")
    validate_1min(symbol, df)
    return df


def validate_1min(symbol: str, df: pd.DataFrame) -> None:
    """Hard checks per PLAN_v3 Section 6.2 — raise on any failure."""
    if df.empty:
        raise BarValidationError(f"{symbol}: empty bar frame")
    if not df.index.is_monotonic_increasing:
        raise BarValidationError(f"{symbol}: timestamps not monotonic")
    if df.index.duplicated().any():
        raise BarValidationError(f"{symbol}: duplicate timestamps")
    px = df[["open", "high", "low", "close"]]
    if (px <= 0).any().any():
        raise BarValidationError(f"{symbol}: non-positive prices")
    if (df["volume"] < 0).any():
        raise BarValidationError(f"{symbol}: negative volume")
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        raise BarValidationError(f"{symbol}: {bad_hl} bars with high < low")
    # bar-count sanity: a full session has ~375 1-min bars; alert under 300
    per_day = df.groupby(df.index.date).size()
    thin = per_day[per_day < 300]
    if len(thin) > len(per_day) * 0.05:
        raise BarValidationError(
            f"{symbol}: {len(thin)}/{len(per_day)} sessions have <300 bars (feed gaps)"
        )


def cross_check_bhavcopy(symbol: str, day: date, df_1min: pd.DataFrame) -> None:
    """Gate-1 check: 1-min aggregate close vs official bhavcopy close within 0.1%."""
    p = ROOT / load_config()["data"]["bhavcopy_path"] / f"eq_{day.isoformat()}.parquet"
    if not p.exists():
        raise BarValidationError(f"bhavcopy file missing for cross-check: {p.name}")
    bh = pd.read_parquet(p)
    row = bh[bh.symbol == symbol]
    if row.empty:
        raise BarValidationError(f"{symbol}: not in bhavcopy {day}")
    day_bars = df_1min[df_1min.index.date == day]
    if day_bars.empty:
        raise BarValidationError(f"{symbol}: no 1-min bars on {day}")
    mismatch = abs(day_bars["close"].iloc[-1] - float(row["close_price"].iloc[0])) / float(
        row["close_price"].iloc[0]
    )
    if mismatch > 0.001:
        raise BarValidationError(f"{symbol} {day}: close mismatch vs bhavcopy {mismatch:.4%}")


def resample(df_1min: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-min bars to 5min/15min. Bars labelled by window START;
    a bar is only 'closed' (usable) at index_time + freq."""
    out = df_1min.resample(freq, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return out.dropna(subset=["open"])


def atr_2h(df_5min: pd.DataFrame, window_minutes: int | None = None) -> pd.Series:
    """Expected 2-hour move, per 5-min bar (PLAN_v3 Section 8).

    True range on 5-min bars, rolling mean over the trailing 5 sessions of
    bars, scaled by sqrt(bars-per-window) (diffusive scaling). Strictly
    trailing — uses only data before each bar.
    """
    cfg = load_config()
    wmin = window_minutes or cfg["geometry"]["atr_window_minutes"]
    bars_per_window = wmin // 5
    prev_close = df_5min["close"].shift(1)
    tr = pd.concat(
        [
            df_5min["high"] - df_5min["low"],
            (df_5min["high"] - prev_close).abs(),
            (df_5min["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # ~75 5-min bars/session; 5 sessions of context
    atr_5 = tr.rolling(75 * 5, min_periods=75).mean()
    return atr_5 * np.sqrt(bars_per_window)


def session_dates(symbol: str, start: date, end: date) -> list[date]:
    """Trading dates with bar data on disk for a symbol."""
    df = load_1min(symbol, start, end)
    return sorted(set(df.index.date))
