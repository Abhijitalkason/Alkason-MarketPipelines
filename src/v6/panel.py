"""PLAN_v6 §4.1 — daily panel from the bhavcopy store, corporate-action
adjusted, with strictly-trailing ATR(14).

Bhavcopy prices are AS-TRADED (unlike the Upstox 1-min store, which is
back-adjusted at source — see src/intraday/corporate_actions.py). Barrier
resolution must run on adjusted paths (F19), so this module BACK-adjusts:
prices strictly before an ex-date are multiplied by that action's factor
(volume divided). Ex-dates falling inside a hold window are counted by the
labeler and reported.

ATR is strictly trailing: atr_pct on row (symbol, D) uses true ranges of days
<= D only, so a signal decided at D's close never sees D+1.
"""

from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd

from src.intraday import ROOT

logger = logging.getLogger(__name__)

BHAVCOPY_DIR = ROOT / "data" / "bhavcopy"
CA_PATH = ROOT / "data" / "reference" / "corporate_actions.csv"
ATR_WINDOW = 14

PANEL_CACHE = ROOT / "data" / "processed" / "v6_panel.parquet"


def load_panel(use_cache: bool = True) -> pd.DataFrame:
    """Long panel: one row per (symbol, date) with adjusted OHLC, turnover,
    delivery %, prev adjusted close, trailing atr_pct, and overnight-gap
    metadata. Sorted by (symbol, date)."""
    if use_cache and PANEL_CACHE.exists():
        return pd.read_parquet(PANEL_CACHE)

    files = sorted(glob.glob(str(BHAVCOPY_DIR / "eq_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no bhavcopy files under {BHAVCOPY_DIR}")
    frames = [pd.read_parquet(f, columns=[
        "symbol", "date", "prev_close", "open_price", "high_price",
        "low_price", "close_price", "ttl_trd_qnty", "turnover_lacs", "deliv_per",
    ]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "open_price": "open", "high_price": "high", "low_price": "low",
        "close_price": "close", "ttl_trd_qnty": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["deliv_per"] = pd.to_numeric(df["deliv_per"], errors="coerce")

    # drop rows that cannot be priced
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df = _adjust(df)
    df = _add_atr(df)

    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_CACHE, index=False)
    logger.info("panel: %d rows, %d symbols, %s..%s", len(df),
                df["symbol"].nunique(), df["date"].min().date(), df["date"].max().date())
    return df


def _adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust as-traded prices: rows strictly before an ex-date are
    multiplied by the cumulative product of factors with ex_date > row date."""
    ca = pd.read_csv(CA_PATH, comment="#", parse_dates=["ex_date"])
    df["adj_factor"] = 1.0
    for sym, ex_date, factor in zip(ca["symbol"], ca["ex_date"], ca["factor"]):
        mask = (df["symbol"] == sym) & (df["date"] < ex_date)
        if mask.any():
            df.loc[mask, "adj_factor"] *= factor
    for col in ("open", "high", "low", "close", "prev_close"):
        df[col] = df[col] * df["adj_factor"]
    df["volume"] = df["volume"] / df["adj_factor"]
    return df


def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing ATR(14) as a fraction of close, computed on adjusted prices.

    prev_close is recomputed from the adjusted series (the bhavcopy column can
    disagree with shift(1) across corporate actions and long gaps). True range
    uses the standard max(h-l, |h-pc|, |l-pc|)."""
    g = df.groupby("symbol", sort=False)
    pc = g["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    df["prev_close_adj"] = pc
    df["atr_pct"] = (
        tr.groupby(df["symbol"], sort=False)
        .rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()
        .reset_index(level=0, drop=True)
    ) / df["close"]
    # circuit-lock proxy: the whole session traded at one price (h == l).
    df["locked_day"] = df["high"] == df["low"]
    return df
