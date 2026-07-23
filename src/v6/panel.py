"""PLAN_v6 §4.1 — daily panel from the bhavcopy store, corporate-action
adjusted, with strictly-trailing ATR(14).

Bhavcopy prices are AS-TRADED (unlike the Upstox 1-min store, which is
back-adjusted at source — see src/intraday/corporate_actions.py). Barrier
resolution must run on adjusted paths (F19), so this module BACK-adjusts:
prices strictly before an ex-date are multiplied by that action's factor
(volume divided). Ex-dates falling inside a hold window are counted by the
labeler and reported.

**F25 (review 2, 2026-07-10):** the CSV at data/reference/corporate_actions.csv
covers BONUS issues only (verified: 170/170 rows are "Bonus x:y") — splits and
demergers are absent, and NSE bhavcopy's own `prev_close` field is NOT
adjusted at source for either (verified against NAUKRI/DRREDDY/SHRIRAMFIN
splits and the TATAMOTORS demerger: prev_close carries the raw prior close
straight through the ex-date). `_detect_residual_ca()` below closes this gap:
after the CSV pass, any day-to-day price discontinuity that SURVIVES it is,
by construction, an undocumented corporate action, and is back-adjusted the
same way using the empirical open/prev-close ratio as the factor. This folds
that day's true return into the factor (a small, disclosed imprecision —
splits/bonuses would ideally use the official ratio; demergers have no
"official" ratio to round to, so the empirical residual-value ratio is
actually the more correct number there).

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
CA_EVENTS_CACHE = ROOT / "data" / "processed" / "v6_ca_events_all.parquet"

# F25/R1: a same-day open/prev-close ratio outside this band cannot be an
# organic move for top-100-liquidity names (NSE price bands cap most sessions
# well inside it) — it is treated as an undocumented corporate action.
RESIDUAL_CA_LOW, RESIDUAL_CA_HIGH = 0.5, 2.0


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

    df, csv_events = _adjust(df)
    df, residual_events = _detect_residual_ca(df)
    df = _add_atr(df)

    all_events = pd.concat([csv_events, residual_events], ignore_index=True)
    all_events.to_parquet(CA_EVENTS_CACHE, index=False)

    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_CACHE, index=False)
    logger.info("panel: %d rows, %d symbols, %s..%s", len(df),
                df["symbol"].nunique(), df["date"].min().date(), df["date"].max().date())
    logger.info("CA events: %d from CSV (bonus), %d detected residual (split/demerger/other)",
                len(csv_events), len(residual_events))
    return df


def load_ca_events() -> pd.DataFrame:
    """The combined (symbol, ex_date) table used by the labeler's F19
    ex-date-in-hold counter — CSV bonuses + detected residual events (F25).
    Rebuilt as a side effect of load_panel(); call load_panel() first if the
    panel cache exists but this file is missing."""
    if not CA_EVENTS_CACHE.exists():
        load_panel(use_cache=False)
    return pd.read_parquet(CA_EVENTS_CACHE)


def _adjust(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-adjust as-traded prices: rows strictly before an ex-date are
    multiplied by the cumulative product of factors with ex_date > row date.
    Returns the adjusted frame and the CSV events table (for F19 tracking)."""
    ca = pd.read_csv(CA_PATH, comment="#", parse_dates=["ex_date"])
    df["adj_factor"] = 1.0
    for sym, ex_date, factor in zip(ca["symbol"], ca["ex_date"], ca["factor"]):
        mask = (df["symbol"] == sym) & (df["date"] < ex_date)
        if mask.any():
            df.loc[mask, "adj_factor"] *= factor
    for col in ("open", "high", "low", "close", "prev_close"):
        df[col] = df[col] * df["adj_factor"]
    df["volume"] = df["volume"] / df["adj_factor"]
    events = ca[["symbol", "ex_date", "purpose", "factor"]].copy()
    return df, events


def _detect_residual_ca(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F25/R1 — detect and back-adjust corporate actions absent from the CSV
    (splits, demergers): a day-to-day open/prev-close ratio that survives the
    CSV-bonus pass and falls outside [RESIDUAL_CA_LOW, RESIDUAL_CA_HIGH] is,
    for these liquid names, not an organic move — it is the discontinuity
    itself, used directly as the back-adjustment factor (see module
    docstring for the precision tradeoff this implies)."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    ratio = df["open"] / prev_close
    flagged = prev_close.notna() & ((ratio < RESIDUAL_CA_LOW) | (ratio > RESIDUAL_CA_HIGH))

    events = df.loc[flagged, ["symbol", "date"]].rename(columns={"date": "ex_date"}).copy()
    events["purpose"] = "detected_residual"
    events["factor"] = ratio[flagged].to_numpy()

    if len(events):
        df["resid_factor"] = 1.0
        for sym, ex_date, factor in zip(events["symbol"], events["ex_date"], events["factor"]):
            mask = (df["symbol"] == sym) & (df["date"] < ex_date)
            if mask.any():
                df.loc[mask, "resid_factor"] *= factor
        for col in ("open", "high", "low", "close", "prev_close"):
            df[col] = df[col] * df["resid_factor"]
        df["volume"] = df["volume"] / df["resid_factor"]
        df["adj_factor"] = df["adj_factor"] * df["resid_factor"]
        df = df.drop(columns=["resid_factor"])
        logger.info("residual CA detector: %d discontinuities flagged (%s)",
                    len(events), ", ".join(sorted(events["symbol"].unique())[:10]))
    return df, events.reset_index(drop=True)


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
