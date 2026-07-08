"""PLAN_v6 §4.1(b) — the GO/KILL basis: a PIT-by-construction liquidity
universe (F20).

Top-N symbols by trailing 6-month median daily turnover, recomputed at each
month start from the full bhavcopy market using STRICTLY-TRAILING data only.
Survivorship-free by construction: a name that later fell was liquid *then*
and is included *then*.

Surveillance exclusions (F15) `[VERIFY — unknown #12]`: historical PIT
ASM/GSM/T2T lists are not on disk, so a structural proxy is applied and
disclosed: symbols with > MAX_LOCKED_SHARE circuit-locked sessions (high ==
low all day) in the trailing window are excluded. The proxy under-excludes
soft surveillance names; the report states this openly.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TOP_N = 100
LOOKBACK_MONTHS = 6
MIN_TRADED_SHARE = 0.9      # must have traded >= 90% of trailing sessions
MAX_LOCKED_SHARE = 0.05     # circuit-lock proxy for surveillance exclusion


def monthly_pit_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (month_start, symbol) for the top-N liquidity universe.

    For month m: rank by median daily turnover over [m - 6 months, m), using
    only rows dated strictly before m. The first constructible month is the
    first with a full lookback window.
    """
    p = panel[["symbol", "date", "turnover_lacs", "locked_day"]].copy()
    p["month"] = p["date"].dt.to_period("M")
    months = sorted(p["month"].unique())
    first = months[0].to_timestamp() + pd.DateOffset(months=LOOKBACK_MONTHS)
    out = []
    for m in months:
        m_start = m.to_timestamp()
        if m_start < first:
            continue
        lo = m_start - pd.DateOffset(months=LOOKBACK_MONTHS)
        win = p[(p["date"] >= lo) & (p["date"] < m_start)]
        n_sessions = win["date"].nunique()
        stats = win.groupby("symbol").agg(
            med_turnover=("turnover_lacs", "median"),
            n_days=("date", "count"),
            locked_share=("locked_day", "mean"),
        )
        eligible = stats[(stats["n_days"] >= MIN_TRADED_SHARE * n_sessions)
                         & (stats["locked_share"] <= MAX_LOCKED_SHARE)]
        top = eligible.nlargest(TOP_N, "med_turnover")
        out.append(pd.DataFrame({"month": m, "symbol": top.index}))
    uni = pd.concat(out, ignore_index=True)
    logger.info("PIT universe: %d months, %d unique symbols",
                uni["month"].nunique(), uni["symbol"].nunique())
    return uni


def in_universe_mask(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """Boolean mask over panel rows: is (symbol, month-of-date) in the PIT
    universe? Signal rows outside it never reach the GO/KILL grid."""
    key = panel["symbol"] + "|" + panel["date"].dt.to_period("M").astype(str)
    uni_key = set(universe["symbol"] + "|" + universe["month"].astype(str))
    return key.isin(uni_key)
