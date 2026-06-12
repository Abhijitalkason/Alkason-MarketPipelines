"""09:30 morning momentum screen (PLAN_v3 Section 7).

score = w1·z(first-15min volume vs 20d same-bar avg)
      + w2·z(first-15min range  vs 20d same-bar avg)
      + w3·gap_quality + w4·preopen_imbalance + w5·delivery_z + w6·index_alignment

Point-in-time rule: consumes nothing time-stamped after 09:30. In backtest the
same function runs on stored data for a historical date; live it runs on the
recorder's data. Direction hint = sign(gap + first-15min return).
"""

from __future__ import annotations

import logging
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config, load_universe
from src.intraday.bars import load_1min, resample

logger = logging.getLogger(__name__)

FIRST_BAR = time(9, 15)


def _first15(df_1min: pd.DataFrame, day: date) -> pd.Series | None:
    bars = df_1min[(df_1min.index.date == day) & (df_1min.index.time >= FIRST_BAR)
                   & (df_1min.index.time < time(9, 30))]
    if bars.empty:
        return None
    return pd.Series({
        "open": bars["open"].iloc[0], "high": bars["high"].max(),
        "low": bars["low"].min(), "close": bars["close"].iloc[-1],
        "volume": bars["volume"].sum(),
    })


def _history_stats(df_1min: pd.DataFrame, day: date, lookback: int) -> tuple[float, float, float, float]:
    """Mean/std of first-15min volume and range over prior `lookback` sessions."""
    days = sorted({d for d in set(df_1min.index.date) if d < day})[-lookback:]
    vols, rngs = [], []
    for d in days:
        f = _first15(df_1min, d)
        if f is not None and f["open"] > 0:
            vols.append(f["volume"])
            rngs.append((f["high"] - f["low"]) / f["open"])
    if len(vols) < 10:
        raise ValueError(f"only {len(vols)} prior sessions for screen stats (need ≥10)")
    return float(np.mean(vols)), float(np.std(vols) or 1.0), float(np.mean(rngs)), float(np.std(rngs) or 1e-9)


def _delivery_z(symbol: str, day: date, lookback: int) -> float:
    """Prior-day delivery % z-score vs its own trailing distribution. 0 if files absent."""
    bdir = ROOT / load_config()["data"]["bhavcopy_path"]
    vals = []
    for d in pd.bdate_range(end=day - timedelta(days=1), periods=lookback).date:
        p = bdir / f"eq_{d.isoformat()}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["symbol", "deliv_per"])
            row = df[df.symbol == symbol]
            if not row.empty:
                vals.append(float(row["deliv_per"].iloc[0]))
    if len(vals) < 5:
        return 0.0
    s = pd.Series(vals)
    return float((s.iloc[-1] - s.mean()) / (s.std() or 1.0))


def _preopen_imbalance(symbol: str, day: date) -> float:
    p = ROOT / load_config()["data"]["preopen_path"] / f"{day.isoformat()}.parquet"
    if not p.exists():
        return 0.0
    df = pd.read_parquet(p)
    row = df[df.symbol == symbol]
    return float(row["imbalance"].iloc[0]) if not row.empty and pd.notna(row["imbalance"].iloc[0]) else 0.0


def screen_day(day: date, symbols: list[str] | None = None,
               weights: dict | None = None) -> pd.DataFrame:
    """Rank the universe at 09:30 on `day`. Returns top_n rows:
    [symbol, score, direction, gap, vol_z, rng_z, imbalance, delivery_z]."""
    cfg = load_config()
    sc = cfg["screen"]
    w = weights or sc["weights"]
    lookback = sc["lookback_days"]
    gap_lo, gap_hi = sc["gap_band"]
    symbols = symbols or load_universe()["symbol"].tolist()

    # index first-15 direction for alignment term (best effort)
    rows = []
    for sym in symbols:
        try:
            df = load_1min(sym, day - timedelta(days=lookback * 2), day)
        except Exception as e:  # noqa: BLE001 — symbol skipped is logged, not silent
            logger.warning("screen: %s skipped (%s)", sym, e)
            continue
        f15 = _first15(df, day)
        if f15 is None:
            continue
        prior_days = sorted({d for d in set(df.index.date) if d < day})
        if not prior_days:
            continue
        prev_close = float(df[df.index.date == prior_days[-1]]["close"].iloc[-1])
        try:
            vmean, vstd, rmean, rstd = _history_stats(df, day, lookback)
        except ValueError as e:
            logger.warning("screen: %s skipped (%s)", sym, e)
            continue

        gap = (f15["open"] - prev_close) / prev_close
        gap_quality = 1.0 if gap_lo <= abs(gap) <= gap_hi else 0.0
        vol_z = (f15["volume"] - vmean) / vstd
        rng = (f15["high"] - f15["low"]) / f15["open"]
        rng_z = (rng - rmean) / rstd
        f15_ret = (f15["close"] - f15["open"]) / f15["open"]
        imb = _preopen_imbalance(sym, day)
        dz = _delivery_z(sym, day, lookback)

        rows.append({
            "symbol": sym, "gap": gap, "vol_z": vol_z, "rng_z": rng_z,
            "f15_ret": f15_ret, "imbalance": imb, "delivery_z": dz,
            "gap_quality": gap_quality,
        })
    if not rows:
        raise RuntimeError(f"screen produced 0 candidates on {day} — check bar store")
    df = pd.DataFrame(rows)

    # index alignment: agreement of stock first-15 direction with universe median
    mkt_dir = np.sign(df["f15_ret"].median())
    df["index_alignment"] = (np.sign(df["f15_ret"]) == mkt_dir).astype(float)

    df["score"] = (
        w["volume_surge"] * df["vol_z"].clip(-3, 3)
        + w["range_expansion"] * df["rng_z"].clip(-3, 3)
        + w["gap_quality"] * df["gap_quality"]
        + w["preopen_imbalance"] * df["imbalance"].abs().clip(0, 1)
        + w["delivery_z"] * df["delivery_z"].clip(-3, 3)
        + w["index_alignment"] * df["index_alignment"]
    )
    df["direction"] = np.sign(df["gap"] + df["f15_ret"]).replace(0, 1).astype(int)
    top = df.sort_values("score", ascending=False).head(sc["top_n"]).reset_index(drop=True)
    logger.info("screen %s: top=%s", day, top.symbol.tolist())
    return top
