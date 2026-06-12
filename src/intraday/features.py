"""Intraday feature engineering (PLAN_v3 Section 9).

One row per (symbol, decision 15-min bar) in the entry window. Two channels:
  PRICE_FEATURES — consumed by price_model (LightGBM)
  FLOW_FEATURES  — consumed by flow_model (CatBoost)

Rules enforced here: every rolling stat strictly trails the decision bar;
volume/range normalized by 20-day same-time-of-day averages; no ffill across
sessions; the schema (FEATURE_ORDER) is versioned and checked at serve time.
"""

from __future__ import annotations

import logging
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config
from src.intraday.bars import atr_2h, load_1min, resample

logger = logging.getLogger(__name__)

PRICE_FEATURES = [
    "or_position", "or_breakout_atr", "minutes_since_open",
    "vwap_dist_atr", "vwap_slope", "bars_above_vwap",
    "gap_pct", "gap_filled_frac", "gap_f15_agree",
    "cumvol_vs_norm", "vol_slope_3",
    "ret_1b", "ret_3b", "ret_6b", "ret_12b", "accel",
    "rsi14_5m", "hod_proximity", "lod_proximity",
    "day_of_week", "atr_pct",
]
FLOW_FEATURES = [
    "oi_change_z", "basis_pct", "delivery_z", "preopen_imbalance", "fii_5d_z",
]
FEATURE_ORDER = PRICE_FEATURES + FLOW_FEATURES + ["direction"]
SCHEMA_VERSION = "v3.0"


class FeatureSchemaError(ValueError):
    pass


def check_schema(df: pd.DataFrame) -> None:
    """Serve-time guard (v1 post-mortem #5): exact column set + order."""
    cols = [c for c in df.columns if c in FEATURE_ORDER]
    if cols != FEATURE_ORDER:
        raise FeatureSchemaError(
            f"feature schema mismatch vs {SCHEMA_VERSION}: got {cols}"
        )


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _flow_row(symbol: str, day: date) -> dict:
    """Flow features from the latest bhavcopy strictly BEFORE `day`."""
    cfg = load_config()
    bdir = ROOT / cfg["data"]["bhavcopy_path"]
    out = {k: np.nan for k in FLOW_FEATURES}

    # futures OI + basis: trailing 20 sessions for z-score
    oi_hist, basis = [], np.nan
    for d in pd.bdate_range(end=day - timedelta(days=1), periods=20).date:
        p = bdir / f"fo_{d.isoformat()}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            row = df[df.symbol == symbol]
            if not row.empty:
                oi_hist.append(float(row["oi_change"].iloc[0]))
                basis = float(row["basis_pct"].iloc[0])
    if len(oi_hist) >= 5:
        s = pd.Series(oi_hist)
        out["oi_change_z"] = float((s.iloc[-1] - s.mean()) / (s.std() or 1.0))
        out["basis_pct"] = basis

    # delivery % z (same computation as screener, kept local to avoid coupling)
    dvals = []
    for d in pd.bdate_range(end=day - timedelta(days=1), periods=20).date:
        p = bdir / f"eq_{d.isoformat()}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["symbol", "deliv_per"])
            row = df[df.symbol == symbol]
            if not row.empty:
                dvals.append(float(row["deliv_per"].iloc[0]))
    if len(dvals) >= 5:
        s = pd.Series(dvals)
        out["delivery_z"] = float((s.iloc[-1] - s.mean()) / (s.std() or 1.0))

    pre = ROOT / cfg["data"]["preopen_path"] / f"{day.isoformat()}.parquet"
    if pre.exists():
        df = pd.read_parquet(pre)
        row = df[df.symbol == symbol]
        if not row.empty and pd.notna(row["imbalance"].iloc[0]):
            out["preopen_imbalance"] = float(row["imbalance"].iloc[0])

    fii = bdir / "fii_dii.parquet"
    if fii.exists():
        try:
            df = pd.read_parquet(fii)
            num = df.select_dtypes("number")
            if len(num) >= 5:
                col = num.columns[-1]
                s = num[col].tail(20)
                out["fii_5d_z"] = float((s.tail(5).mean() - s.mean()) / (s.std() or 1.0))
        except Exception as e:  # noqa: BLE001
            logger.warning("fii_dii parse failed: %s", e)
    return out


def features_for_day(symbol: str, day: date, direction: int = 1,
                     df_1min: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for every decision bar of one symbol-day (entry window only)."""
    cfg = load_config()
    geo = cfg["geometry"]
    if df_1min is None:
        df_1min = load_1min(symbol, day - timedelta(days=45), day)

    df5 = resample(df_1min, "5min")
    atr = atr_2h(df5)
    today1 = df_1min[df_1min.index.date == day]
    if today1.empty:
        return pd.DataFrame()
    df5_today = df5[df5.index.date == day]

    prior_days = sorted({d for d in set(df_1min.index.date) if d < day})
    if len(prior_days) < 20:
        return pd.DataFrame()
    prev_close = float(df_1min[df_1min.index.date == prior_days[-1]]["close"].iloc[-1])

    # 20-day same-time-of-day cumulative volume norm
    hist = df_1min[df_1min.index.date.astype("O").isin(prior_days[-20:])]
    cumvol_norm = hist.groupby([hist.index.date, hist.index.time])["volume"].sum() \
                      .groupby(level=1).mean().groupby(level=0).cumsum() if False else None
    # simpler + correct: per-day cumvol by time, averaged across days
    cv = hist.assign(d=hist.index.date, t=hist.index.time)
    cv["cum"] = cv.groupby("d")["volume"].cumsum()
    cumvol_by_time = cv.groupby("t")["cum"].mean()

    open_px = float(today1["open"].iloc[0])
    or_bars = today1[today1.index.time < time(9, 30)]
    if or_bars.empty:
        return pd.DataFrame()
    or_hi, or_lo = float(or_bars["high"].max()), float(or_bars["low"].min())
    gap = (open_px - prev_close) / prev_close

    flow = _flow_row(symbol, day)
    w_start = time.fromisoformat(geo["entry_window"][0])
    w_end = time.fromisoformat(geo["entry_window"][1])

    rows = []
    rsi = _rsi(df5["close"])
    for bar_ts in df5_today.index:
        if not (w_start <= bar_ts.time() < w_end) or bar_ts.time() < time(9, 30):
            continue
        if (bar_ts.minute % 15) != 0:   # decision on 15-min closes only
            continue
        upto = today1[today1.index <= bar_ts + pd.Timedelta("15min") - pd.Timedelta("1min")]
        c = float(upto["close"].iloc[-1])
        a = atr[atr.index <= bar_ts].dropna()
        if a.empty or a.iloc[-1] <= 0:
            continue
        a = float(a.iloc[-1])

        tp = (upto["high"] + upto["low"] + upto["close"]) / 3
        vwap_s = (tp * upto["volume"]).cumsum() / upto["volume"].cumsum().replace(0, np.nan)
        vwap = float(vwap_s.iloc[-1])
        vwap_slope = float(vwap_s.diff(15).iloc[-1] / a) if len(vwap_s) > 15 else 0.0
        above = (upto["close"].tail(30) > vwap_s.tail(30)).mean()

        d5 = df5[df5.index <= bar_ts]
        closes = d5["close"]
        hod = float(upto["high"].max())
        lod = float(upto["low"].min())
        cum_vol = float(upto["volume"].sum())
        norm = cumvol_by_time[cumvol_by_time.index <= upto.index[-1].time()]
        norm_v = float(norm.iloc[-1]) if len(norm) else np.nan

        gap_filled = 0.0
        if gap != 0:
            gap_filled = float(np.clip((open_px - c) / (open_px - prev_close), 0, 1))

        rows.append({
            "symbol": symbol, "ts": bar_ts, "date": day,
            "or_position": (c - or_lo) / (or_hi - or_lo) if or_hi > or_lo else 0.5,
            "or_breakout_atr": (c - or_hi) / a if c > or_hi else ((c - or_lo) / a if c < or_lo else 0.0),
            "minutes_since_open": (bar_ts.hour * 60 + bar_ts.minute) - (9 * 60 + 15),
            "vwap_dist_atr": (c - vwap) / a,
            "vwap_slope": vwap_slope,
            "bars_above_vwap": float(above),
            "gap_pct": gap,
            "gap_filled_frac": gap_filled,
            "gap_f15_agree": float(np.sign(gap) == np.sign(closes.pct_change().tail(3).sum())),
            "cumvol_vs_norm": cum_vol / norm_v if norm_v and norm_v > 0 else np.nan,
            "vol_slope_3": float(d5["volume"].tail(3).diff().mean() / (d5["volume"].tail(20).mean() or 1)),
            "ret_1b": float(closes.pct_change(1).iloc[-1]),
            "ret_3b": float(closes.pct_change(3).iloc[-1]),
            "ret_6b": float(closes.pct_change(6).iloc[-1]),
            "ret_12b": float(closes.pct_change(12).iloc[-1]) if len(closes) > 12 else np.nan,
            "accel": float(closes.pct_change(1).iloc[-1] - closes.pct_change(1).iloc[-2]) if len(closes) > 2 else 0.0,
            "rsi14_5m": float(rsi[rsi.index <= bar_ts].iloc[-1]) if not rsi[rsi.index <= bar_ts].dropna().empty else np.nan,
            "hod_proximity": (hod - c) / a,
            "lod_proximity": (c - lod) / a,
            "day_of_week": float(bar_ts.dayofweek),
            "atr_pct": a / c,
            "direction": float(direction),
            **flow,
        })
    return pd.DataFrame(rows)


def build_matrix(plan: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for a screen plan: rows [date, symbol, direction].
    Drops rows with missing PRICE features (logged); flow NaNs filled 0
    (absence of flow data is itself information-neutral)."""
    frames = []
    for _, r in plan.iterrows():
        try:
            f = features_for_day(r["symbol"], r["date"], int(r["direction"]))
            if not f.empty:
                frames.append(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("features: %s %s skipped (%s)", r["symbol"], r["date"], e)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    n0 = len(out)
    out = out.dropna(subset=PRICE_FEATURES)
    out[FLOW_FEATURES] = out[FLOW_FEATURES].fillna(0.0)
    if n0 - len(out):
        logger.info("features: dropped %d/%d rows with missing price features", n0 - len(out), n0)
    return out
