"""Intraday feature engineering (PLAN_v3 §9; v3_supplementry §4) — schema v3.1.

One row per (symbol, decision 15-min bar) in the entry window. Two channels:
  PRICE_FEATURES — consumed by price_model (LightGBM)
  FLOW_FEATURES  — consumed by flow_model (CatBoost), each with a _present
                   flag so imputation never masquerades as data (B-5)

Rules enforced here: every rolling stat strictly trails the decision bar;
FII/DII filtered strictly before the decision day (B-4); volume normalized by
20-day same-time-of-day averages; no ffill across sessions; the schema is
versioned and checked at every serve-time predict (H-10).

Missingness policy (build_matrix):
  flow NaN → 0.0 ONLY alongside _present=0; flow_real_share logged each build;
  when the flow channel is required, share below the configured floor RAISES
  FlowDataError (training on majority-imputed flow was v1's FinBERT failure).
  Price-NaN rows are dropped but COUNTED; skip share above tolerance RAISES.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config, point_in_time
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
FLOW_PRESENT = [f + "_present" for f in FLOW_FEATURES]
FEATURE_ORDER = PRICE_FEATURES + FLOW_FEATURES + FLOW_PRESENT + ["direction"]
SCHEMA_VERSION = "v3.1"


class FeatureSchemaError(ValueError):
    pass


class FlowDataError(RuntimeError):
    pass


class DatasetThinningError(RuntimeError):
    pass


def check_schema(df: pd.DataFrame) -> None:
    """Serve-time guard (v1 post-mortem #5 / H-10): every schema column present,
    in order, with no NaN in price features. Called before EVERY predict —
    backtester, paper runner, and API."""
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise FeatureSchemaError(f"schema {SCHEMA_VERSION}: missing columns {missing}")
    cols = [c for c in df.columns if c in FEATURE_ORDER]
    if cols != FEATURE_ORDER:
        raise FeatureSchemaError(f"schema {SCHEMA_VERSION}: column order mismatch: {cols}")
    bad = df[PRICE_FEATURES].isna().any()
    if bad.any():
        raise FeatureSchemaError(
            f"schema {SCHEMA_VERSION}: NaN in price features {list(bad[bad].index)}"
        )


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bhav_dir() -> Path:
    p = Path(load_config()["data"]["bhavcopy_path"])
    return p if p.is_absolute() else ROOT / p


def _preopen_dir() -> Path:
    p = Path(load_config()["data"]["preopen_path"])
    return p if p.is_absolute() else ROOT / p


def _flow_row(symbol: str, day: date) -> dict:
    """Flow features for `day`, with a _present flag per feature.

    Point-in-time guarantees:
      - bhavcopy inputs read strictly-prior-day files only;
      - pre-open snapshot filtered to capture_ts ≤ 09:30 of `day`;
      - FII/DII rows filtered to date strictly before `day` (B-4 fix).
    """
    out: dict[str, float] = {}
    for k in FLOW_FEATURES:
        out[k] = np.nan
        out[k + "_present"] = 0.0
    bdir = _bhav_dir()
    decision_floor = pd.Timestamp(datetime.combine(day, time(9, 30)))

    # futures OI + basis: trailing 20 sessions for z-score (strictly-prior files)
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
        std = float(s.std())
        if std > 0:
            out["oi_change_z"] = float((s.iloc[-1] - s.mean()) / std)
            out["oi_change_z_present"] = 1.0
        if np.isfinite(basis):
            out["basis_pct"] = basis
            out["basis_pct_present"] = 1.0

    # delivery % z (strictly-prior files)
    dvals = []
    for d in pd.bdate_range(end=day - timedelta(days=1), periods=20).date:
        p = bdir / f"eq_{d.isoformat()}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["symbol", "deliv_per"])
            row = df[df.symbol == symbol]
            if not row.empty and pd.notna(row["deliv_per"].iloc[0]):
                dvals.append(float(row["deliv_per"].iloc[0]))
    if len(dvals) >= 5:
        s = pd.Series(dvals)
        std = float(s.std())
        if std > 0:
            out["delivery_z"] = float((s.iloc[-1] - s.mean()) / std)
            out["delivery_z_present"] = 1.0

    # pre-open imbalance — same-day file, PIT-filtered on capture_ts
    pre = _preopen_dir() / f"{day.isoformat()}.parquet"
    if pre.exists():
        df = point_in_time(pd.read_parquet(pre), day, decision_floor, date_col="__none__")
        row = df[df.symbol == symbol]
        if not row.empty and pd.notna(row["imbalance"].iloc[0]):
            out["preopen_imbalance"] = float(row["imbalance"].iloc[0])
            out["preopen_imbalance_present"] = 1.0

    # FII/DII 5-day net z — STRICTLY before `day` (B-4). Parse failures RAISE;
    # only file absence degrades (to NaN + present=0).
    fii = bdir / "fii_dii.parquet"
    if fii.exists():
        df = pd.read_parquet(fii)
        if "date" not in df.columns or "fii_net_cr" not in df.columns:
            raise FlowDataError(
                "fii_dii.parquet lacks date/fii_net_cr columns — run "
                "`python main.py --mode bhavcopy` to rebuild (schema migrated in v3.1)"
            )
        df = df[pd.to_datetime(df["date"]) < pd.Timestamp(day)].sort_values("date")
        s = df["fii_net_cr"].tail(20)
        if len(s) >= 5:
            std = float(s.std())
            if std > 0:
                out["fii_5d_z"] = float((s.tail(5).mean() - s.mean()) / std)
                out["fii_5d_z_present"] = 1.0
    return out


def features_for_day(symbol: str, day: date, direction: int = 1,
                     df_1min: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for every decision bar of one symbol-day (entry window only).

    df_1min injection is the live-path contract (B-2): the paper runner passes
    history + the recorder's provisional bars for today; backtest/training load
    from the store. Same function either way — train == serve by construction.
    """
    cfg = load_config()
    geo = cfg["geometry"]
    if df_1min is None:
        df_1min = load_1min(symbol, day - timedelta(days=45), day)

    df5 = resample(df_1min, cfg["data"]["bar_freq_feature"])
    atr = atr_2h(df5)
    today1 = df_1min[df_1min.index.date == day]
    if today1.empty:
        return pd.DataFrame()
    df5_today = df5[df5.index.date == day]

    prior_days = sorted({d for d in set(df_1min.index.date) if d < day})
    if len(prior_days) < 20:
        return pd.DataFrame()
    prev_close = float(df_1min[df_1min.index.date == prior_days[-1]]["close"].iloc[-1])

    # 20-day same-time-of-day cumulative volume norm (U-curve normalization)
    day_index = pd.Series(df_1min.index.date, index=df_1min.index)
    hist = df_1min[day_index.isin(prior_days[-20:]).to_numpy()]
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
        entry_ts = bar_ts + pd.Timedelta("15min")
        upto = today1[today1.index <= entry_ts - pd.Timedelta("1min")]
        c = float(upto["close"].iloc[-1])
        # ATR as-of the SAME instant the labeler uses to place target/stop
        # (last 5-min bar closed by entry_ts) — the feature the model sees and the
        # geometry the label encodes must share one ruler. Still strictly trailing:
        # the bar labelled (entry_ts-5min) is complete from data < entry_ts.
        a = atr[atr.index < entry_ts].dropna()
        if a.empty or a.iloc[-1] <= 0:
            continue
        a = float(a.iloc[-1])

        tp = (upto["high"] + upto["low"] + upto["close"]) / 3
        vwap_s = (tp * upto["volume"]).cumsum() / upto["volume"].cumsum().replace(0, np.nan)
        vwap = float(vwap_s.iloc[-1])
        vwap_slope = float(vwap_s.diff(15).iloc[-1] / a) if len(vwap_s) > 15 else 0.0
        above = (upto["close"].tail(30) > vwap_s.tail(30)).mean()

        d5 = df5[df5.index < entry_ts]   # same as-of as ATR/VWAP — one ruler for the whole row
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
            "rsi14_5m": float(rsi[rsi.index < entry_ts].iloc[-1]) if not rsi[rsi.index < entry_ts].dropna().empty else np.nan,
            "hod_proximity": (hod - c) / a,
            "lod_proximity": (c - lod) / a,
            "day_of_week": float(bar_ts.dayofweek),
            "atr_pct": a / c,
            "direction": float(direction),
            **flow,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    meta = ["symbol", "ts", "date"]
    return df[meta + FEATURE_ORDER]      # canonical column order for check_schema


def impute_flow(df: pd.DataFrame) -> pd.DataFrame:
    """flow NaN → 0.0 ONLY alongside its _present flag (already 0 for imputed
    values) — information preserved (B-5). Returns a copy."""
    out = df.copy()
    out[FLOW_FEATURES] = out[FLOW_FEATURES].fillna(0.0)
    return out


def flow_real_share(df: pd.DataFrame) -> float:
    """Mean over rows of mean(FLOW_PRESENT) — the flow channel's real-data share."""
    if df.empty:
        return 0.0
    return float(df[FLOW_PRESENT].mean(axis=1).mean())


def build_matrix(plan: pd.DataFrame, require_flow: bool | None = None) -> pd.DataFrame:
    """Feature matrix for a screen plan: rows [date, symbol, direction].

    require_flow defaults to config gate.flow_model_active: when the flow
    channel is in use, a real-data share below the floor RAISES FlowDataError;
    in price-only mode it is logged (the trainer uses the logged share to
    decide activation). Row skips are counted; over-tolerance RAISES (M-1).
    """
    cfg = load_config()
    if require_flow is None:
        require_flow = bool(cfg["gate"]["flow_model_active"])

    frames, failed = [], 0
    for _, r in plan.iterrows():
        try:
            f = features_for_day(r["symbol"], r["date"], int(r["direction"]))
            if not f.empty:
                frames.append(f)
            # An empty return is an EXPECTED outcome (insufficient warm-up history,
            # no opening-range bars yet) — not an exception, so it is NOT counted
            # against the skip tolerance (M-1 §4.3 counts feature *exceptions*).
            # Genuine bar holes are caught upstream by validate_1min /
            # cross_check_bhavcopy, not by silently shrinking the matrix here.
        except FlowDataError:
            raise  # structural problem — never count this as a row skip
        except Exception as e:  # noqa: BLE001 — counted, raises over tolerance below
            failed += 1
            logger.warning("features: %s %s skipped (%s)", r["symbol"], r["date"], e)
    if failed and failed > len(plan) * cfg["data"]["max_skip_share"]:
        raise DatasetThinningError(
            f"{failed}/{len(plan)} symbol-days failed feature build "
            f"(> {cfg['data']['max_skip_share']:.0%} tolerance) — investigate the bar store"
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    n0 = len(out)
    out = out.dropna(subset=PRICE_FEATURES)
    dropped = n0 - len(out)
    if dropped and dropped > n0 * cfg["data"]["max_skip_share"]:
        raise DatasetThinningError(
            f"{dropped}/{n0} rows dropped for missing price features "
            f"(> {cfg['data']['max_skip_share']:.0%} tolerance)"
        )
    if dropped:
        logger.info("features: dropped %d/%d rows with missing price features", dropped, n0)

    share = flow_real_share(out)
    logger.info("features: flow_real_share=%.3f over %d rows", share, len(out))
    floor = 1.0 - cfg["data"]["max_flow_missing_share"]
    if require_flow and share < floor:
        raise FlowDataError(
            f"flow_real_share {share:.2f} < {floor:.2f} — training the flow channel on "
            f"majority-imputed data is the v1 FinBERT failure; backfill bhavcopies first "
            f"or set gate.flow_model_active=false"
        )
    return impute_flow(out)
