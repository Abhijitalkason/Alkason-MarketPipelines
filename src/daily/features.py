"""Daily feature engineering — schema daily-v1. One row per (symbol, decision day).

Mirrors src/intraday/features.py conventions exactly:
  - a FIXED, versioned schema checked before every predict (check_schema);
  - every OPTIONAL feature carries a `_present` flag so imputation never
    masquerades as data (B-5) — F&O, global, macro, sentiment blocks;
  - strictly-trailing computation: a decision-day row uses only data dated on/
    before that day (India EOD) or released on/before its close (global/macro),
    so the no-lookahead property test passes by construction.

Decision model (next_day/swing): decision at day T's close (India EOD of T is
public after the close). Features use the adjusted daily series through T and
global/macro with release_ts <= T close. Entry/label (in labeler.py) fill at the
NEXT session open — never at the close you just observed.

Missingness policy (build_matrix): optional NaN → 0.0 ONLY alongside _present=0;
required (market) NaN rows are dropped but COUNTED; skip share over tolerance
RAISES DatasetThinningError. There is no flow-floor raise here (global/macro are
genuinely optional in v1); the v1 model trains on market + F&O alone.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.daily import daily_path, load_daily_config
from src.daily.panel import load_symbol_daily

logger = logging.getLogger(__name__)


# ── cached file readers (mtime-keyed) — assembled datasets re-read these tens of
# thousands of times; caching turns the per-call I/O into one read per file. ─────
@lru_cache(maxsize=128)
def _read_parquet_cached(path_str: str, mtime: int) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@lru_cache(maxsize=128)
def _read_csv_cached(path_str: str, mtime: int) -> pd.DataFrame:
    return pd.read_csv(path_str, comment="#")


def _cached_parquet(path: Path) -> pd.DataFrame | None:
    return _read_parquet_cached(str(path), path.stat().st_mtime_ns) if path.exists() else None


def _cached_csv(path: Path) -> pd.DataFrame | None:
    return _read_csv_cached(str(path), path.stat().st_mtime_ns) if path.exists() else None

# ── the fixed schema ──────────────────────────────────────────────────
# Required market block (computed from the on-disk panel; never NaN after build).
MARKET_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "vol_10d", "vol_20d", "rsi_14", "ma_dist_20", "ma_dist_50",
    "atr_pct", "pos_52w", "gap_prev", "vol_ratio_20",
    "deliv_per", "deliv_z", "turnover_z",
]
# Optional blocks — each value paired with a <feat>_present flag.
FNO_FEATURES = ["oi_change_z", "basis_pct", "fii_5d_z"]
GLOBAL_FEATURES = [
    "overnight_spx", "overnight_ndx", "gift_nifty_ret", "asia_ret",
    "india_vix", "india_vix_chg", "dxy_chg", "us10y_chg",
    "crude_chg", "gold_chg", "usdinr_chg",
]
MACRO_FEATURES = [
    "days_to_macro", "days_since_macro", "days_to_earnings", "days_since_earnings",
    "monsoon_departure",
]
SENTIMENT_FEATURES = ["sent_mean_5d", "sent_surprise", "sent_coverage"]

OPTIONAL_FEATURES = FNO_FEATURES + GLOBAL_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES
PRESENT_FLAGS = [f + "_present" for f in OPTIONAL_FEATURES]
FEATURE_ORDER = MARKET_FEATURES + OPTIONAL_FEATURES + PRESENT_FLAGS
SCHEMA_VERSION = "daily-v1"


class FeatureSchemaError(ValueError):
    pass


class DatasetThinningError(RuntimeError):
    pass


def check_schema(df: pd.DataFrame) -> None:
    """Serve-time guard (H-10): every schema column present, in order, no NaN in
    required market features. Called before EVERY predict — backtester, screener,
    API."""
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise FeatureSchemaError(f"schema {SCHEMA_VERSION}: missing columns {missing}")
    cols = [c for c in df.columns if c in FEATURE_ORDER]
    if cols != FEATURE_ORDER:
        raise FeatureSchemaError(f"schema {SCHEMA_VERSION}: column order mismatch")
    bad = df[MARKET_FEATURES].isna().any()
    if bad.any():
        raise FeatureSchemaError(
            f"schema {SCHEMA_VERSION}: NaN in market features {list(bad[bad].index)}"
        )


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR(n) as a fraction of close, on the daily series."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean() / close


def _zscore_last(s: pd.Series, window: int) -> float:
    s = s.dropna().tail(window)
    if len(s) < max(5, window // 2):
        return np.nan
    std = float(s.std())
    return float((s.iloc[-1] - s.mean()) / std) if std > 0 else np.nan


def _zcol(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling z-score (matches _zscore_last at each point), vectorized."""
    m = s.rolling(window, min_periods=max(5, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 2)).std()
    return ((s - m) / sd).where(sd > 0, np.nan)


def market_fno_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All MARKET + F&O features for EVERY row of an adjusted daily series, in one
    vectorized pass (indexed by date). Each value is strictly trailing (rolling is
    causal), so the row at day T is identical whether or not future rows exist —
    the no-lookahead guarantee, computed O(n) instead of O(n²)."""
    fcfg = load_daily_config()["features"]
    w = fcfg["zscore_window"]
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    for n in fcfg["return_windows"]:
        out[f"ret_{n}d"] = close.pct_change(n)
    rets = close.pct_change()
    for n in fcfg["vol_windows"]:
        out[f"vol_{n}d"] = rets.rolling(n).std()
    out["rsi_14"] = _rsi(close, fcfg["rsi_window"])
    for n in fcfg["ma_windows"]:
        out[f"ma_dist_{n}"] = close / close.rolling(n).mean() - 1.0
    out["atr_pct"] = _atr_pct(df, fcfg["atr_window"])
    roll = close.rolling(252, min_periods=20)
    rng = roll.max() - roll.min()
    out["pos_52w"] = ((close - roll.min()) / rng).where(rng > 0, 0.5)
    pc = df["prev_close"].astype(float)
    out["gap_prev"] = (df["open"].astype(float) / pc - 1.0).where(pc > 0, 0.0)
    vma = vol.rolling(w).mean()
    out["vol_ratio_20"] = (vol / vma).where(vma > 0, 1.0)
    out["deliv_per"] = df["deliv_per"].astype(float) / 100.0
    out["deliv_z"] = _zcol(df["deliv_per"].astype(float), w)
    out["turnover_z"] = _zcol(df["turnover_lacs"].astype(float), w)
    for f in FNO_FEATURES:
        out[f] = np.nan
        out[f + "_present"] = 0.0
    if fcfg.get("fno_enabled"):
        if "oi_change" in df.columns:
            oiz = _zcol(df["oi_change"].astype(float), w)
            out["oi_change_z"] = oiz
            out["oi_change_z_present"] = oiz.notna().astype(float)
        if "basis_pct" in df.columns:
            b = df["basis_pct"].astype(float)
            out["basis_pct"] = b
            out["basis_pct_present"] = b.notna().astype(float)
    return out


# ── optional blocks (PIT readers; present=0 when data absent) ──────────

def _global_dir() -> Path:
    return daily_path("global_path")


def _read_global(name: str, decision_ts: pd.Timestamp) -> pd.Series:
    """A global daily series (date, value, release_ts) filtered to release_ts <=
    decision_ts, indexed by date. Empty if the file is absent."""
    df = _cached_parquet(_global_dir() / f"{name}.parquet")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "release_ts" in df.columns:
        df = df[pd.to_datetime(df["release_ts"]) <= decision_ts]
    df = df.sort_values("date")
    return pd.Series(df["value"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]))


def _global_row(decision_ts: pd.Timestamp) -> dict:
    """Symbol-independent → memoized per decision_ts (dataset assembly calls this
    once per (symbol, day); without the cache that is O(symbols × days) re-reads)."""
    return dict(_global_row_cached(decision_ts))


@lru_cache(maxsize=8192)
def _global_row_cached(decision_ts: pd.Timestamp) -> tuple:
    return tuple(_global_row_impl(decision_ts).items())


def _global_row_impl(decision_ts: pd.Timestamp) -> dict:
    out = {f: np.nan for f in GLOBAL_FEATURES}
    out.update({f + "_present": 0.0 for f in GLOBAL_FEATURES})
    if not load_daily_config()["features"].get("global_enabled"):
        return out
    # change-type features: pct change of the last two PIT levels
    changes = {
        "overnight_spx": "spx", "overnight_ndx": "ndx", "gift_nifty_ret": "gift_nifty",
        "asia_ret": "asia", "dxy_chg": "dxy", "us10y_chg": "us10y",
        "crude_chg": "crude", "gold_chg": "gold", "usdinr_chg": "usdinr",
    }
    for feat, series in changes.items():
        s = _read_global(series, decision_ts)
        if len(s) >= 2 and pd.notna(s.iloc[-1]) and pd.notna(s.iloc[-2]) and s.iloc[-2] != 0:
            out[feat] = float(s.iloc[-1] / s.iloc[-2] - 1.0)
            out[feat + "_present"] = 1.0
    # level + change for India VIX
    vix = _read_global("india_vix", decision_ts)
    if len(vix) >= 1 and pd.notna(vix.iloc[-1]):
        out["india_vix"] = float(vix.iloc[-1]); out["india_vix_present"] = 1.0
    if len(vix) >= 2 and pd.notna(vix.iloc[-1]) and pd.notna(vix.iloc[-2]) and vix.iloc[-2] != 0:
        out["india_vix_chg"] = float(vix.iloc[-1] / vix.iloc[-2] - 1.0)
        out["india_vix_chg_present"] = 1.0
    return out


def _macro_row(symbol: str, day: date, decision_ts: pd.Timestamp) -> dict:
    out = {f: np.nan for f in MACRO_FEATURES}
    out.update({f + "_present": 0.0 for f in MACRO_FEATURES})
    if not load_daily_config()["features"].get("macro_enabled"):
        return out
    df = _cached_csv(daily_path("reference_path") / "macro_calendar.csv")
    if df is not None:
        if "release_ts" in df.columns:
            df = df[pd.to_datetime(df["release_ts"]) <= decision_ts]
        if "event_date" in df.columns and not df.empty:
            ed = pd.to_datetime(df["event_date"]).dt.date
            future = sorted(d for d in ed if d > day)
            past = sorted(d for d in ed if d <= day)
            if future:
                out["days_to_macro"] = float((future[0] - day).days); out["days_to_macro_present"] = 1.0
            if past:
                out["days_since_macro"] = float((day - past[-1]).days); out["days_since_macro_present"] = 1.0
    # per-stock earnings dates from events.csv (event_type == 'results')
    df = _cached_csv(daily_path("reference_path") / "events.csv")
    if df is not None:
        # PIT: if an announcement timestamp is present, only use events public by
        # decision_ts (an earnings DATE is scheduled ahead, but treat its presence
        # as knowable only from release_ts). Mirrors the macro_calendar/monsoon guard.
        if "release_ts" in df.columns:
            df = df[pd.to_datetime(df["release_ts"]) <= decision_ts]
        col = "type" if "type" in df.columns else ("event_type" if "event_type" in df.columns else None)
        if col and "symbol" in df.columns and "date" in df.columns and not df.empty:
            sub = df[(df["symbol"].astype(str) == symbol) & (df[col].astype(str).str.contains("result", case=False))]
            ed = pd.to_datetime(sub["date"], errors="coerce").dt.date.dropna()
            future = sorted(d for d in ed if d > day)
            past = sorted(d for d in ed if d <= day)
            if future:
                out["days_to_earnings"] = float((future[0] - day).days); out["days_to_earnings_present"] = 1.0
            if past:
                out["days_since_earnings"] = float((day - past[-1]).days); out["days_since_earnings_present"] = 1.0
    # monsoon departure (latest weekly reading, PIT)
    df = _cached_csv(daily_path("reference_path") / "monsoon.csv")
    if df is not None:
        if "release_ts" in df.columns:
            df = df[pd.to_datetime(df["release_ts"]) <= decision_ts]
        if "departure_pct" in df.columns and not df.empty:
            out["monsoon_departure"] = float(df.sort_values("date")["departure_pct"].iloc[-1])
            out["monsoon_departure_present"] = 1.0
    return out


def _sentiment_row(symbol: str, day: date, decision_ts: pd.Timestamp) -> dict:
    out = {f: np.nan for f in SENTIMENT_FEATURES}
    out.update({f + "_present": 0.0 for f in SENTIMENT_FEATURES})
    # Phase 5 — wired once the Track B forward archive has accrued. Until then,
    # sentiment_enabled stays false and these remain present=0 (no lookahead risk).
    if not load_daily_config()["features"].get("sentiment_enabled"):
        return out
    from src.daily.sentiment.features import sentiment_features  # local import (optional pkg)
    return sentiment_features(symbol, day, decision_ts, out)


def _fii_5d_z(decision_ts: pd.Timestamp, day: date) -> tuple[float, float]:
    """Market-wide FII 5-day net z (same for all symbols). Strictly before `day`
    (B-4). Returns (value, present). present=0 when history is too short (the known
    data gap: fii_dii.parquet currently holds ~1 row)."""
    df = _cached_parquet(daily_path("bhavcopy_path") / "fii_dii.parquet")
    if df is None or "date" not in df.columns or "fii_net_cr" not in df.columns:
        return np.nan, 0.0
    # PIT: strictly before `day` (B-4) AND, if a publication time is recorded,
    # only rows released by decision_ts (mirrors _read_global / _macro_row).
    if "release_ts" in df.columns:
        df = df[pd.to_datetime(df["release_ts"]) <= decision_ts]
    df = df[pd.to_datetime(df["date"]) < pd.Timestamp(day)].sort_values("date")
    s = df["fii_net_cr"].tail(20)
    if len(s) < 5:
        return np.nan, 0.0
    std = float(s.std())
    if std <= 0:
        return np.nan, 0.0
    return float((s.tail(5).mean() - s.mean()) / std), 1.0


# ── the per-day feature row ───────────────────────────────────────────

def features_for_symbol(symbol: str, days, df_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for a symbol across many decision days in ONE vectorized pass
    (the market/F&O frame is computed once; optional blocks read cached files per
    day). Returns rows only for requested days that clear warm-up. This is the hot
    path for dataset assembly; features_for_day is the single-day wrapper."""
    cfg = load_daily_config()
    fcfg = cfg["features"]
    # warm-up must cover the longest trailing window so market features are never NaN
    warmup = max(cfg["data"]["min_warmup_days"],
                 max(fcfg["ma_windows"]) + 1, max(fcfg["return_windows"]) + 1,
                 max(fcfg["vol_windows"]) + 1)
    fno = fcfg.get("fno_enabled")
    df = load_symbol_daily(symbol, adjusted=True, df_panel=df_daily)
    if df.empty:
        return pd.DataFrame()
    frame = market_fno_frame(df)
    pos = {ts.date(): i for i, ts in enumerate(df.index)}
    want = set(days)

    rows = []
    for d in sorted(want):
        i = pos.get(d)
        if i is None or i + 1 < warmup:
            continue
        decision_ts = pd.Timestamp(datetime.combine(d, time(15, 30)))
        mrow = frame.iloc[i]
        row: dict = {"symbol": symbol, "date": pd.Timestamp(d)}
        for c in MARKET_FEATURES + FNO_FEATURES:
            row[c] = mrow[c]
        for f in FNO_FEATURES:
            row[f + "_present"] = mrow[f + "_present"]
        if fno:
            fz, fp = _fii_5d_z(decision_ts, d)
            row["fii_5d_z"] = fz
            row["fii_5d_z_present"] = fp
        row.update(_global_row(decision_ts))
        row.update(_macro_row(symbol, d, decision_ts))
        row.update(_sentiment_row(symbol, d, decision_ts))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[["symbol", "date"] + FEATURE_ORDER]


def features_for_day(symbol: str, day: date, df_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """One feature row for (symbol, decision day T) — thin wrapper over
    features_for_symbol so serve == train by construction."""
    return features_for_symbol(symbol, [day], df_daily=df_daily)


def impute_optional(df: pd.DataFrame) -> pd.DataFrame:
    """Optional NaN → 0.0, paired with the (already-0) _present flag (B-5)."""
    out = df.copy()
    out[OPTIONAL_FEATURES] = out[OPTIONAL_FEATURES].fillna(0.0)
    return out


def present_share(df: pd.DataFrame) -> dict[str, float]:
    """Mean of each block's _present flags — the real-data share, logged per build."""
    if df.empty:
        return {}
    blocks = {"fno": FNO_FEATURES, "global": GLOBAL_FEATURES,
              "macro": MACRO_FEATURES, "sentiment": SENTIMENT_FEATURES}
    return {k: float(df[[f + "_present" for f in feats]].mean(axis=1).mean())
            for k, feats in blocks.items()}


def build_matrix(plan: pd.DataFrame, df_panels: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Feature matrix for a plan of rows [date, symbol]. df_panels optionally maps
    symbol → injected daily panel (live/test). Row skips counted; over-tolerance
    RAISES (M-1)."""
    cfg = load_daily_config()
    frames, failed = [], 0
    for sym, g in plan.groupby("symbol"):
        try:
            inj = (df_panels or {}).get(sym)
            days = pd.to_datetime(g["date"]).dt.date.tolist()
            f = features_for_symbol(sym, days, df_daily=inj)
            if not f.empty:
                frames.append(f)
        except Exception as e:  # noqa: BLE001 — counted; raises over tolerance below
            failed += 1
            logger.warning("daily features: %s skipped (%s)", sym, e)
    if failed and failed > len(plan) * cfg["data"]["max_skip_share"]:
        raise DatasetThinningError(
            f"{failed}/{len(plan)} symbol-days failed feature build "
            f"(> {cfg['data']['max_skip_share']:.0%}) — investigate the daily panel"
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    n0 = len(out)
    out = out.dropna(subset=MARKET_FEATURES)
    dropped = n0 - len(out)
    if dropped and dropped > n0 * cfg["data"]["max_skip_share"]:
        raise DatasetThinningError(
            f"{dropped}/{n0} rows dropped for missing market features "
            f"(> {cfg['data']['max_skip_share']:.0%})"
        )
    logger.info("daily features: %d rows, present_share=%s", len(out), present_share(out))
    return impute_optional(out)
