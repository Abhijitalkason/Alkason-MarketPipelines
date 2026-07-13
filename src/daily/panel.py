"""Daily point-in-time panel — assemble per-symbol daily series from the NSE EOD
files already on disk (data/bhavcopy/eq_*.parquet + fo_*.parquet).

The eq/fo bhavcopies are already DAILY and already carry capture_ts, so this is a
join/pivot, not a fetch. NSE bhavcopy prices are AS-TRADED; we store them raw and
keep a corporate-action `adj_factor` column (reusing src.intraday.corporate_actions
factors). load_symbol_daily(adjusted=True) applies it on load — storage is never
mutated, exactly like the intraday bar store (H-2).

PIT contract: a daily row's content date is the trading day; its EOD values become
public after the close, so a row dated D is usable for any decision at/after D's
close. The no-lookahead feature test relies on features reading only rows with
date <= decision day, which this panel preserves.

Adjustment note (honest caveat): adjusted prices use the full factor table, the
standard backtest convention. A split's factor is mechanical (not predictive), so
the negligible "future factor" leak does not create edge; using adjusted returns
across an ex-date is in fact MORE correct than as-traded.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.daily import daily_path, load_daily_config, load_universe, now_ist
from src.intraday.corporate_actions import load_factors

logger = logging.getLogger(__name__)

# Per-symbol daily panel schema (raw, as-traded prices + fo fields + adj_factor).
PANEL_COLUMNS = [
    "date", "open", "high", "low", "close", "prev_close", "volume",
    "turnover_lacs", "deliv_per", "oi", "oi_change", "basis_pct",
    "fut_close", "spot_close", "adj_factor", "capture_ts",
]

_EQ_RENAME = {
    "open_price": "open", "high_price": "high", "low_price": "low",
    "close_price": "close", "ttl_trd_qnty": "volume",
}


class PanelError(RuntimeError):
    pass


def _bhav_dir() -> Path:
    return daily_path("bhavcopy_path")


def _out_dir() -> Path:
    d = daily_path("daily_path")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_eq(d: date) -> pd.DataFrame | None:
    p = _bhav_dir() / f"eq_{d.isoformat()}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.rename(columns=_EQ_RENAME)
    keep = ["symbol", "open", "high", "low", "close", "prev_close", "volume",
            "turnover_lacs", "deliv_per", "capture_ts"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.Timestamp(d)
    return df


def _read_fo(d: date) -> pd.DataFrame | None:
    p = _bhav_dir() / f"fo_{d.isoformat()}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    keep = ["symbol", "oi", "oi_change", "basis_pct", "fut_close", "spot_close"]
    return df[[c for c in keep if c in df.columns]].copy()


def _adj_factor(symbol: str, dates: pd.Series) -> pd.Series:
    """Cumulative back-adjustment factor per date: prod(factor for ex_date > date).
    Multiply as-traded price by this to get the adjusted (continuous) series."""
    factors = load_factors(symbol)
    out = pd.Series(1.0, index=dates.index)
    for ex_date, factor in factors:
        out *= pd.Series([factor if d.date() < ex_date else 1.0 for d in dates], index=dates.index)
    return out


def build_panel(start: date, end: date, symbols: list[str] | None = None) -> dict[str, int]:
    """Build/refresh per-symbol daily parquet for every symbol in the PIT universe
    (or `symbols`) over [start, end]. Returns {symbol: rows_written}.

    Universe is point-in-time (load_universe(as_of=end)); pass `symbols` to override
    (e.g. the full F&O list). Missing trading days are simply absent (a file exists
    iff the market traded), consistent with the intraday calendar derivation.
    """
    if symbols is None:
        symbols = load_universe(as_of=end)["symbol"].tolist()
    want = set(symbols)

    days = pd.bdate_range(start, end).date
    eq_frames, fo_frames = [], []
    for d in days:
        eq = _read_eq(d)
        if eq is not None:
            eq_frames.append(eq[eq["symbol"].isin(want)])
        fo = _read_fo(d)
        if fo is not None:
            fo = fo[fo["symbol"].isin(want)].copy()
            fo["date"] = pd.Timestamp(d)
            fo_frames.append(fo)
    if not eq_frames:
        raise PanelError(
            f"no eq_*.parquet found in {_bhav_dir()} for {start}..{end} — "
            f"run the intraday bhavcopy backfill first"
        )
    eq_all = pd.concat(eq_frames, ignore_index=True)
    fo_all = pd.concat(fo_frames, ignore_index=True) if fo_frames else pd.DataFrame(
        columns=["symbol", "date", "oi", "oi_change", "basis_pct", "fut_close", "spot_close"]
    )

    written: dict[str, int] = {}
    for sym, g in eq_all.groupby("symbol"):
        g = g.sort_values("date").drop_duplicates(subset="date", keep="last")
        fo_g = fo_all[fo_all["symbol"] == sym].drop(columns="symbol")
        merged = g.merge(fo_g, on="date", how="left")
        merged["adj_factor"] = _adj_factor(sym, merged["date"]).to_numpy()
        for col in PANEL_COLUMNS:
            if col not in merged.columns:
                merged[col] = pd.NA
        out = merged[PANEL_COLUMNS].sort_values("date").reset_index(drop=True)
        path = _out_dir() / f"{sym}.parquet"
        out.to_parquet(path, index=False)
        written[sym] = len(out)
    logger.info("daily panel: %d symbols written to %s (%s..%s)",
                len(written), _out_dir(), start, end)
    return written


def load_symbol_daily(symbol: str, start: date | None = None, end: date | None = None,
                      adjusted: bool = True, df_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-symbol daily frame indexed by date (tz-naive Timestamp).

    df_panel injection is the live/test contract (mirrors features_for_day's
    df_1min): pass a frame to bypass disk. adjusted=True applies adj_factor to
    OHLC (×factor) and volume (÷factor) — the continuous series for returns/MA.
    Raw prices stay available via adjusted=False.
    """
    if df_panel is None:
        path = _out_dir() / f"{symbol}.parquet"
        if not path.exists():
            raise PanelError(f"{path} missing — run build_panel first")
        df = pd.read_parquet(path)
    else:
        df = df_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.sort_values("date").set_index("date")
    if adjusted and "adj_factor" in df.columns:
        f = df["adj_factor"].fillna(1.0).astype(float)
        for col in ("open", "high", "low", "close", "prev_close", "fut_close", "spot_close"):
            if col in df.columns:
                df[col] = df[col].astype(float) * f
        if "volume" in df.columns:
            df["volume"] = (df["volume"].astype(float) / f.replace(0, 1.0))
    return df


def trading_days(start: date, end: date) -> list[date]:
    """NSE trading days in [start, end].

    Primary source: eq bhavcopy presence (holiday-free by construction — a file
    exists iff the market traded). Fallback (tests/live with a built panel but no
    bhavcopy on disk): the union of dates across the per-symbol daily panels.
    Both are holiday-free."""
    base = _bhav_dir()
    days = [d for d in pd.bdate_range(start, end).date
            if (base / f"eq_{d.isoformat()}.parquet").exists()]
    if days:
        return days
    have: set[date] = set()
    out = daily_path("daily_path")
    if out.exists():
        for pq in out.glob("*.parquet"):
            ds = pd.to_datetime(pd.read_parquet(pq, columns=["date"])["date"]).dt.date
            have.update(d for d in ds if start <= d <= end)
    return sorted(have)


def panel_capture_ts(symbol: str) -> pd.Timestamp:
    """Most recent capture_ts in a symbol's panel (PIT bookkeeping for live runs)."""
    df = load_symbol_daily(symbol, adjusted=False)
    return pd.to_datetime(df["capture_ts"]).max() if "capture_ts" in df.columns else now_ist()
