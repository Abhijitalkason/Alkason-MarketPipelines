"""09:30 morning momentum screen (PLAN_v3 §7; v3_supplementry §5).

score = w1·z(first-15min volume vs 20d same-bar avg)
      + w2·z(first-15min range  vs 20d same-bar avg)
      + w3·gap_quality + w4·preopen_imbalance·direction + w5·delivery_z
      + w6·index_alignment

Liquidity floor (H-1): symbols below universe.min_median_1min_volume on the
trailing 20 sessions are dropped before scoring; where recorded depth exists,
symbols above max_median_spread_bps are dropped (no depth → leg skipped and
counted in the day's journal line).

Point-in-time: universe is load_universe(as_of=day); pre-open is PIT-filtered
on capture_ts; bhavcopy reads are strictly-prior files; bars for `day` come
from live_bars injection (paper) or official stored candles (replay).

Weights: per-fold logistic fit via fit_weights (the screen is a model too —
same leakage discipline). Live loads models/v3/screen_weights.json when the
trainer has produced one; config weights are the cold-start default.

index_alignment NOTE (recorded deviation from PLAN_v3 §7): stock-vs-sector-vs-
NIFTY agreement needs index 1-min bars which the bar store does not yet carry;
until then the universe-median first-15 direction is the market proxy.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config, load_universe, point_in_time
from src.intraday.bars import load_1min

logger = logging.getLogger(__name__)

FIRST_BAR = time(9, 15)
SCORE_COMPONENTS = ["vol_z", "rng_z", "gap_quality", "imbalance_aligned",
                    "delivery_z", "index_alignment"]
WEIGHT_KEYS = ["volume_surge", "range_expansion", "gap_quality",
               "preopen_imbalance", "delivery_z", "index_alignment"]


class ScreenError(RuntimeError):
    pass


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
        raise ScreenError(f"only {len(vols)} prior sessions for screen stats (need ≥10)")
    return float(np.mean(vols)), float(np.std(vols) or 1.0), float(np.mean(rngs)), float(np.std(rngs) or 1e-9)


def _delivery_z(symbol: str, day: date, lookback: int) -> float | None:
    """Prior-day delivery % z-score vs its own trailing distribution.
    None (not 0) when files are absent — missing data is visible, not neutral."""
    bdir = Path(load_config()["data"]["bhavcopy_path"])
    bdir = bdir if bdir.is_absolute() else ROOT / bdir
    vals = []
    for d in pd.bdate_range(end=day - timedelta(days=1), periods=lookback).date:
        p = bdir / f"eq_{d.isoformat()}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["symbol", "deliv_per"])
            row = df[df.symbol == symbol]
            if not row.empty and pd.notna(row["deliv_per"].iloc[0]):
                vals.append(float(row["deliv_per"].iloc[0]))
    if len(vals) < 5:
        return None
    s = pd.Series(vals)
    std = float(s.std())
    return float((s.iloc[-1] - s.mean()) / std) if std > 0 else 0.0


def _preopen_imbalance(symbol: str, day: date) -> float | None:
    p = Path(load_config()["data"]["preopen_path"])
    p = p if p.is_absolute() else ROOT / p
    p = p / f"{day.isoformat()}.parquet"
    if not p.exists():
        return None
    decision_floor = pd.Timestamp(datetime.combine(day, time(9, 30)))
    df = point_in_time(pd.read_parquet(p), day, decision_floor, date_col="__none__")
    row = df[df.symbol == symbol]
    if row.empty or pd.isna(row["imbalance"].iloc[0]):
        return None
    return float(row["imbalance"].iloc[0])


def _median_spread_bps(symbol: str, day: date, lookback: int = 5) -> float | None:
    """Median quoted spread (bps) from recorded depth over recent prior days.
    None when no depth recorded for the symbol (the leg is then skipped+counted)."""
    base = Path(load_config()["data"]["depth_path"])
    base = base if base.is_absolute() else ROOT / base
    spreads = []
    for d in pd.bdate_range(end=day - timedelta(days=1), periods=lookback).date:
        ddir = base / d.isoformat()
        if not ddir.exists():
            continue
        for f in ddir.glob("*.parquet"):
            df = pd.read_parquet(f, columns=["symbol", "bid", "ask", "ltp"])
            row = df[(df.symbol == symbol) & df.bid.notna() & df.ask.notna() & (df.ltp > 0)]
            if not row.empty:
                spreads.extend(((row["ask"] - row["bid"]) / row["ltp"] * 1e4).tolist())
    if not spreads:
        return None
    return float(np.median(spreads))


def _load_fitted_weights() -> dict | None:
    p = ROOT / load_config()["paths"]["state"] / "screen_weights.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def clipped_components(df: pd.DataFrame) -> pd.DataFrame:
    """The ONE component transform — used by both fit_weights (train) and
    score_rows (serve), so the screen weights are learned on exactly the values
    they are later applied to (no train/serve transform skew, PLAN_v3 §7)."""
    return pd.DataFrame({
        "vol_z": df["vol_z"].clip(-3, 3),
        "rng_z": df["rng_z"].clip(-3, 3),
        "gap_quality": df["gap_quality"],
        "imbalance_aligned": df["imbalance_aligned"].clip(-1, 1),
        "delivery_z": df["delivery_z"].clip(-3, 3),
        "index_alignment": df["index_alignment"],
    }, index=df.index)[SCORE_COMPONENTS]


def score_rows(df: pd.DataFrame, w: dict) -> pd.Series:
    """Score from the component columns — one formula for live and refit ranking."""
    c = clipped_components(df)
    return (
        w["volume_surge"] * c["vol_z"]
        + w["range_expansion"] * c["rng_z"]
        + w["gap_quality"] * c["gap_quality"]
        + w["preopen_imbalance"] * c["imbalance_aligned"]
        + w["delivery_z"] * c["delivery_z"]
        + w["index_alignment"] * c["index_alignment"]
    )


def screen_day(day: date, symbols: list[str] | None = None,
               weights: dict | None = None,
               live_bars: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Rank the PIT universe at 09:30 on `day`. Returns top_n rows with the
    score, direction hint, and all six score components (for per-fold refits).

    live_bars: {symbol: 1-min frame incl. today} from the recorder (B-2) —
    without it, today's bars come from the store (replay/backtest path).
    """
    cfg = load_config()
    sc = cfg["screen"]
    uni = cfg["universe"]
    w = weights or _load_fitted_weights() or sc["weights"]
    lookback = sc["lookback_days"]
    gap_lo, gap_hi = sc["gap_band"]
    symbols = symbols if symbols is not None else load_universe(as_of=day)["symbol"].tolist()

    rows = []
    skipped: dict[str, int] = {"bars": 0, "history": 0, "liquidity": 0, "spread": 0,
                               "spread_unchecked": 0, "no_delivery": 0, "no_preopen": 0}
    for sym in symbols:
        if live_bars is not None and sym in live_bars:
            df = live_bars[sym]
        else:
            try:
                df = load_1min(sym, day - timedelta(days=lookback * 2), day)
            except Exception as e:  # noqa: BLE001 — counted; day raises if all skipped
                skipped["bars"] += 1
                logger.warning("screen: %s skipped (%s)", sym, e)
                continue
        f15 = _first15(df, day)
        if f15 is None:
            skipped["bars"] += 1
            continue
        prior = df[df.index.date < day]
        prior_days = sorted(set(prior.index.date))
        if not prior_days:
            skipped["history"] += 1
            continue

        # ── liquidity floor (H-1) ─────────────────────────────────────
        recent = prior[prior.index.date >= prior_days[max(0, len(prior_days) - 20)]]
        if float(recent["volume"].median()) < uni["min_median_1min_volume"]:
            skipped["liquidity"] += 1
            continue
        spread = _median_spread_bps(sym, day)
        if spread is None:
            skipped["spread_unchecked"] += 1
        elif spread > uni["max_median_spread_bps"]:
            skipped["spread"] += 1
            continue

        prev_close = float(prior[prior.index.date == prior_days[-1]]["close"].iloc[-1])
        try:
            vmean, vstd, rmean, rstd = _history_stats(df, day, lookback)
        except ScreenError as e:
            skipped["history"] += 1
            logger.warning("screen: %s skipped (%s)", sym, e)
            continue

        gap = (f15["open"] - prev_close) / prev_close
        f15_ret = (f15["close"] - f15["open"]) / f15["open"]
        direction = int(np.sign(gap + f15_ret)) or 1
        imb = _preopen_imbalance(sym, day)
        if imb is None:
            skipped["no_preopen"] += 1
            imb_aligned = 0.0
        else:
            imb_aligned = imb * direction      # signed imbalance, credited when it agrees
        dz = _delivery_z(sym, day, lookback)
        if dz is None:
            skipped["no_delivery"] += 1
            dz = 0.0

        rows.append({
            "symbol": sym, "gap": gap, "f15_ret": f15_ret, "direction": direction,
            "vol_z": (f15["volume"] - vmean) / vstd,
            "rng_z": ((f15["high"] - f15["low"]) / f15["open"] - rmean) / rstd,
            "gap_quality": 1.0 if gap_lo <= abs(gap) <= gap_hi else 0.0,
            "imbalance_aligned": imb_aligned,
            "delivery_z": dz,
        })
    if not rows:
        raise ScreenError(f"screen produced 0 candidates on {day} — check bar store ({skipped})")
    df = pd.DataFrame(rows)

    # index alignment: agreement of stock first-15 direction with universe median
    # (proxy — see module docstring)
    mkt_dir = np.sign(df["f15_ret"].median())
    df["index_alignment"] = (np.sign(df["f15_ret"]) == mkt_dir).astype(float)

    df["score"] = score_rows(df, w)
    top = df.sort_values("score", ascending=False).head(sc["top_n"]).reset_index(drop=True)
    logger.info("screen %s: top=%s skipped=%s", day, top.symbol.tolist(),
                {k: v for k, v in skipped.items() if v})
    top.attrs["skipped"] = skipped
    return top


def fit_weights(screen_rows: pd.DataFrame, day_won: pd.Series) -> dict:
    """Per-fold screen-weight fit (PLAN_v3 §7 / H-8): L2 logistic regression of
    'this (day, symbol) produced ≥1 triple-barrier win' on the six score
    components. Returns a weight dict in config key order; signs are logged so
    a wrong-signed component is visible, never silent."""
    from sklearn.linear_model import LogisticRegression

    X = clipped_components(screen_rows).to_numpy(dtype=float)   # same transform as score_rows
    y = day_won.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ScreenError("fit_weights: outcomes are single-class — cannot fit screen weights")
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, y)
    coefs = lr.coef_[0]
    scale = float(np.sum(np.abs(coefs))) or 1.0
    w = {k: float(c / scale) for k, c in zip(WEIGHT_KEYS, coefs)}
    logger.info("screen weights fit on %d rows: %s", len(y), w)
    return w


def save_weights(w: dict) -> Path:
    p = ROOT / load_config()["paths"]["state"] / "screen_weights.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(w, indent=2))
    return p
