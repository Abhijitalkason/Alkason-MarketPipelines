"""Triple-barrier labeling on 1-min paths (PLAN_v3 §8; v3_supplementry §4.5).

For each candidate (symbol, decision bar) in the entry window:
  entry  = next 15-min bar open (realistic fill)
  target = entry ± a·ATR(2h)   stop = entry ∓ b·ATR(2h)
  t_bar  = entry + 3h, capped at 14:45
  label  = 1 target-first | 0 stop-first | sign(PnL) at time barrier

Conservative tie-break: if one 1-min bar spans BOTH barriers → counted as stop.
Geometry (a, b) is read from config so geometry_study can sweep it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from src.intraday import load_config
from src.intraday.bars import atr_2h, load_1min, resample

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    symbol: str
    entry_ts: datetime
    direction: int          # +1 long, -1 short
    entry: float
    target: float
    stop: float
    exit_ts: datetime
    exit_price: float
    barrier: str            # 'target' | 'stop' | 'time'
    label: int              # 1 win, 0 loss
    pnl_pct: float          # signed, pre-cost


def _resolve_path(path: pd.DataFrame, direction: int, entry: float,
                  target: float, stop: float) -> tuple[pd.Timestamp, float, str]:
    """Scan a 1-min path for first barrier touch. Conservative on dual-touch bars."""
    for ts, bar in path.iterrows():
        if direction == 1:
            hit_t, hit_s = bar.high >= target, bar.low <= stop
        else:
            hit_t, hit_s = bar.low <= target, bar.high >= stop
        if hit_t and hit_s:
            return ts, stop, "stop"          # pessimistic by construction
        if hit_s:
            return ts, stop, "stop"
        if hit_t:
            return ts, target, "target"
    last = path.iloc[-1]
    return path.index[-1], last.close, "time"


def label_day(symbol: str, day: date, directions: dict[pd.Timestamp, int] | None = None,
              df_1min: pd.DataFrame | None = None,
              geometry: dict | None = None) -> list[TradeOutcome]:
    """Label every decision bar in the entry window for one symbol-day.

    directions: optional {decision_bar_ts: ±1} from the screener; default long.
    geometry:   optional {target_atr, stop_atr, ...} override for geometry_study;
                config geometry otherwise.
    """
    cfg = load_config()
    geo = geometry or cfg["geometry"]

    def _geo(key):
        # honor the geometry_study override dict for EVERY tuned key, config
        # fallback otherwise (PLAN_v4 §2 — the override was silently ignored for
        # squareoff/time_barrier/atr_window, breaking any grid that swept them).
        return geo[key] if key in geo else cfg["geometry"][key]

    if df_1min is None:
        # ATR needs ~5 prior sessions of bars
        df_1min = load_1min(symbol, day - timedelta(days=10), day)
    df5 = resample(df_1min, cfg["data"]["bar_freq_feature"])
    atr = atr_2h(df5, window_minutes=_geo("atr_window_minutes"))

    w_start = time.fromisoformat(_geo("entry_window")[0])
    w_end = time.fromisoformat(_geo("entry_window")[1])
    squareoff = time.fromisoformat(_geo("squareoff"))
    t_barrier_h = _geo("time_barrier_hours")
    df15 = resample(df_1min[df_1min.index.date == day], cfg["data"]["bar_freq_decision"])

    outcomes: list[TradeOutcome] = []
    decision_bars = df15[(df15.index.time >= w_start) & (df15.index.time < w_end)]
    for bar_ts in decision_bars.index:
        # decision at bar close; entry at NEXT 15-min bar open
        entry_ts = bar_ts + pd.Timedelta("15min")
        later = df15.index[df15.index > bar_ts]
        if len(later) == 0 or later[0] != entry_ts:
            continue
        entry = float(df15.loc[entry_ts, "open"])
        # ATR known as of decision time (last closed 5-min bar before entry)
        atr_known = atr[atr.index < entry_ts].dropna()
        if atr_known.empty:
            continue
        a = float(atr_known.iloc[-1])
        if a <= 0:
            continue
        direction = (directions or {}).get(bar_ts, 1)
        target = entry + direction * geo["target_atr"] * a
        stop = entry - direction * geo["stop_atr"] * a

        t_bar = min(
            entry_ts + pd.Timedelta(hours=t_barrier_h),
            pd.Timestamp(datetime.combine(day, squareoff)),
        )
        path = df_1min[(df_1min.index >= entry_ts) & (df_1min.index <= t_bar)]
        if path.empty:
            continue
        exit_ts, exit_price, barrier = _resolve_path(path, direction, entry, target, stop)
        pnl_pct = direction * (exit_price - entry) / entry
        outcomes.append(TradeOutcome(
            symbol=symbol, entry_ts=entry_ts.to_pydatetime(), direction=direction,
            entry=entry, target=target, stop=stop,
            exit_ts=pd.Timestamp(exit_ts).to_pydatetime(), exit_price=float(exit_price),
            barrier=barrier, label=int(pnl_pct > 0) if barrier == "time" else int(barrier == "target"),
            pnl_pct=float(pnl_pct),
        ))
    return outcomes


def outcomes_frame(outcomes: list[TradeOutcome]) -> pd.DataFrame:
    return pd.DataFrame([o.__dict__ for o in outcomes])


def geometric_baseline(geometry: dict | None = None) -> float:
    """b/(a+b) — the no-edge win-rate floor for the configured geometry."""
    geo = geometry or load_config()["geometry"]
    return geo["stop_atr"] / (geo["target_atr"] + geo["stop_atr"])
