"""Multi-horizon triple-barrier labeling on the daily series (one labeler; a
horizon is a config block, per the plan). Mirrors src/intraday/labeler.py
geometry/TradeOutcome, generalized to daily bars and an N-day time barrier.

For a decision day T and horizon H:
  entry  = next session OPEN (never the close you just observed — no lookahead)
  target = entry · (1 + target_atr · atr_pct_T)      (long perspective)
  stop   = entry · (1 − stop_atr  · atr_pct_T)
  t_bar  = close of the H-th session after entry
  label  = 1 target-first | 0 stop-first | int(pnl>0) at the time barrier
Labels are computed LONG-only and direction-agnostic; the backtester applies the
predicted direction sign. Conservative tie-break: a daily bar touching BOTH
barriers counts as the stop (pessimistic, matches intraday).

The primary horizon is swing_1_5d (H=5); a horizon is fully defined by its
config block (target/stop ATR + time_barrier_days), so no code changes per horizon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.daily import horizon_config
from src.daily.features import _atr_pct
from src.daily.panel import load_symbol_daily

logger = logging.getLogger(__name__)


@dataclass
class DailyOutcome:
    symbol: str
    decision_day: date
    entry_ts: pd.Timestamp
    entry: float
    target: float
    stop: float
    exit_ts: pd.Timestamp
    exit_price: float
    barrier: str            # 'target' | 'stop' | 'time'
    label: int              # 1 up / 0 down (long perspective)
    pnl_pct: float          # signed long return, pre-cost


def _resolve_daily_path(path: pd.DataFrame, entry: float, target: float,
                        stop: float) -> tuple[pd.Timestamp, float, str]:
    """Scan daily bars for first barrier touch (long). Conservative on a bar that
    spans both barriers (counts as stop)."""
    for ts, bar in path.iterrows():
        hit_t, hit_s = bar.high >= target, bar.low <= stop
        if hit_s:                      # dual-touch or stop-only → pessimistic
            return ts, stop, "stop"
        if hit_t:
            return ts, target, "target"
    last = path.iloc[-1]
    return path.index[-1], float(last.close), "time"


def label_day(symbol: str, day: date, horizon: str = "swing_1_5d",
              df_daily: pd.DataFrame | None = None) -> DailyOutcome | None:
    """Triple-barrier outcome for one (symbol, decision day) — thin wrapper over
    label_symbol. Returns None when there is no entry bar / insufficient history."""
    return label_symbol(symbol, [day], horizon, df_daily=df_daily).get(day)


def label_symbol(symbol: str, days, horizon: str = "swing_1_5d",
                 df_daily: pd.DataFrame | None = None) -> dict:
    """Triple-barrier outcomes for many decision days in one pass (ATR computed
    once over the full series; trailing, so no lookahead). Returns {day: outcome}.
    The hot path for dataset assembly; label_day is the single-day wrapper."""
    hc = horizon_config(horizon)
    if hc["path"] != "daily_close":
        raise NotImplementedError(
            f"horizon {horizon!r} path {hc['path']!r} is built in Phase 4 (intraday_open)"
        )
    h_days = int(hc["time_barrier_days"])
    df = load_symbol_daily(symbol, adjusted=True, df_panel=df_daily)
    if len(df) < 21:
        return {}
    atr = _atr_pct(df, 14)
    pos = {ts.date(): i for i, ts in enumerate(df.index)}
    out: dict = {}
    for d in days:
        i = pos.get(d)
        if i is None or i < 20 or i + 1 >= len(df):
            continue
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue
        entry = float(df.iloc[i + 1]["open"])
        entry_ts = df.index[i + 1]
        target = entry * (1 + hc["target_atr"] * a)
        stop = entry * (1 - hc["stop_atr"] * a)
        path = df.iloc[i + 1:i + 1 + h_days]
        if path.empty:
            continue
        exit_ts, exit_price, barrier = _resolve_daily_path(path, entry, target, stop)
        pnl_pct = (exit_price - entry) / entry
        label = int(pnl_pct > 0) if barrier == "time" else int(barrier == "target")
        out[d] = DailyOutcome(symbol=symbol, decision_day=d, entry_ts=entry_ts, entry=entry,
                              target=target, stop=stop, exit_ts=exit_ts,
                              exit_price=float(exit_price), barrier=barrier, label=label,
                              pnl_pct=float(pnl_pct))
    return out


def outcomes_frame(outcomes: list[DailyOutcome]) -> pd.DataFrame:
    return pd.DataFrame([o.__dict__ for o in outcomes if o is not None])


def geometric_baseline(horizon: str = "swing_1_5d") -> float:
    """stop_atr / (target_atr + stop_atr) — the no-edge win-rate floor for the
    horizon's geometry (symmetric ⇒ 0.5)."""
    hc = horizon_config(horizon)
    return hc["stop_atr"] / (hc["target_atr"] + hc["stop_atr"])
