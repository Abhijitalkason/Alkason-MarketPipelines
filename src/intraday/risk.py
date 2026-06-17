"""Risk layer (PLAN_v3 §13; v3_supplementry §11) — sizing, limits, blackout
calendar, kill switches.

Exists because at 0.4/3.2 geometry one full stop erases ~8 winners. High-win-
rate systems die by tails; every control here is load-bearing AND has a live
call site (the paper runner) plus a test that fails if the call site is removed
— no dead controls (v3_supplementry §0 non-negotiable #2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd

from src.intraday import ROOT, load_config, load_universe, now_ist, today_ist

logger = logging.getLogger(__name__)


# ── event blackout ────────────────────────────────────────────────────

def _events() -> pd.DataFrame:
    p = ROOT / load_config()["data"]["reference_path"] / "events.csv"
    if not p.exists():
        raise FileNotFoundError("events.csv missing — blackout calendar is mandatory (Section 13)")
    df = pd.read_csv(p, comment="#")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].fillna("")
    return df


def is_blackout(day: date, symbol: str | None = None) -> bool:
    """True if no signals may be emitted for `day` (market-wide or symbol
    results day ±1). Weekly index expiry auto-flagged from config weekday."""
    cfg = load_config()["risk"]
    if day.weekday() == cfg["expiry_weekday"]:        # weekly index expiry
        return True
    ev = _events()
    market_wide = ev[(ev.symbol == "") & (ev.date == day)]
    if not market_wide.empty:
        return True
    if symbol:
        sym_ev = ev[(ev.symbol == symbol) & (ev.type == "results")]
        for d in sym_ev.date:
            if abs((day - d).days) <= 1:
                return True
    return False


# ── position sizing ───────────────────────────────────────────────────

def position_size(capital_inr: float, entry: float, stop: float) -> int:
    """Fixed ₹-risk sizing: every full stop costs capital × risk_per_trade_pct.
    Leverage-capped at risk.max_leverage of capital per position."""
    cfg = load_config()["risk"]
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        raise ValueError("stop distance must be positive")
    qty = int((capital_inr * cfg["risk_per_trade_pct"]) / stop_dist)
    max_qty = int((capital_inr * cfg["max_leverage"]) / entry)
    return max(0, min(qty, max_qty))


# ── intra-day limits (persisted in the session state file) ────────────

@dataclass
class DayState:
    capital_inr: float
    pnl_inr: float = 0.0
    open_positions: dict = field(default_factory=dict)     # symbol → sector
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DayState":
        return cls(**d)


class RiskManager:
    def __init__(self, capital_inr: float, state: DayState | None = None,
                 as_of: date | None = None):
        self.cfg = load_config()["risk"]
        self.state = state or DayState(capital_inr=capital_inr)
        uni = load_universe(as_of=as_of or today_ist())
        self._sectors = dict(zip(uni["symbol"], uni["sector"]))

    def can_open(self, symbol: str, day: date) -> tuple[bool, str]:
        s = self.state
        if s.halted:
            return False, s.halt_reason
        if is_blackout(day, symbol):
            return False, "blackout"
        if len(s.open_positions) >= self.cfg["max_positions"]:
            return False, "max_positions"
        sector = self._sectors.get(symbol, "UNKNOWN")
        if list(s.open_positions.values()).count(sector) >= self.cfg["max_per_sector"]:
            return False, f"max_per_sector:{sector}"
        return True, "ok"

    def on_open(self, symbol: str) -> None:
        self.state.open_positions[symbol] = self._sectors.get(symbol, "UNKNOWN")

    def on_close(self, symbol: str, pnl_inr: float) -> None:
        self.state.open_positions.pop(symbol, None)
        self.state.pnl_inr += pnl_inr
        if self.state.pnl_inr <= -self.cfg["daily_loss_limit_pct"] * self.state.capital_inr:
            self.state.halted = True
            self.state.halt_reason = "daily_loss_limit"
            logger.warning("DAILY LOSS LIMIT HIT (%.0f INR) — halted for the day", self.state.pnl_inr)


# ── kill switches (Section 13) — evaluated post-market ────────────────

def check_kill_switches(trades: pd.DataFrame, backtest_win_rate: float) -> list[str]:
    """Two switches over the live/paper trade log [date, label, pnl_pct_net]:
      (a) mean net expectancy over the last 20 sessions' trades < 0
      (b) last 30 trades' win rate < backtest win rate − 0.10
    Breach → _persist_halt → next session refuses to start. Callers: paper
    runner post-market and gates.py (no dead control)."""
    breaches = []
    if trades.empty:
        return breaches
    trades = trades.sort_values("date")
    daily = trades.groupby("date")["pnl_pct_net"].mean()
    if len(daily) >= 20:
        recent_days = set(daily.tail(20).index)
        recent = trades[trades.date.isin(recent_days)]
        if recent["pnl_pct_net"].mean() < 0:
            breaches.append("rolling_20d_expectancy_negative")
    last30 = trades.tail(30)
    if len(last30) == 30 and last30["label"].mean() < backtest_win_rate - 0.10:
        breaches.append("live_win_rate_10pts_below_backtest")
    if breaches:
        _persist_halt(breaches)
    return breaches


def _persist_halt(breaches: list[str]) -> None:
    p = ROOT / load_config()["paths"]["state"] / "halt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"halted_at": now_ist().isoformat(), "breaches": breaches}, indent=2))
    logger.error("KILL SWITCH: %s — system halted (delete models/v3/halt.json after review)", breaches)
    try:
        from src.intraday.journal import journal_write
        journal_write("halt", {"breaches": breaches})
    except Exception:  # noqa: BLE001 — never let journaling failure mask the halt
        pass


def is_halted() -> bool:
    return (ROOT / load_config()["paths"]["state"] / "halt.json").exists()


def halt_reason() -> list[str]:
    p = ROOT / load_config()["paths"]["state"] / "halt.json"
    return json.loads(p.read_text())["breaches"] if p.exists() else []


def clear_halt() -> None:
    p = ROOT / load_config()["paths"]["state"] / "halt.json"
    if p.exists():
        p.unlink()
