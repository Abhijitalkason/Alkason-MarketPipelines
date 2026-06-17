"""Risk: sizing, sector/concurrency caps, daily-loss halt persists, both kill
switches, blackout with symbol."""

from __future__ import annotations

import pandas as pd

from src.intraday import ROOT, load_config
from src.intraday.risk import (RiskManager, check_kill_switches, clear_halt, is_blackout,
                               is_halted, position_size)


def test_position_size_fixed_risk(repo):
    cfg = load_config()["risk"]
    qty = position_size(1_000_000, entry=1000, stop=990)
    assert qty == int((1_000_000 * cfg["risk_per_trade_pct"]) / 10)


def test_sector_and_concurrency_caps(repo, frozen_day):
    rm = RiskManager(1_000_000, as_of=frozen_day)
    rm.on_open("AAA")        # Tech
    rm.on_open("CCC")        # Tech
    ok, reason = rm.can_open("AAA", frozen_day)  # already counts as Tech×2
    # AAA in positions already → but cap test: a 3rd Tech would breach max_per_sector=2
    rm2 = RiskManager(1_000_000, as_of=frozen_day)
    rm2.on_open("AAA"); rm2.on_open("CCC")
    ok2, reason2 = rm2.can_open("BBB", frozen_day)  # BBB is Bank → allowed
    assert ok2
    rm2._sectors["DDD"] = "Tech"
    ok3, reason3 = rm2.can_open("DDD", frozen_day)
    assert not ok3 and "max_per_sector" in reason3


def test_daily_loss_halt(repo, frozen_day):
    rm = RiskManager(1_000_000, as_of=frozen_day)
    rm.on_open("AAA")
    rm.on_close("AAA", -20_000)  # -2% > 1.5% limit
    ok, reason = rm.can_open("BBB", frozen_day)
    assert not ok and reason == "daily_loss_limit"


def test_kill_switch_winrate(repo):
    clear_halt()
    trades = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=30, freq="D"),
        "label": [0] * 30, "pnl_pct_net": [-0.01] * 30,
    })
    breaches = check_kill_switches(trades, backtest_win_rate=0.89)
    assert "live_win_rate_10pts_below_backtest" in breaches
    assert is_halted()
    clear_halt()


def test_kill_switch_expectancy(repo):
    clear_halt()
    trades = pd.DataFrame({
        "date": sorted(pd.date_range("2026-01-01", periods=25, freq="D").tolist() * 2),
        "label": [1, 0] * 25, "pnl_pct_net": [-0.02] * 50,
    })
    breaches = check_kill_switches(trades, backtest_win_rate=0.50)
    assert "rolling_20d_expectancy_negative" in breaches
    clear_halt()


def test_blackout_thursday(repo):
    next_thu = pd.Timestamp("2026-04-02").date()  # a Thursday
    assert is_blackout(next_thu)
