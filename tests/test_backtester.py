"""Backtester: dataset assembly, day-boundary inner split, fit_fold OOF
discipline, end-to-end smoke run on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from datetime import date as _date

from src.intraday import load_config
from src.intraday.backtester import (LeakageAlarm, _finalize, _subsidy_breaches,
                                     _inner_split, build_dataset, fit_fold, make_folds,
                                     run_backtest)


def _trade(day, symbol, sector, label, p_cal=0.95, pnl=0.01, atr=0.01, ts_hour=9):
    """One trade row in the exact shape _finalize/_fairness_tables consume."""
    entry_ts = pd.Timestamp(f"{day}T{ts_hour:02d}:45:00")
    from src.intraday.backtester import _tod_bucket
    return {
        "fold": 1, "date": day, "symbol": symbol, "sector": sector,
        "entry_ts": entry_ts, "p_cal": p_cal, "label": int(label), "barrier": "target",
        "atr_pct": atr, "tod_bucket": _tod_bucket(entry_ts),
        "pnl_pct_gross": pnl, "cost_pct": 0.0005, "pnl_pct_net": pnl - 0.0005,
    }


# skip the first ~25 sessions (screen/feature warm-up needs ≥20 prior days) —
# in production history extends years back, so warm-up is never a real fraction.
def test_build_dataset_and_folds(repo):
    days = repo["days"][25:160]
    data = build_dataset(days)
    assert not data.empty and "label" in data.columns
    assert data["label"].isin([0, 1]).all()


def test_inner_split_day_boundary(repo):
    days = repo["days"][25:160]
    data = build_dataset(days)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    head, tail = _inner_split(data)
    head_days = set(pd.to_datetime(head["date"]).dt.date)
    tail_days = set(pd.to_datetime(tail["date"]).dt.date)
    assert head_days.isdisjoint(tail_days)  # no day straddles the boundary


def test_fit_fold_runs(repo):
    days = repo["days"][25:160]
    data = build_dataset(days)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    pm, fm, blender, record = fit_fold(data, tune=False)
    assert pm.model is not None
    assert blender.calibrator is not None
    assert "calib_brier" in record


def test_backtest_smoke(repo):
    """Full walk-forward on synthetic data emits a report with the metric
    contract keys and a gates dict (values may fail — synthetic data has no
    real edge, which is the honest outcome)."""
    start, end = repo["days"][0], repo["days"][-1]
    report = run_backtest(start, end, stress=False, tune=False, args_note="test")
    for key in ["win_rate", "expectancy_pct_net", "coverage_of_candidates",
                "sharpe_daily", "brier_score", "calibration_gap", "fairness",
                "per_symbol_coverage", "gates"]:
        assert key in report
    assert "accuracy" not in report  # banned word


def test_run_registry_appended(repo):
    from src.intraday import ROOT, load_config
    start, end = repo["days"][0], repo["days"][-1]
    run_backtest(start, end, tune=False, args_note="reg-test")
    reg = ROOT / load_config()["paths"]["run_registry"]
    assert reg.exists()
    df = pd.read_csv(reg)
    assert "win_rate" in df.columns and "all_pass" in df.columns


# ── PLANTED-LEAK alarm (PLAN_v3 §12.2 / §8) ──────────────────────────────

def test_planted_leak_trips_leakage_alarm(repo):
    """An implausibly high win rate at non-trivial coverage is the signature of
    lookahead. Plant exactly that — win rate above gates.leakage_alarm_win_rate
    and coverage above gates.leakage_alarm_coverage — and _finalize MUST raise
    LeakageAlarm rather than emit a report. Removing the alarm would let the
    leaked run through (this is the executable leakage guard)."""
    g = load_config()["gates"]
    n = 60                                   # 60 fired trades
    candidates = 200                         # → coverage 0.30 > leakage_alarm_coverage
    assert n / candidates > g["leakage_alarm_coverage"]
    day = _date(2026, 1, 5)
    # 59/60 wins → win rate 0.983 > leakage_alarm_win_rate (0.92)
    trades = [_trade(day, f"S{i % 4}", "Tech", label=1) for i in range(n - 1)]
    trades.append(_trade(day, "S0", "Tech", label=0))
    win_rate = sum(t["label"] for t in trades) / n
    assert win_rate > g["leakage_alarm_win_rate"]

    data = pd.DataFrame({"date": [day] * candidates})
    with pytest.raises(LeakageAlarm):
        _finalize(trades, [], data, stress=False, start=day, end=day,
                  args_note="planted-leak", all_test_days=[day])


def test_no_alarm_below_thresholds(repo):
    """The control must NOT fire on an honest run: a believable win rate at the
    same coverage emits a report, not an alarm — proving the alarm keys on the
    win-rate threshold and isn't always-on."""
    g = load_config()["gates"]
    n, candidates = 60, 200
    day = _date(2026, 1, 6)
    # ~60% win rate — well under leakage_alarm_win_rate; vary p_cal so the
    # reliability-diagram binning has distinct edges
    trades = [_trade(day, f"S{i % 4}", "Tech", label=int(i % 5 < 3),
                     p_cal=0.50 + 0.005 * i) for i in range(n)]
    wr = sum(t["label"] for t in trades) / n
    assert wr < g["leakage_alarm_win_rate"]
    data = pd.DataFrame({"date": [day] * candidates})
    report = _finalize(trades, [], data, stress=False, start=day, end=day,
                       args_note="honest", all_test_days=[day])
    assert report["win_rate"] == pytest.approx(wr)
    assert "gates" in report


# ── H-5 subsidy / per-symbol fairness path (§8.3) ────────────────────────

def test_subsidy_breach_fails_fairness_gate(repo):
    """H-5: a symbol with ≥30 trades and negative expectancy is being subsidised
    by the book and must trip the no-subsidy gate. We craft one losing symbol
    (LOSER) carried by profitable others; the report's no_subsidy_ok gate must be
    False and the symbol named in the subsidy list."""
    day = _date(2026, 1, 7)
    trades = []
    # LOSER: 35 trades, mostly losses, clearly negative expectancy
    for i in range(35):
        trades.append(_trade(day, "LOSER", "Energy", label=int(i % 7 == 0),
                             p_cal=0.50 + 0.004 * i, pnl=-0.02, atr=0.012))
    # WINNER & friends: profitable, keep the headline win rate believable but
    # under the leakage alarm so the report is actually produced
    for i in range(40):
        trades.append(_trade(day, f"W{i % 3}", "Tech", label=int(i % 3 != 0),
                             p_cal=0.55 + 0.005 * i, pnl=0.012, atr=0.008 + (i % 3) * 0.002))

    # direct unit assertion on the fairness control
    from src.intraday.backtester import _fairness_tables
    fairness = _fairness_tables(pd.DataFrame(trades))
    breaches = _subsidy_breaches(fairness)
    assert "symbol:LOSER" in breaches
    assert fairness["symbol"]["LOSER"]["n"] >= 30
    assert fairness["symbol"]["LOSER"]["expectancy"] < 0

    # and end-to-end through the report's gate dict
    data = pd.DataFrame({"date": [day] * 300})
    report = _finalize(trades, [], data, stress=False, start=day, end=day,
                       args_note="subsidy", all_test_days=[day])
    assert report["gates"]["no_subsidy_ok"] is False
    assert "symbol:LOSER" in _subsidy_breaches(report["fairness"])
    assert report["gates"]["all_pass"] is False   # a subsidy must sink the run


def test_no_subsidy_when_all_positive_expectancy(repo):
    """Inverse: every symbol with ≥30 trades is profitable → no subsidy breach,
    gate passes. Proves the subsidy detector keys on negative expectancy, not
    merely on trade count."""
    day = _date(2026, 1, 8)
    trades = [_trade(day, f"S{i % 3}", "Tech", label=int(i % 4 != 0),
                     p_cal=0.90, pnl=0.015, atr=0.008 + (i % 3) * 0.002)
              for i in range(99)]
    from src.intraday.backtester import _fairness_tables
    fairness = _fairness_tables(pd.DataFrame(trades))
    assert _subsidy_breaches(fairness) == []
