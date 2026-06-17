"""Backtester: dataset assembly, day-boundary inner split, fit_fold OOF
discipline, end-to-end smoke run on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.intraday.backtester import (_inner_split, build_dataset, fit_fold, make_folds,
                                     run_backtest)


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
