"""Daily prediction-and-listing system tests.

The load-bearing one is test_no_lookahead_property (cloned from test_features.py):
a decision-day row computed on data truncated at that day must equal the row from
the full panel. Plus schema parity, the staged-gate pipeline end-to-end on
synthetic daily panels, and the per-horizon cost profiles.
"""

from __future__ import annotations

import textwrap
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _make_panel(symbol: str, days: list[date], seed: int) -> pd.DataFrame:
    """Synthetic daily panel with mild momentum (so AUC can exceed 0.5) — exercises
    the pipeline without asserting a particular verdict."""
    rng = np.random.RandomState(seed)
    n = len(days)
    rets = np.zeros(n)
    prev = 0.0
    for i in range(n):
        rets[i] = 0.35 * prev + rng.normal(0, 0.012)   # AR(1) momentum
        prev = rets[i]
    close = 1000 * np.cumprod(1 + rets)
    prev_close = np.concatenate([[close[0] / (1 + rets[0])], close[:-1]])
    openp = prev_close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum.reduce([openp, close]) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum.reduce([openp, close]) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = (1e6 * (1 + np.abs(rng.normal(0, 0.3, n)))).astype(float)
    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d in days],
        "open": openp, "high": high, "low": low, "close": close, "prev_close": prev_close,
        "volume": volume, "turnover_lacs": close * volume / 1e5,
        "deliv_per": 50 + rng.normal(0, 8, n),
        "oi": (5e6 * (1 + rng.normal(0, 0.1, n))), "oi_change": rng.normal(0, 1e5, n),
        "basis_pct": rng.normal(0, 0.002, n),
        "fut_close": close * (1 + rng.normal(0, 0.001, n)), "spot_close": close,
        "adj_factor": 1.0, "capture_ts": [datetime.combine(d, time(18, 0)) for d in days],
    })


@pytest.fixture
def drepo(tmp_path, monkeypatch):
    import src.daily as sd
    import src.daily.features as feat  # noqa: F401  (cache clears below)

    data = tmp_path / "data"
    ref = data / "reference"
    daily = data / "daily"
    glob = data / "global"
    for d in (ref, daily, glob):
        d.mkdir(parents=True)
    for sub in ("", "charts", "lists", "signals"):
        (tmp_path / "reports" / "daily" / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "daily").mkdir(parents=True)

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    ref.joinpath("universe_membership.csv").write_text(
        "symbol,name,sector,from_date,to_date\n"
        + "\n".join(f"{s},{s} Ltd,Tech,2019-01-01," for s in symbols) + "\n"
    )

    days = [d.date() for d in pd.bdate_range(date(2023, 1, 2), date(2024, 12, 31))]
    for i, s in enumerate(symbols):
        _make_panel(s, days, seed=i * 100 + 7).to_parquet(daily / f"{s}.parquet", index=False)

    base = yaml.safe_load((ROOT / "config" / "config_daily.yaml").read_text())
    base["universe"]["membership_file"] = str(ref / "universe_membership.csv")
    base["data"]["bhavcopy_path"] = str(data / "bhavcopy")
    base["data"]["daily_path"] = str(daily)
    base["data"]["global_path"] = str(glob)
    base["data"]["reference_path"] = str(ref)
    base["data"]["min_warmup_days"] = 40
    base["training"].update(train_min_months=6, test_fold_months=2, min_folds=4,
                            inner_calib_days=20, tune=False)
    base["paths"] = {k: str(tmp_path / v) for k, v in base["paths"].items()}

    cfg_path = tmp_path / "config_daily.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False))
    monkeypatch.setattr(sd, "DAILY_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(sd, "ROOT", tmp_path)
    sd.load_daily_config.cache_clear()
    yield {"root": tmp_path, "symbols": symbols, "days": days}
    sd.load_daily_config.cache_clear()


@pytest.fixture
def frozen_day(drepo):
    return drepo["days"][120]   # well past the 40-day warmup


def test_no_lookahead_property(drepo, frozen_day):
    """Features for a decision day computed on a panel truncated at that day must
    equal the features from the full panel (the daily PIT guard)."""
    from src.daily.features import FEATURE_ORDER, features_for_day
    from src.daily.panel import load_symbol_daily

    panel = pd.read_parquet(drepo["root"] / "data" / "daily" / "AAA.parquet")
    full = features_for_day("AAA", frozen_day, df_daily=panel)
    assert not full.empty
    truncated = panel[pd.to_datetime(panel["date"]).dt.date <= frozen_day]
    sub = features_for_day("AAA", frozen_day, df_daily=truncated)
    assert not sub.empty
    for col in FEATURE_ORDER:
        a, b = full.iloc[0][col], sub.iloc[0][col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"{col} leaked under truncation"


def test_future_panel_rows_do_not_change_features(drepo, frozen_day):
    """Appending rows dated after the decision day must not change that day's row."""
    from src.daily.features import FEATURE_ORDER, features_for_day

    panel = pd.read_parquet(drepo["root"] / "data" / "daily" / "AAA.parquet")
    before = features_for_day("AAA", frozen_day, df_daily=panel).iloc[0]
    trunc = panel[pd.to_datetime(panel["date"]).dt.date <= frozen_day]
    after = features_for_day("AAA", frozen_day, df_daily=trunc).iloc[0]
    for col in FEATURE_ORDER:
        if pd.isna(before[col]) and pd.isna(after[col]):
            continue
        assert before[col] == pytest.approx(after[col], rel=1e-9, abs=1e-9)


def test_schema_check_catches_missing(drepo, frozen_day):
    from src.daily.features import FeatureSchemaError, build_matrix, check_schema

    plan = pd.DataFrame({"symbol": ["AAA", "BBB"], "date": [frozen_day, frozen_day]})
    m = build_matrix(plan)
    check_schema(m)
    with pytest.raises(FeatureSchemaError):
        check_schema(m.drop(columns=["atr_pct"]))


def test_cost_profiles(drepo):
    from src.daily.costs import round_trip_cost_pct

    deliv = round_trip_cost_pct("next_day", price=1000.0, qty=100)
    assert 0 < deliv < 0.01                       # delivery stack is cheap, no slippage
    deliv_stress = round_trip_cost_pct("next_day", price=1000.0, qty=100, stress=True)
    assert deliv_stress > deliv


def test_label_day_next_day(drepo, frozen_day):
    from src.daily.labeler import label_day

    panel = pd.read_parquet(drepo["root"] / "data" / "daily" / "AAA.parquet")
    out = label_day("AAA", frozen_day, "next_day", df_daily=panel)
    assert out is not None
    assert out.label in (0, 1)
    assert out.barrier in ("target", "stop", "time")
    assert out.entry_ts.date() > frozen_day      # entry is the NEXT session (no lookahead)


def test_staged_gate_pipeline(drepo):
    """End-to-end: build → walk-forward → staged gate produces a structured verdict.
    We assert structure + that the gate RAN, not that it passed (honest)."""
    from src.daily.backtester import run_backtest

    rep = run_backtest(date(2023, 1, 2), date(2024, 12, 31), horizon="next_day")
    assert rep["horizon"] == "next_day"
    assert rep["n_folds"] >= 4
    m1, m2 = rep["milestone1_predictive"], rep["milestone2_after_cost"]
    assert 0.0 <= m1["oos_auc"] <= 1.0
    assert set(m1["criteria"]) == {"auc_ok", "fold_consistency_ok", "calibration_ok", "lift_ok"}
    assert isinstance(m1["passed"], bool) and isinstance(m2["passed"], bool)
    assert "verdict" in rep
    # artifacts written
    reg = drepo["root"] / "reports" / "daily" / "run_registry.csv"
    assert reg.exists()

    # the at-scale scoreboard reads the OOS artifact this run wrote
    from src.daily.evaluate import scoreboard_from_oos
    sb = scoreboard_from_oos("next_day")
    assert sb["n_picks"] > 0 and sb["n_days"] > 0
    assert 0.0 <= sb["hit_rate"] <= 1.0
    assert "edge_vs_base" in sb and "mean_return_net" in sb


def test_live_predict_then_evaluate(drepo):
    """The live loop: train → list (predict) → evaluate the list against the real
    matured outcome → scoreboard. Lookahead-free (train cutoff precedes the
    predicted day)."""
    from src.daily.evaluate import evaluate_day, scoreboard
    from src.daily.run_list import run
    from src.daily.trainer import train_production

    train_production(horizon="next_day", end_day=date(2024, 12, 18))
    pred_day = date(2024, 12, 24)                 # after the train cutoff; 12-26/27 mature it
    run(day=pred_day, horizons=["next_day"])
    res = evaluate_day(pred_day, horizon="next_day")
    assert res["status"] == "matured"
    assert res["n_picks"] >= 1
    assert 0.0 <= res["hit_rate"] <= 1.0
    for p in res["picks"]:
        assert p["actual_up"] in (0, 1) and "actual_return" in p
    sb = scoreboard("next_day")                   # running live scoreboard accrued one day
    assert sb["n_days"] == 1
    # base rate is the universe-wide up-rate (not the picks' own), so edge_vs_base is meaningful
    assert 0.0 <= res["base_up_rate"] <= 1.0
    assert res["base_up_rate"] == res["base_up_rate"]     # not NaN


def test_earnings_release_ts_filtering(drepo, frozen_day):
    """A future earnings event whose release_ts is after the decision time must NOT
    leak into days_to_earnings (the PIT fix for events.csv)."""
    import pandas as pd
    from datetime import datetime, time
    from src.daily.features import _read_csv_cached, features_for_day

    day = frozen_day
    ref = drepo["root"] / "data" / "reference"
    panel = pd.read_parquet(drepo["root"] / "data" / "daily" / "AAA.parquet")

    # event dated 5 days out, but "announced" (release_ts) 3 days AFTER the decision.
    future_release = datetime.combine(day + pd.Timedelta(days=3), time(9, 0))
    ref.joinpath("events.csv").write_text(
        "date,type,symbol,release_ts\n"
        f"{(day + pd.Timedelta(days=5)).isoformat()},results,AAA,{future_release.isoformat()}\n"
    )
    _read_csv_cached.cache_clear()
    f = features_for_day("AAA", day, df_daily=panel)
    assert f.iloc[0]["days_to_earnings_present"] == 0.0   # not yet public → excluded

    # same event, but released BEFORE the decision → now visible
    past_release = datetime.combine(day - pd.Timedelta(days=1), time(9, 0))
    ref.joinpath("events.csv").write_text(
        "date,type,symbol,release_ts\n"
        f"{(day + pd.Timedelta(days=5)).isoformat()},results,AAA,{past_release.isoformat()}\n"
    )
    _read_csv_cached.cache_clear()
    f2 = features_for_day("AAA", day, df_daily=panel)
    assert f2.iloc[0]["days_to_earnings_present"] == 1.0
    assert f2.iloc[0]["days_to_earnings"] == 5.0
