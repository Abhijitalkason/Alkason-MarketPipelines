"""Regression guards for the production bug-fix pass.

Each test pins one fix so a future change can't silently reintroduce the bug.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


# ── screener: fit_weights learns on the SAME transform score_rows applies ──

def test_screener_clip_transform_shared(repo):
    from src.intraday.screener import clipped_components, score_rows
    raw = pd.DataFrame({
        "vol_z": [10.0, -10.0], "rng_z": [5.0, -5.0], "gap_quality": [1.0, 0.0],
        "imbalance_aligned": [3.0, -3.0], "delivery_z": [9.0, -9.0],
        "index_alignment": [1.0, 0.0],
    })
    c = clipped_components(raw)
    assert c["vol_z"].tolist() == [3.0, -3.0]          # clipped to ±3
    assert c["imbalance_aligned"].tolist() == [1.0, -1.0]  # clipped to ±1
    # score_rows must consume exactly these clipped values
    w = {k: 1.0 for k in ["volume_surge", "range_expansion", "gap_quality",
                          "preopen_imbalance", "delivery_z", "index_alignment"]}
    expected = c.sum(axis=1)
    pd.testing.assert_series_equal(score_rows(raw, w), expected, check_names=False)


# ── price model: tuning skips (not crashes) on a single-class inner block ──

def test_tune_single_class_returns_empty(repo):
    from src.intraday.price_model import PriceModel
    n = 200
    X = pd.DataFrame({f: np.random.RandomState(0).normal(size=n)
                      for f in PriceModel.FEATURES})
    X["date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 10, "D")
    y = pd.Series([1] * n)                      # single class everywhere
    assert PriceModel().tune(X, y, n_trials=3) == {}   # no crash, empty params


# ── backtester: fairness table survives low-cardinality atr_pct ──

def test_fairness_table_constant_atr_no_crash(repo):
    from src.intraday.backtester import _fairness_tables
    tdf = pd.DataFrame({
        "symbol": ["AAA"] * 5, "sector": ["Tech"] * 5,
        "atr_pct": [0.007] * 5,                 # constant → qcut would collapse
        "tod_bucket": ["0930-1000"] * 5,
        "label": [1, 0, 1, 1, 0], "pnl_pct_net": [0.01, -0.01, 0.01, 0.01, -0.01],
        "p_cal": [0.9] * 5,
    })
    out = _fairness_tables(tdf)                 # must not raise
    assert "vol_regime" in out and "all" in out["vol_regime"]


# ── drift: streak is keyed by date — two checks same day don't double-count ──

def test_drift_streak_not_double_counted(repo):
    from src.intraday import drift
    drift._clear_breach_streak()
    d = date(2026, 4, 1)
    assert drift._consecutive_breaches(d) == 1
    assert drift._consecutive_breaches(d) == 1   # same day → unchanged
    assert drift._consecutive_breaches(date(2026, 4, 2)) == 2  # next day → +1


# ── trainer: config flow flag write preserves comments and is a line edit ──

def test_set_flow_active_preserves_comments(repo):
    import src.intraday as si
    from src.intraday.trainer import _set_flow_active

    # inject a sentinel comment next to the flag so we can prove the line-edit
    # leaves the rest of the file (comments and all) untouched
    text = si.CONFIG_PATH.read_text().replace(
        "flow_model_active: false",
        "flow_model_active: false   # SENTINEL-COMMENT keep me",
    )
    si.CONFIG_PATH.write_text(text)
    si.load_config.cache_clear()

    _set_flow_active(True)
    after = si.CONFIG_PATH.read_text()
    assert "flow_model_active: true" in after
    assert "SENTINEL-COMMENT keep me" in after        # comment survived the edit
    si.load_config.cache_clear()
    assert si.load_config()["gate"]["flow_model_active"] is True


# ── paper runner: the explainer is actually wired (PLAN_v3 §15) ──

def test_explainer_is_callable_from_fill_payload():
    from src.slm.explainer import explain_signal
    fill = {"symbol": "AAA", "direction": 1, "entry": 1000.0, "qty": 10,
            "p_cal": 0.93, "shap_top": {"vwap_dist_atr": 0.4, "ret_3b": -0.1}}
    text = explain_signal(fill)                  # Ollama absent → template fallback
    assert isinstance(text, str) and "AAA" in text


def test_paper_runner_wires_explain():
    import inspect
    from src.intraday import paper_runner
    src = inspect.getsource(paper_runner)
    assert "_explain_async" in src and 'journal_write("explain"' in src
