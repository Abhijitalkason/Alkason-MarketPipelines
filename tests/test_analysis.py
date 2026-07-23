"""Tests for the single-stock deep-analysis engine (src/analysis) — pure logic
only (scorecard math, verdict thresholds, honest degradation). No network."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.engine import (
    BUY_MIN,
    SELL_MAX,
    _lexicon_score,
    build_scorecard,
    technicals,
    verdict_from_scorecard,
)


def _tech(**over):
    base = {"close": 100.0, "sma20": 95.0, "sma50": 90.0, "sma200": 85.0,
            "ret_1d": 0.01, "ret_1w": 0.03, "vol_ratio_20": 1.8,
            "rsi14": 55.0, "pos_52w": 0.85}
    base.update(over)
    return base


_EMPTY = {"present": False}


def test_bullish_evidence_yields_buy():
    card = build_scorecard(_tech(), {"present": True, "deliv_z": 1.5},
                           _EMPTY,
                           {"present": True, "mean_sentiment": 0.5, "n_week": 6,
                            "n_articles": 10, "engine": "lexicon"},
                           {"present": True, "india_vix": 12.0, "note": "calm"},
                           _EMPTY)
    v = verdict_from_scorecard(card)
    assert v["action"] == "BUY" and v["composite_score"] >= BUY_MIN


def test_bearish_evidence_yields_sell():
    bear = _tech(close=80.0, sma20=85.0, sma50=90.0, sma200=95.0,
                 ret_1d=-0.03, ret_1w=-0.08, vol_ratio_20=2.0,
                 rsi14=38.0, pos_52w=0.05)
    card = build_scorecard(bear, {"present": True, "deliv_z": 1.5},
                           _EMPTY,
                           {"present": True, "mean_sentiment": -0.6, "n_week": 8,
                            "n_articles": 12, "engine": "lexicon"},
                           {"present": True, "india_vix": 26.0, "note": "elevated"},
                           _EMPTY)
    # deliv surge into a falling price is conviction selling → negative
    v = verdict_from_scorecard(card)
    assert v["action"] == "SELL" and v["composite_score"] <= SELL_MAX


def test_mixed_evidence_holds():
    card = build_scorecard(_tech(ret_1d=0.0, ret_1w=0.0, vol_ratio_20=1.0,
                                 pos_52w=0.5, rsi14=50.0,
                                 close=90.0, sma20=91.0, sma50=89.0, sma200=88.0),
                           _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY)
    v = verdict_from_scorecard(card)
    assert v["action"] == "HOLD"


def test_absent_components_score_zero_and_lower_completeness():
    card = build_scorecard(_tech(), _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY)
    absent = [r for r in card if not r["present"]]
    assert absent and all(r["score"] == 0.0 for r in absent)
    v = verdict_from_scorecard(card)
    assert v["data_completeness"] < 1.0


def test_all_absent_is_hold_never_fabricated():
    none_tech = {k: None for k in _tech()}
    card = build_scorecard(none_tech, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY)
    v = verdict_from_scorecard(card)
    assert v["action"] == "HOLD" and v["composite_score"] == 0.0
    assert v["data_completeness"] == 0.0 and v["confidence"] == "LOW"


def test_lexicon_score_directions():
    assert _lexicon_score("Company wins record order, profit surges") > 0
    assert _lexicon_score("Regulator probe deepens as losses mount") < 0
    assert _lexicon_score("Board meeting scheduled for Tuesday") == 0.0


def test_technicals_on_synthetic_series():
    n = 300
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(np.linspace(100, 150, n))  # steady uptrend
    df = pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                       "low": close * 0.99, "close": close, "volume": 1000.0})
    t = technicals(df)
    assert t["close"] == 150.0
    assert t["sma20"] < t["close"] and t["sma200"] < t["sma50"]  # uptrend ordering
    assert t["pos_52w"] == 1.0
    assert t["ret_1m"] > 0


def test_trade_plan_geometry_buy_1to5d():
    from src.analysis.engine import trade_plan
    tech = {"close": 100.0, "atr_pct": 0.02}   # ATR=2 → R=1×ATR=2 (swing_1_5d contract)
    p = trade_plan("BUY", tech)
    assert p["risk_per_share"] == 2.0
    assert p["stop_price"] == 98.0
    assert p["target_1"] == 102.0 and p["target_2"] == 104.0
    assert p["max_hold_days"] == 5
    assert p["qty_ref"] == 1000
    assert p["qty_if_risking_1pct"] == 500   # 1% of 1L = 1000 / R=2


def test_trade_plan_none_for_hold_or_missing_data():
    from src.analysis.engine import trade_plan
    assert trade_plan("HOLD", {"close": 100.0, "atr_pct": 0.02}) is None
    assert trade_plan("BUY", {"close": None, "atr_pct": 0.02}) is None
