"""Regime channel (PLAN_v4 §5/§7): point-in-time global reads, veto logic, and
graceful degradation of the forward-only news channel — all on synthetic data,
no network, no yfinance/FinBERT required."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import pytest

from src.intraday import load_config
from src.intraday.regime_data import GLOBAL_COLUMNS, REGIME_MARKET_FEATURES, regime_features


def _regime_dir() -> Path:
    from src.intraday import ROOT
    base = load_config()["data"].get("regime_path", "data/regime")
    p = Path(base)
    p = p if p.is_absolute() else ROOT / p
    return p / "global"


def _write_global(rows: list[dict]) -> None:
    d = _regime_dir()
    d.mkdir(parents=True, exist_ok=True)
    by_day: dict[date, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["date"], []).append(r)
    for day, recs in by_day.items():
        pd.DataFrame(recs, columns=GLOBAL_COLUMNS).to_parquet(d / f"{day.isoformat()}.parquet",
                                                              index=False)


def test_regime_features_absent_all_present_zero(repo):
    """No regime files → every feature present=0 (fail-open contract)."""
    feats = regime_features(date(2026, 3, 2))
    for k in REGIME_MARKET_FEATURES:
        assert k + "_present" in feats
        assert feats[k + "_present"] == 0.0


def test_prior_close_is_pit_safe_and_used(repo):
    """Prior-session closes (content date < day) are always usable and produce a
    real return feature with present=1."""
    day = date(2026, 3, 10)
    cap = datetime.combine(day, time(8, 50)).isoformat()
    rows = [
        {"date": date(2026, 3, 6), "instrument": "spx", "field": "close",
         "value": 5000.0, "source": "t", "capture_ts": cap},
        {"date": date(2026, 3, 9), "instrument": "spx", "field": "close",
         "value": 5050.0, "source": "t", "capture_ts": cap},
    ]
    _write_global(rows)
    feats = regime_features(day)
    assert feats["spx_ret_1d_present"] == 1.0
    assert feats["spx_ret_1d"] == pytest.approx(5050.0 / 5000.0 - 1.0)


def test_same_day_snapshot_excluded_when_captured_after_decision(repo):
    """A same-day GIFT snapshot captured AFTER 09:30 must not leak into a 09:30
    decision — point_in_time drops it (PLAN_v4 §5)."""
    day = date(2026, 3, 11)
    late = datetime.combine(day, time(9, 45)).isoformat()   # AFTER the 09:30 decision
    early_prior = datetime.combine(day, time(8, 50)).isoformat()
    rows = [
        {"date": date(2026, 3, 10), "instrument": "nsei", "field": "close",
         "value": 22000.0, "source": "t", "capture_ts": early_prior},
        {"date": day, "instrument": "gift", "field": "close",
         "value": 22200.0, "source": "t", "capture_ts": late},
    ]
    _write_global(rows)
    feats = regime_features(day, pd.Timestamp(datetime.combine(day, time(9, 30))))
    assert feats["gift_gap_pct_present"] == 0.0   # late snapshot excluded


def test_same_day_snapshot_used_when_captured_before_decision(repo):
    day = date(2026, 3, 12)
    early = datetime.combine(day, time(9, 5)).isoformat()    # BEFORE 09:30
    prior = datetime.combine(day, time(8, 50)).isoformat()
    rows = [
        {"date": date(2026, 3, 11), "instrument": "nsei", "field": "close",
         "value": 22000.0, "source": "t", "capture_ts": prior},
        {"date": day, "instrument": "gift", "field": "close",
         "value": 22110.0, "source": "t", "capture_ts": early},
    ]
    _write_global(rows)
    feats = regime_features(day, pd.Timestamp(datetime.combine(day, time(9, 30))))
    assert feats["gift_gap_pct_present"] == 1.0
    assert feats["gift_gap_pct"] == pytest.approx(22110.0 / 22000.0 - 1.0)


# ── veto logic (PLAN_v4 §7) ───────────────────────────────────────────

def _risk_off(day: date) -> None:
    """Write a coherent risk-off morning: equities down, INR down, VIX spiking."""
    from datetime import timedelta
    cap = datetime.combine(day, time(8, 50)).isoformat()
    prior = day - timedelta(days=1)
    prior2 = day - timedelta(days=2)
    prior6 = day - timedelta(days=6)
    rows = []
    for instr, v_prev2, v_prev in [("spx", 5000, 4900), ("ndx", 17000, 16600),
                                    ("nikkei", 39000, 38200), ("usdinr", 83.0, 83.6)]:
        rows.append({"date": prior2, "instrument": instr, "field": "close",
                     "value": float(v_prev2), "source": "t", "capture_ts": cap})
        rows.append({"date": prior, "instrument": instr, "field": "close",
                     "value": float(v_prev), "source": "t", "capture_ts": cap})
    # VIX 5-day change > vix_spike_5d (0.20): 14 → 20 (~43%)
    rows.append({"date": prior6, "instrument": "india_vix", "field": "close",
                 "value": 14.0, "source": "t", "capture_ts": cap})
    rows.append({"date": prior, "instrument": "india_vix", "field": "close",
                 "value": 20.0, "source": "t", "capture_ts": cap})
    _write_global(rows)


def test_veto_inactive_is_failopen(repo):
    """With regime.active=false (default) the veto never fires."""
    from src.intraday.regime import veto
    day = date(2026, 3, 13)
    _risk_off(day)
    assert veto(day, +1)[0] is False


def _activate_regime() -> None:
    import yaml
    import src.intraday as si
    cfg = yaml.safe_load(si.CONFIG_PATH.read_text())
    cfg["regime"]["active"] = True
    si.CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
    si.load_config.cache_clear()


def test_day_regime_reads_risk_off(repo):
    from src.intraday.regime import day_regime
    day = date(2026, 3, 14)
    _risk_off(day)
    v = day_regime(day)
    assert v.present is True
    assert v.risk_score < 0        # coherent risk-off tape
    assert v.bias == -1


def test_veto_suppresses_conflicting_long_when_active(repo):
    from src.intraday.regime import veto
    day = date(2026, 3, 17)
    _risk_off(day)
    _activate_regime()
    # a long into a strong risk-off regime is vetoed; a short is not
    assert veto(day, +1)[0] is True
    assert veto(day, -1)[0] is False


# ── news channel degrades gracefully (PLAN_v4 §8) ─────────────────────

def test_news_features_absent_present_zero(repo):
    from src.intraday.news_regime import NEWS_FEATURES, news_features
    feats = news_features("AAA", "Tech", date(2026, 3, 2))
    for k in NEWS_FEATURES:
        assert feats[k + "_present"] == 0.0


# ── explicit lookahead property for the regime group (PLAN_v4 §12) ────
# The regime channel's analogue of test_no_lookahead_property: adding data that
# a 09:30 decision on day D could not have seen (future content dates, same-day
# rows captured after 09:30) must leave day D's regime features bit-identical.

def test_regime_features_future_data_immunity(repo):
    from datetime import timedelta
    day = date(2026, 3, 20)
    cap_ok = datetime.combine(day, time(8, 45)).isoformat()
    _write_global([
        {"date": day - timedelta(days=2), "instrument": "spx", "field": "close",
         "value": 5000.0, "source": "t", "capture_ts": cap_ok},
        {"date": day - timedelta(days=1), "instrument": "spx", "field": "close",
         "value": 5100.0, "source": "t", "capture_ts": cap_ok},
        {"date": day - timedelta(days=1), "instrument": "india_vix", "field": "close",
         "value": 15.0, "source": "t", "capture_ts": cap_ok},
    ])
    decision = pd.Timestamp(datetime.combine(day, time(9, 30)))
    before = regime_features(day, decision)

    # inject everything a 09:30 decision must NOT see:
    late_cap = datetime.combine(day, time(15, 45)).isoformat()
    _write_global([
        # future content dates (tomorrow's closes, backfilled later)
        {"date": day + timedelta(days=1), "instrument": "spx", "field": "close",
         "value": 4000.0, "source": "t", "capture_ts": late_cap},
        {"date": day + timedelta(days=1), "instrument": "india_vix", "field": "close",
         "value": 40.0, "source": "t", "capture_ts": late_cap},
        # a same-day snapshot captured AFTER the decision
        {"date": day, "instrument": "gift", "field": "close",
         "value": 99999.0, "source": "t", "capture_ts": late_cap},
    ])
    after = regime_features(day, decision)
    assert after == before          # bit-identical row → no lookahead path exists


def test_regime_day_cache_consistency(repo):
    """The per-day cache in features._regime_row must return the same market
    values for every symbol on a day (and not error when data is absent)."""
    from src.intraday.features import _REGIME_DAY_CACHE, _market_regime_for_day
    _REGIME_DAY_CACHE.clear()
    day = date(2026, 3, 23)
    cap = datetime.combine(day, time(8, 45)).isoformat()
    _write_global([
        {"date": date(2026, 3, 19), "instrument": "spx", "field": "close",
         "value": 5000.0, "source": "t", "capture_ts": cap},
        {"date": date(2026, 3, 20), "instrument": "spx", "field": "close",
         "value": 5050.0, "source": "t", "capture_ts": cap},
    ])
    f1, v1 = _market_regime_for_day(day, None)
    f2, v2 = _market_regime_for_day(day, None)   # second symbol, same day → cache hit
    assert f1 is f2 and v1 is v2                 # identical objects: one read per day
    assert f1["spx_ret_1d_present"] == 1.0
