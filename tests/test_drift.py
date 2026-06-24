"""Drift → ACI recalibration flag → halt (v3_supplementry §12.4, B-7 /
PLAN_v3 §13 switch c).

run_check computes the drifted-column share, and on a breach (share ≥
share_drifted_columns_alert) it MUST: set the recalibrate-next-session flag,
append an alert, and after consecutive_breaches_to_halt breaching SESSIONS,
persist a halt. Clean data must do none of these. We drive the control flow by
controlling the Evidently share (the share computation itself is exercised
end-to-end in test_real_evidently_clean_no_breach), so each test would fail if
the corresponding control were removed.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.intraday import ROOT, load_config
from src.intraday.features import PRICE_FEATURES
from src.intraday import drift as drift_mod
from src.intraday.drift import run_check, recalib_flag_set, clear_recalib_flag
from src.intraday.risk import clear_halt, is_halted, halt_reason


def _state_dir():
    p = ROOT / load_config()["paths"]["state"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_reference(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cols = PRICE_FEATURES[:8]
    ref = pd.DataFrame({c: rng.normal(0, 1, n) for c in cols})
    ref.to_parquet(_state_dir() / "feature_matrix.parquet", index=False)
    return ref


def _write_signals(day: date, n: int = 60, shift: float = 0.0, seed: int = 1) -> None:
    """A signals-log file of live rows carrying full feature vectors."""
    rng = np.random.RandomState(seed)
    cols = PRICE_FEATURES[:8]
    sig_dir = ROOT / load_config()["paths"]["signals_log"]
    sig_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for _ in range(n):
        feats = {c: float(rng.normal(shift, 1)) for c in cols}
        recs.append({"symbol": "AAA", "label": 1, "pnl_pct_net": 0.0, "features": feats})
    (sig_dir / f"paper_{day.isoformat()}.json").write_text(json.dumps(recs, default=str))


@pytest.fixture(autouse=True)
def _clean_halt_and_flag(repo):
    clear_halt()
    clear_recalib_flag()
    f = _state_dir() / "drift_breach_streak.json"
    if f.exists():
        f.unlink()
    yield
    clear_halt()
    clear_recalib_flag()


def test_breach_sets_recalib_flag_and_alert(repo, monkeypatch):
    _write_reference()
    day = date(2026, 4, 1)
    _write_signals(day)
    # force a breach via the share (the share→breach→flag wiring is what we test)
    monkeypatch.setattr(drift_mod, "_evidently_drift",
                        lambda ref, cur, d: (0.95, drift_mod._drift_dir() / f"{d.isoformat()}.html"))

    assert not recalib_flag_set()
    summary = run_check(day)
    assert summary["checked"] is True
    assert summary["breach"] is True
    assert recalib_flag_set(), "breach must force ACI recalibration next session"

    # an alert must have been appended
    alerts_dir = ROOT / load_config()["paths"]["alerts"]
    alert_file = alerts_dir / f"{day.isoformat()}.json"
    assert alert_file.exists()
    alerts = json.loads(alert_file.read_text())
    assert any(a["kind"] == "drift" for a in alerts)
    # one breach is below the 2-breach halt threshold
    assert not is_halted()


def test_two_consecutive_breaches_halt(repo, monkeypatch):
    _write_reference()
    monkeypatch.setattr(drift_mod, "_evidently_drift",
                        lambda ref, cur, d: (0.95, drift_mod._drift_dir() / f"{d.isoformat()}.html"))
    d1, d2 = date(2026, 4, 1), date(2026, 4, 2)
    _write_signals(d1, seed=1)
    s1 = run_check(d1)
    assert s1["breach"] and not is_halted()        # first breach: flagged, not halted

    _write_signals(d2, seed=2)
    s2 = run_check(d2)
    assert s2["breach"]
    assert is_halted(), "2 consecutive breaching sessions must halt (switch c)"
    assert "drift" in halt_reason()
    assert s2.get("halted") is True


def test_same_day_rerun_does_not_double_count(repo, monkeypatch):
    """Running the drift check twice on ONE day must not trip the 2-breach halt —
    the streak is keyed by session date (guards `--mode drift` + post-market)."""
    _write_reference()
    monkeypatch.setattr(drift_mod, "_evidently_drift",
                        lambda ref, cur, d: (0.95, drift_mod._drift_dir() / f"{d.isoformat()}.html"))
    day = date(2026, 4, 1)
    _write_signals(day)
    run_check(day)
    assert not is_halted()
    run_check(day)            # same day again
    assert not is_halted(), "same-session re-run must not double-increment the streak"


def test_clean_data_no_flag_no_halt(repo, monkeypatch):
    _write_reference()
    day = date(2026, 4, 1)
    _write_signals(day)
    # share below the alert threshold → no breach
    monkeypatch.setattr(drift_mod, "_evidently_drift",
                        lambda ref, cur, d: (0.0, drift_mod._drift_dir() / f"{d.isoformat()}.html"))
    summary = run_check(day)
    assert summary["checked"] is True
    assert summary["breach"] is False
    assert not recalib_flag_set(), "clean data must not flag recalibration"
    assert not is_halted()


def test_breach_then_clean_resets_streak(repo, monkeypatch):
    """A clean session between breaches resets the streak, so two NON-consecutive
    breaches must not halt."""
    _write_reference()
    shares = iter([0.95, 0.0, 0.95])
    monkeypatch.setattr(drift_mod, "_evidently_drift",
                        lambda ref, cur, d: (next(shares),
                                             drift_mod._drift_dir() / f"{d.isoformat()}.html"))
    for i, dd in enumerate((date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3))):
        _write_signals(dd, seed=10 + i)
        run_check(dd)
    assert not is_halted(), "a clean session must reset the breach streak"


def test_real_evidently_clean_no_breach(repo):
    """End-to-end through real Evidently: reference and current drawn from the
    SAME distribution → drifted-column share below the alert threshold, so no
    breach. Proves the share extraction + thresholding actually works (not just
    the mocked path)."""
    _write_reference(seed=7)
    day = date(2026, 4, 1)
    _write_signals(day, shift=0.0, seed=7)        # same distribution as reference
    summary = run_check(day)
    assert summary["checked"] is True
    assert summary["share_drifted"] < load_config()["drift"]["share_drifted_columns_alert"]
    assert summary["breach"] is False
    assert not is_halted()
