"""ACI gate: update direction, clip bounds, persistence, history, coverage CI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config
from src.intraday.conformal_gate import ACIGate, GateState, coverage_report


def test_err_above_alpha_raises_tau(repo):
    g = ACIGate(GateState(tau=0.90, alpha=0.11, gamma=0.05))
    before = g.state.tau
    g.update([0, 0, 0, 0])          # err=1.0 ≫ alpha → τ must rise (fewer signals)
    assert g.state.tau > before


def test_err_below_alpha_lowers_tau(repo):
    g = ACIGate(GateState(tau=0.90, alpha=0.11, gamma=0.05))
    before = g.state.tau
    g.update([1, 1, 1, 1])          # err=0 < alpha → τ relaxes
    assert g.state.tau < before


def test_tau_clipped(repo):
    g = ACIGate(GateState(tau=0.994, alpha=0.11, gamma=0.5))
    for _ in range(50):
        g.update([0])
    assert g.state.tau <= 0.995


def test_no_signals_no_update(repo):
    g = ACIGate(GateState(tau=0.90, alpha=0.11, gamma=0.05))
    assert g.update([]) == 0.90


def test_persistence_roundtrip_and_history(repo):
    g = ACIGate(GateState(tau=0.91, alpha=0.11, gamma=0.01))
    g.update([1, 0])
    g.save()
    g2 = ACIGate.load()
    assert g2.state.tau == g.state.tau
    hist = ROOT / load_config()["paths"]["state"] / "gate_state_history.jsonl"
    assert hist.exists() and len(hist.read_text().splitlines()) >= 1


def test_fire_requires_floor(repo):
    g = ACIGate(GateState(tau=0.5, alpha=0.11, gamma=0.01))
    p_cal = np.array([0.99, 0.99])
    p_price = np.array([0.80, 0.70])  # second below floor 0.75
    mask = g.fire(p_cal, p_price)
    assert mask[0] and not mask[1]


def test_coverage_report_ci(repo):
    trades = pd.DataFrame({"symbol": ["AAA"] * 40, "label": [1] * 38 + [0] * 2})
    rep = coverage_report(trades, min_n=30)
    assert "AAA" in rep
    assert 0 <= rep["AAA"]["ci_low"] <= rep["AAA"]["win_rate"] <= rep["AAA"]["ci_high"] <= 1
