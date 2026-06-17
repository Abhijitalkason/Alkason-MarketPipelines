"""Blend: weight pinned when flow inactive, isotonic monotone, unfitted raises."""

from __future__ import annotations

import numpy as np
import pytest

from src.intraday.blend import Blender


def test_weight_pinned_when_flow_inactive(repo):
    b = Blender(flow_active=False)
    assert b.weight == 1.0
    b.fit_weight(np.array([0.6, 0.4]), None, np.array([1, 0]))
    assert b.weight == 1.0  # unchanged


def test_blend_ignores_flow_when_inactive(repo):
    b = Blender(flow_active=False)
    p_price = np.array([0.7, 0.3])
    out = b.blend(p_price, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(out, p_price)


def test_isotonic_monotone_and_unfitted_raises(repo):
    b = Blender(flow_active=False)
    with pytest.raises(RuntimeError):
        b.calibrated(np.array([0.5]))
    p = np.linspace(0, 1, 50)
    y = (p > 0.5).astype(int)
    b.fit_calibration(p, y)
    cal = b.calibrated(p)
    assert np.all(np.diff(cal) >= -1e-9)  # non-decreasing


def test_flow_active_weight_grid(repo):
    b = Blender(flow_active=True)
    # price perfectly separates, flow is noise → weight should favour price
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    p_price = np.clip(y * 0.9 + 0.05 + rng.normal(0, 0.02, 200), 0, 1)
    p_flow = rng.uniform(0, 1, 200)
    w = b.fit_weight(p_price, p_flow, y)
    assert w >= 0.5
