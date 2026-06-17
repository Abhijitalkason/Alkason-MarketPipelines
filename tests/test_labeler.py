"""Labeler: dual-touch → stop, time-barrier sign, squareoff cap, entry = next bar."""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd

from src.intraday.labeler import _resolve_path, geometric_baseline, label_day


def _bar(ts, o, h, l, c, v=1000):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_dual_touch_counts_as_stop():
    idx = pd.date_range("2026-01-05 10:00", periods=1, freq="1min")
    path = pd.DataFrame([_bar(idx[0], 100, 110, 90, 100)], index=idx)
    _, price, barrier = _resolve_path(path, direction=1, entry=100, target=105, stop=95)
    assert barrier == "stop" and price == 95


def test_target_first():
    idx = pd.date_range("2026-01-05 10:00", periods=2, freq="1min")
    path = pd.DataFrame([_bar(idx[0], 100, 106, 99, 105), _bar(idx[1], 105, 106, 104, 105)], index=idx)
    _, price, barrier = _resolve_path(path, direction=1, entry=100, target=105, stop=90)
    assert barrier == "target" and price == 105


def test_time_barrier_sign():
    idx = pd.date_range("2026-01-05 10:00", periods=2, freq="1min")
    path = pd.DataFrame([_bar(idx[0], 100, 101, 99, 100), _bar(idx[1], 100, 102, 99, 101.5)], index=idx)
    _, price, barrier = _resolve_path(path, direction=1, entry=100, target=110, stop=90)
    assert barrier == "time"  # neither barrier hit


def test_geometric_baseline():
    assert geometric_baseline({"target_atr": 0.4, "stop_atr": 3.2}) == 3.2 / 3.6


def test_label_day_entry_is_next_bar(repo, frozen_day):
    outs = label_day("AAA", frozen_day)
    assert outs, "expected at least one labelled decision bar"
    for o in outs:
        assert o.entry_ts.time() >= time(9, 45)   # first decision 09:30 → entry 09:45
        assert o.exit_ts <= datetime.combine(frozen_day, time(14, 45))
