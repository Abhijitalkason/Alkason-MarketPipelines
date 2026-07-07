"""PLAN_v4 §4 — labeler honors the geometry override for EVERY tuned key, and
the 80% geometry gives the expected floor. Regression guard for the latent bug
where squareoff / time_barrier / atr_window were read from config, silently
ignoring the geometry_study override (which broke any grid that swept them)."""

from __future__ import annotations

from datetime import datetime, time

from src.intraday.labeler import geometric_baseline, label_day


def test_v4_geometric_baseline_is_80pct():
    assert geometric_baseline({"target_atr": 0.75, "stop_atr": 3.0}) == 3.0 / 3.75


def test_label_day_honors_time_barrier_override(repo, frozen_day):
    """A tight 1-hour time-barrier override must cap exits an hour after entry —
    proving label_day reads t_barrier_h from the override, not config."""
    geo = {"target_atr": 5.0, "stop_atr": 5.0,      # barriers far away → time exits
           "time_barrier_hours": 1, "squareoff": "14:45",
           "atr_window_minutes": 120, "entry_window": ["09:30", "11:00"]}
    outs = label_day("AAA", frozen_day, geometry=geo)
    assert outs, "expected labelled bars"
    for o in outs:
        # exit no later than entry + 1h (the overridden barrier), never the 3h/2h default
        assert (o.exit_ts - o.entry_ts).total_seconds() <= 3600 + 1


def test_label_day_honors_squareoff_override(repo, frozen_day):
    geo = {"target_atr": 5.0, "stop_atr": 5.0,
           "time_barrier_hours": 6, "squareoff": "10:30",   # early square-off cap
           "atr_window_minutes": 120, "entry_window": ["09:30", "11:00"]}
    outs = label_day("AAA", frozen_day, geometry=geo)
    assert outs
    for o in outs:
        assert o.exit_ts <= datetime.combine(frozen_day, time(10, 30))
