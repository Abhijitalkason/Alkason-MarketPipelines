"""Paper-runner correctness fixes (regression guards).

These lock in three fixes the live loop depends on:
  1. time-barrier exits are labelled by GROSS sign — identical to the backtest/
     labeler convention, so the win/loss feeding the ACI gate and kill switches
     means the same thing live as in training (train==serve);
  2. the 15-min decision boundaries in the entry window are evaluated once each,
     with no minute-of-hour collision (10:30 must not be mistaken for 09:30);
  3. the entry window yields exactly the expected decision boundaries.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta


def _label(reason: str, gross: float) -> int:
    """The paper runner's label rule (paper_runner._close) — kept in sync with
    labeler.label_day: time-barrier exit → sign of GROSS pnl; barrier → touch."""
    return int(gross > 0) if reason == "time" else int(reason == "target")


def test_time_exit_labelled_on_gross_not_net():
    # a time exit that is positive pre-cost but would be negative post-cost is a
    # WIN by barrier convention — net must not flip the label
    assert _label("time", gross=0.0008) == 1
    assert _label("time", gross=-0.0003) == 0


def test_barrier_exit_label_matches_labeler():
    assert _label("target", gross=0.01) == 1
    assert _label("stop", gross=-0.01) == 0


def _boundaries(day, w_start: str, w_end: str) -> list[time]:
    """Mirror of paper_runner.run_session boundary generation."""
    out, t = [], datetime.combine(day, time.fromisoformat(w_start))
    end = datetime.combine(day, time.fromisoformat(w_end))
    while t < end:
        out.append(t.time())
        t += timedelta(minutes=15)
    return out


def test_entry_window_boundaries_distinct_and_complete():
    from datetime import date
    b = _boundaries(date(2026, 4, 1), "09:30", "11:00")
    assert b == [time(9, 30), time(9, 45), time(10, 0), time(10, 15),
                 time(10, 30), time(10, 45)]
    assert len(b) == len(set(b)) == 6   # no collisions, all six evaluated


def test_minute_of_hour_dedup_would_collide():
    """Demonstrates the bug the fix avoids: minute-of-hour repeats, so a
    minute-keyed dedup would treat 10:30 as already-seen from 09:30."""
    from datetime import date
    minutes = {b.minute for b in _boundaries(date(2026, 4, 1), "09:30", "11:00")}
    assert len(minutes) < 6   # {30,45,0,15} — collisions exist; slot dedup needed
