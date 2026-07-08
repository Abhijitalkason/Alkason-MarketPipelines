"""Hand-built path checks for the PLAN_v6 §4.2 daily triple-barrier labeler.

Each test constructs a tiny OHLC series where the correct resolution is
computable by hand; the labeler must reproduce it exactly.
"""

import numpy as np
import pandas as pd
import pytest

from src.v6.labeler import Cell, label_symbol

ATR_PCT = 0.02  # constant 2% ATR for hand arithmetic


def _panel(rows):
    """rows: list of (open, high, low, close). Dates are consecutive weekdays."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["date"] = pd.bdate_range("2024-01-01", periods=len(df))
    df["symbol"] = "TEST"
    df["atr_pct"] = ATR_PCT
    df["locked_day"] = df["high"] == df["low"]
    return df


def test_target_hit_intraday():
    # D close=100, ATR_abs=2. Entry D+1 open=100; a=1 → target 102, b=2 → stop 96.
    # D+1 high 103 touches target, low 99 never nears stop → target exit, ret +2%.
    df = _panel([(100, 101, 99, 100), (100, 103, 99, 101), (101, 102, 100, 101),
                 (101, 102, 100, 101), (101, 102, 100, 101)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=2))
    row = out.iloc[0]
    assert row["exit_type"] == "target" and row["win"]
    assert row["ret"] == pytest.approx(0.02)
    assert row["exit_day_offset"] == 1


def test_stop_hit_intraday():
    # Same geometry; D+1 low 95 pierces stop 96, high 101 below target → stop, −4%.
    df = _panel([(100, 101, 99, 100), (100, 101, 95, 96), (96, 97, 95, 96),
                 (96, 97, 95, 96), (96, 97, 95, 96)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=2))
    row = out.iloc[0]
    assert row["exit_type"] == "stop" and not row["win"]
    assert row["ret"] == pytest.approx(-0.04)
    assert not row["gap_through"]


def test_dual_touch_is_stop():
    # D+1 range [94, 104] spans both barriers → conservative stop.
    df = _panel([(100, 101, 99, 100), (100, 104, 94, 100), (100, 101, 99, 100),
                 (100, 101, 99, 100), (100, 101, 99, 100)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=2))
    row = out.iloc[0]
    assert row["exit_type"] == "stop" and bool(row["dual_touch"])


def test_time_exit_win_and_loss():
    # No barrier touched for H=2 days; exit at D+2 close.
    df = _panel([(100, 101, 99, 100), (100, 101, 99, 100.5), (100.5, 101.5, 99.5, 101),
                 (101, 102, 100, 101), (101, 102, 100, 101)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=2))
    row = out.iloc[0]
    assert row["exit_type"] == "time"
    assert row["ret"] == pytest.approx(0.01) and row["win"]  # 101/100 − 1 > 0


def test_gap_through_stop_fills_at_open():
    # Day D+2 opens at 92, below the 96 stop → fill at the OPEN (−8%), not the stop.
    # loss beyond stop = (96 − 92) / (100 − 96) = 1.0 of the stop distance.
    df = _panel([(100, 101, 99, 100), (100, 101, 98, 99), (92, 93, 91, 92),
                 (92, 93, 91, 92), (92, 93, 91, 92)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=3))
    row = out.iloc[0]
    assert row["exit_type"] == "stop" and bool(row["gap_through"])
    assert row["ret"] == pytest.approx(-0.08)
    assert row["loss_beyond_stop_frac"] == pytest.approx(1.0)


def test_short_mirror_flips_sign():
    # Short side: target 98 (below), stop 104. D+1 low 97 → short target, ret +2%.
    df = _panel([(100, 101, 99, 100), (100, 101, 97, 98), (98, 99, 97, 98),
                 (98, 99, 97, 98), (98, 99, 97, 98)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=2), side="short")
    row = out.iloc[0]
    assert row["exit_type"] == "target" and row["win"]
    assert row["ret"] == pytest.approx(0.02)


def test_entry_day_open_never_gap_exits():
    # Entry-day open IS the entry — even though open 100 ≥ nothing here, ensure
    # a first-day extreme open still resolves intraday, not as a gap fill.
    df = _panel([(100, 101, 99, 100), (100, 100.4, 95.9, 96), (96, 97, 95, 96),
                 (96, 97, 95, 96), (96, 97, 95, 96)])
    out = label_symbol(df, Cell(a=1.0, b=2, h=2))
    row = out.iloc[0]
    assert row["exit_type"] == "stop"
    assert row["ret"] == pytest.approx(-0.04)  # filled at the stop level 96


def test_strictly_trailing_atr_requires_finite():
    # NaN atr on the signal day → that day produces no signal row.
    df = _panel([(100, 101, 99, 100)] * 6)
    df.loc[0, "atr_pct"] = np.nan
    out = label_symbol(df, Cell(a=1.0, b=2, h=2))
    assert (pd.to_datetime(out["signal_date"]) != df.loc[0, "date"]).all()
