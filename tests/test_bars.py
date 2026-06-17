"""Bar store: validation raises per defect, resample semantics, ATR trailing +
session-boundary mask, provisional exclusion."""

from __future__ import annotations

import pandas as pd
import pytest

from src.intraday.bars import (BarValidationError, atr_2h, load_1min, resample,
                               validate_1min)


def test_load_and_resample(repo, frozen_day):
    df = load_1min("AAA", frozen_day, frozen_day)
    assert not df.empty and df.index.is_monotonic_increasing
    df5 = resample(df, "5min")
    assert (df5.index.minute % 5 == 0).all()


def test_validation_raises_on_non_positive(repo, frozen_day):
    df = load_1min("AAA", frozen_day, frozen_day)
    df.iloc[0, df.columns.get_loc("close")] = -1
    with pytest.raises(BarValidationError):
        validate_1min("AAA", df)


def test_validation_raises_on_non_monotonic(repo, frozen_day):
    df = load_1min("AAA", frozen_day, frozen_day)
    df = df.iloc[::-1]
    with pytest.raises(BarValidationError):
        validate_1min("AAA", df)


def test_atr_strictly_trailing(repo, frozen_day):
    """ATR at bar t is unchanged by mutating data at/after t (property)."""
    from datetime import timedelta
    df = load_1min("AAA", frozen_day - timedelta(days=15), frozen_day)
    df5 = resample(df, "5min")
    atr = atr_2h(df5)
    mid = len(df5) // 2
    df5_mut = df5.copy()
    df5_mut.iloc[mid:, df5_mut.columns.get_loc("high")] *= 5  # blow up the future
    atr_mut = atr_2h(df5_mut)
    pd.testing.assert_series_equal(atr.iloc[:mid], atr_mut.iloc[:mid])


def test_official_wins_over_provisional(repo, frozen_day):
    """Provisional rows never overwrite official rows for the same ts."""
    from src.intraday.data_feed import save_monthly
    extra = load_1min("AAA", frozen_day, frozen_day).reset_index()[["ts", "open", "high", "low", "close", "volume"]]
    extra["close"] = 99999.0
    save_monthly("AAA", extra, provisional=True)            # collides with official ts
    df = load_1min("AAA", frozen_day, frozen_day, include_provisional=True)
    assert (df["close"] != 99999.0).all()                   # official survived


def test_provisional_only_symbol_excluded_by_default(repo, frozen_day):
    """A symbol with ONLY provisional bars is invisible to the default (training)
    load but visible with include_provisional."""
    from src.intraday.data_feed import save_monthly
    from src.intraday.bars import BarValidationError
    bars = load_1min("AAA", frozen_day, frozen_day).reset_index()[
        ["ts", "open", "high", "low", "close", "volume"]]
    save_monthly("PROV", bars, provisional=True)
    with pytest.raises(BarValidationError):
        load_1min("PROV", frozen_day, frozen_day)           # default excludes provisional → empty
    df = load_1min("PROV", frozen_day, frozen_day, include_provisional=True)
    assert df["provisional"].all()
