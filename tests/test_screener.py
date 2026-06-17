"""Screener: PIT inputs, liquidity floor, fit_weights recovers planted weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intraday import load_config
from src.intraday.screener import SCORE_COMPONENTS, ScreenError, fit_weights, screen_day


def test_screen_produces_candidates(repo, frozen_day):
    top = screen_day(frozen_day)
    assert not top.empty
    assert set(["symbol", "score", "direction"]).issubset(top.columns)
    assert len(top) <= load_config()["screen"]["top_n"]


def test_liquidity_floor_excludes_thin_names(repo, frozen_day):
    cfg = load_config()
    cfg["universe"]["min_median_1min_volume"] = 10_000_000  # nothing can clear it
    with pytest.raises(ScreenError):
        screen_day(frozen_day)


def test_fit_weights_recovers_signal(repo):
    """Planted: only volume_surge drives wins. Fitted weight on it dominates."""
    rng = np.random.RandomState(0)
    n = 400
    rows = pd.DataFrame({c: rng.normal(0, 1, n) for c in SCORE_COMPONENTS})
    won = (rows["vol_z"] + rng.normal(0, 0.3, n) > 0).astype(int)
    w = fit_weights(rows, won)
    assert abs(w["volume_surge"]) == max(abs(v) for v in w.values())
