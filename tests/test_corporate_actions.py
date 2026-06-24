"""Corporate actions (v3_supplementry §3.5, H-2).

Upstox 1-min candles are ALREADY corp-action back-adjusted at the source, so the
stored series is the adjusted series — load_1min(adjusted=True) returns it
unchanged. The factor table is used the OTHER direction: to_as_traded UN-adjusts
(divides pre-ex-date prices by the cumulative factor) to recover as-traded prices
for the bhavcopy cross-check. These tests pin: purpose parsing for splits/bonuses,
continuity of the un-adjustment across a 1:2 split ex-date, and the empty-table
no-op. Each would fail if the control were removed.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from src.intraday import ROOT, load_config
from src.intraday.bars import load_1min
from src.intraday.corporate_actions import (load_factors, parse_purpose, to_as_traded)


def _ca_path():
    return ROOT / load_config()["data"]["reference_path"] / "corporate_actions.csv"


def test_parse_purpose_split():
    # 1:2 split = face value Rs.10 → Rs.5 → factor 0.5 (post/pre price ratio)
    assert parse_purpose("FACE VALUE SPLIT FRM RS.10/- TO RS.5/-") == pytest.approx(0.5)
    # 5:1 split Rs.5 → Rs.1 → factor 0.2
    assert parse_purpose("SPLIT FRM RS 5 TO RS 1") == pytest.approx(0.2)


def test_parse_purpose_bonus():
    # 1:1 bonus → base/(base+bonus) = 1/2 = 0.5
    assert parse_purpose("BONUS 1:1") == pytest.approx(0.5)
    # 1:2 bonus → 2/(2+1) = 0.6667
    assert parse_purpose("BONUS 1:2") == pytest.approx(2 / 3)


def test_parse_purpose_non_price_action_is_none():
    # dividends are intraday-irrelevant and must NOT produce a factor — otherwise
    # to_as_traded would scale prices on every dividend ex-date.
    assert parse_purpose("DIVIDEND - RS 12 PER SHARE") is None
    assert parse_purpose("ANNUAL GENERAL MEETING") is None
    # a "split" that doesn't reduce the face value is rejected (guards the sign).
    assert parse_purpose("FRM RS.5 TO RS.10") is None


def test_to_as_traded_unadjusts_across_split(repo):
    """A 1:2 split (factor 0.5) on a mid-window ex-date: bars STRICTLY BEFORE the
    ex-date are the back-adjusted (halved) series, so un-adjusting must DIVIDE
    them by 0.5 (i.e. double them) to recover the as-traded ~2x level. Bars on/
    after the ex-date are untouched. Volume scales the opposite way."""
    sym = "AAA"
    load_factors.cache_clear()
    # pick an ex-date inside the loaded window with sessions on both sides
    days = repo["days"]
    ex_date = days[len(days) // 2]
    while ex_date.weekday() == 3:  # avoid the sparse expiry days
        ex_date = days[days.index(ex_date) + 1]
    _ca_path().write_text(
        "symbol,ex_date,purpose,factor\n"
        f"{sym},{ex_date.isoformat()},FACE VALUE SPLIT FRM RS.10 TO RS.5,0.5\n"
    )
    load_factors.cache_clear()

    start = days[len(days) // 2 - 5]
    end = days[len(days) // 2 + 5]
    raw = load_1min(sym, start, end, adjusted=True)          # stored = adjusted
    as_traded = load_1min(sym, start, end, adjusted=False)   # un-adjusted

    import numpy as np
    pre = raw.index.normalize() < pd.Timestamp(ex_date)
    post = ~pre
    assert pre.any() and post.any(), "need bars on both sides of the ex-date"

    # pre-ex-date as-traded prices = adjusted / 0.5 = 2× the stored level
    assert np.allclose(as_traded.loc[pre, "close"].to_numpy(),
                       raw.loc[pre, "close"].to_numpy() / 0.5)
    # post-ex-date prices unchanged
    assert np.allclose(as_traded.loc[post, "close"].to_numpy(),
                       raw.loc[post, "close"].to_numpy())
    # volume goes the other way (multiplied by the factor pre-ex-date)
    assert (as_traded.loc[pre, "volume"] <= raw.loc[pre, "volume"]).all()
    # close_adj keeps the adjusted series for downstream consumers
    assert np.allclose(as_traded["close_adj"].to_numpy(), raw["close"].to_numpy())
    load_factors.cache_clear()


def test_to_as_traded_continuity_jump_at_ex_date(repo):
    """The un-adjustment must REMOVE the artificial back-adjustment discontinuity:
    in the as-traded series the price level jumps by ~1/factor across the ex-date
    boundary (real split gap), while WITHOUT the control the adjusted series is
    artificially continuous. This is the property that would silently break if
    to_as_traded stopped applying the factor."""
    sym = "AAA"
    days = repo["days"]
    ex_date = days[len(days) // 2]
    while ex_date.weekday() == 3:
        ex_date = days[days.index(ex_date) + 1]
    _ca_path().write_text(
        "symbol,ex_date,purpose,factor\n"
        f"{sym},{ex_date.isoformat()},SPLIT FRM RS.10 TO RS.5,0.5\n"
    )
    load_factors.cache_clear()

    start = days[len(days) // 2 - 3]
    end = days[len(days) // 2 + 3]
    raw = load_1min(sym, start, end, adjusted=True)
    at = load_1min(sym, start, end, adjusted=False)

    # last bar before ex-date vs first bar on/after
    pre_idx = raw.index[raw.index.normalize() < pd.Timestamp(ex_date)][-1]
    post_idx = raw.index[raw.index.normalize() >= pd.Timestamp(ex_date)][0]

    # adjusted series: roughly continuous (no split jump) — ratio near 1
    adj_ratio = raw.loc[pre_idx, "close"] / raw.loc[post_idx, "close"]
    # as-traded series: pre price is ~2× larger → ratio near 2
    at_ratio = at.loc[pre_idx, "close"] / at.loc[post_idx, "close"]
    assert at_ratio == pytest.approx(adj_ratio / 0.5, rel=1e-9)
    assert abs(at_ratio - 1.0) > abs(adj_ratio - 1.0)  # the split jump reappears
    load_factors.cache_clear()


def test_empty_table_is_noop(repo):
    """No corporate actions → to_as_traded returns prices unchanged (interface
    columns added, no scaling). Empty/header-only table is valid."""
    sym = "BBB"
    _ca_path().write_text("symbol,ex_date,purpose,factor\n")
    load_factors.cache_clear()
    assert load_factors(sym) == ()

    df = pd.DataFrame({
        "open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0],
        "close": [100.5, 101.5], "volume": [1000, 1100],
    }, index=pd.to_datetime([datetime.combine(date(2025, 1, 6), time(9, 15)),
                             datetime.combine(date(2025, 1, 6), time(9, 16))]))
    out = to_as_traded(df, sym)
    assert (out["close"] == df["close"]).all()
    assert (out["volume"] == df["volume"]).all()
    assert (out["unadj_factor"] == 1.0).all()
    assert (out["close_adj"] == df["close"]).all()
    load_factors.cache_clear()
