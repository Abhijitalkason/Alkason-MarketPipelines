"""Cost stack: statutory math, impact scaling, quoted-entry single-spread, stress."""

from __future__ import annotations

import pytest

from src.intraday import load_config
from src.intraday.costs import round_trip_cost_pct, round_trip_costs, slippage_pct_per_side


def test_statutory_components(repo):
    c = round_trip_costs(price=1000.0, qty=100, median_1min_volume=10_000)
    cfg = load_config()["costs"]
    notional = 1000.0 * 100
    expected_stat = (
        notional * cfg["stt_sell_pct"]
        + 2 * notional * cfg["exchange_txn_pct"]
        + 2 * notional * cfg["sebi_fee_pct"]
        + notional * cfg["stamp_buy_pct"]
        + cfg["gst_pct"] * (2 * cfg["brokerage_per_order_inr"] + 2 * notional * cfg["exchange_txn_pct"])
    )
    assert c.brokerage_inr == 2 * cfg["brokerage_per_order_inr"]
    assert c.statutory_inr == pytest.approx(expected_stat)


def test_impact_scales_with_size(repo):
    small = slippage_pct_per_side(qty=100, median_1min_volume=10_000)
    big = slippage_pct_per_side(qty=1000, median_1min_volume=10_000)
    assert big > small  # impact term grows with qty/volume


def test_quoted_entry_drops_one_spread(repo):
    full = round_trip_cost_pct(1000.0, 100, 10_000, entry_is_quoted=False)
    quoted = round_trip_cost_pct(1000.0, 100, 10_000, entry_is_quoted=True)
    assert quoted < full  # entry spread already paid in the fill price


def test_stress_doubles_slippage(repo):
    base = slippage_pct_per_side(100, 10_000, stress=False)
    stress = slippage_pct_per_side(100, 10_000, stress=True)
    mult = load_config()["costs"]["slippage_stress_multiplier"]
    assert stress == pytest.approx(base * mult)


def test_zero_volume_raises(repo):
    with pytest.raises(ValueError):
        slippage_pct_per_side(100, 0)
