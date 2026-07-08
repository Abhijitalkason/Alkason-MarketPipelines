"""Hand-computed checks for the PLAN_v6 §4.4 delivery cost stack (Upstox).

Constants below mirror config_v3.yaml v6.delivery_costs [VERIFIED 2026-07-08
upstox.com/brokerage-charges]; the config-consistency test guards drift.
"""

import pytest

from src.intraday import load_config
from src.v6 import costs

CFG = {
    "brokerage_per_order_inr": 20,
    "stt_pct": 0.001,
    "exchange_txn_pct": 0.0000307,
    "sebi_fee_pct": 0.000001,
    "stamp_buy_pct": 0.00015,
    "ipft_pct": 1.0e-9,
    "dp_sell_flat_inr": 20,
    "gst_pct": 0.18,
    "open_slippage_pct_per_side": 0.0005,
    "slippage_stress_multiplier": 2.0,
}


def test_round_trip_at_1lakh_hand_computed():
    # ₹1,00,000 one-side notional, base slippage:
    #   brokerage 2×20 = 40; STT 2×0.001×1e5 = 200; exchange 2×0.0000307×1e5 = 6.14
    #   SEBI 2×0.000001×1e5 = 0.20; IPFT 2×1e-9×1e5 = 0.0002; stamp 0.00015×1e5 = 15
    #   DP 20; GST 0.18×(40 + 6.14 + 0.0002 + 20) = 11.9052
    #   slippage 2×0.0005×1e5 = 100
    c = costs.round_trip_costs(100_000, cfg=CFG)
    assert c.brokerage_inr == 40
    assert c.dp_inr == 20
    assert c.slippage_inr == pytest.approx(100)
    assert c.statutory_inr == pytest.approx(200 + 6.14 + 0.20 + 0.0002 + 15 + 11.9052, abs=1e-3)
    assert c.total_inr == pytest.approx(393.25, abs=0.01)
    assert costs.round_trip_cost_pct(100_000, cfg=CFG) == pytest.approx(0.0039325, abs=1e-6)


def test_plan_band_at_reference_size():
    # PLAN_v6 §2: delivery round-trip c ≈ 0.28–0.35% at ≥₹1L, ex-slippage.
    pct_ex_slip = (costs.round_trip_costs(100_000, cfg=CFG).total_inr - 100.0) / 100_000
    assert 0.0028 <= pct_ex_slip <= 0.0035


def test_stress_doubles_slippage_only():
    base = costs.round_trip_costs(50_000, cfg=CFG)
    stressed = costs.round_trip_costs(50_000, stress=True, cfg=CFG)
    assert stressed.slippage_inr == pytest.approx(2 * base.slippage_inr)
    assert stressed.statutory_inr == pytest.approx(base.statutory_inr)


def test_minimum_viable_size_floor():
    # flat = (40+20)×1.18 = 70.8; variable ex-slip ≈ 0.22246%/round trip
    # → floor ≈ 70.8 / 0.0022246 ≈ ₹31.8k — inside the plan's indicative ₹25–50k
    floor = costs.minimum_viable_size_inr(cfg=CFG)
    assert floor == pytest.approx(70.8 / 0.0022246, rel=1e-3)
    assert 25_000 <= floor <= 50_000


def test_flat_fee_dominates_below_floor_and_not_above():
    floor = costs.minimum_viable_size_inr(cfg=CFG)
    flat = costs.flat_fee_inr(cfg=CFG)
    for size, should_dominate in [(floor * 0.5, True), (floor * 2, False)]:
        variable = (costs.round_trip_costs(size, cfg=CFG).total_inr
                    - flat - 2 * size * CFG["open_slippage_pct_per_side"])
        assert (flat > variable) is should_dominate


def test_rejects_nonpositive_notional():
    with pytest.raises(ValueError):
        costs.round_trip_costs(0, cfg=CFG)


def test_config_matches_hand_constants():
    # Guards against config drift away from the verified schedule.
    assert load_config()["v6"]["delivery_costs"] == CFG
