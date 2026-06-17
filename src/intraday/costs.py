"""Indian intraday equity cost stack — single source of truth (PLAN_v3 §12.1;
v3_supplementry §7).

All cost logic lives here; backtester, paper runner, and gates import from
this module only. I/O-free by design (median_1min_volume is passed in) so the
math is unit-testable against hand-computed examples.

Slippage model (H-7): per side = half-spread + impact, where
    impact = costs.impact_coeff × (qty / median_1min_volume)
Under stress both components scale by slippage_stress_multiplier.
entry_is_quoted=True (paper/live fills at quoted bid/ask): the entry side
charges impact only — the spread is already paid in the fill price (M-1
double-count fix).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.intraday import load_config


@dataclass(frozen=True)
class TradeCosts:
    statutory_inr: float   # STT + exchange + SEBI + stamp + GST
    brokerage_inr: float
    slippage_inr: float

    @property
    def total_inr(self) -> float:
        return self.statutory_inr + self.brokerage_inr + self.slippage_inr


def slippage_pct_per_side(qty: int, median_1min_volume: float,
                          stress: bool = False, include_spread: bool = True) -> float:
    """Half-spread + size-impact slippage, as a fraction of notional, one side."""
    c = load_config()["costs"]
    if median_1min_volume <= 0:
        raise ValueError("median_1min_volume must be positive (liquidity floor upstream)")
    half_spread = c["slippage_half_spread_pct"] if include_spread else 0.0
    impact = c["impact_coeff"] * (qty / median_1min_volume)
    mult = c["slippage_stress_multiplier"] if stress else 1.0
    return (half_spread + impact) * mult


def round_trip_costs(price: float, qty: int, median_1min_volume: float,
                     stress: bool = False, entry_is_quoted: bool = False) -> TradeCosts:
    """Total round-trip (buy + sell) cost for an intraday equity trade.

    Components (config: costs.*):
      brokerage  — flat per order, two orders
      STT        — sell side only (intraday equity)
      exchange   — both sides
      SEBI fee   — both sides
      stamp duty — buy side only
      GST        — 18% on brokerage + exchange txn charges
      slippage   — half-spread + impact per side; 2× under stress
    """
    c = load_config()["costs"]
    notional = price * qty
    brokerage = 2 * c["brokerage_per_order_inr"]
    stt = notional * c["stt_sell_pct"]
    exchange = 2 * notional * c["exchange_txn_pct"]
    sebi = 2 * notional * c["sebi_fee_pct"]
    stamp = notional * c["stamp_buy_pct"]
    gst = c["gst_pct"] * (brokerage + exchange)
    entry_slip = slippage_pct_per_side(qty, median_1min_volume, stress,
                                       include_spread=not entry_is_quoted)
    exit_slip = slippage_pct_per_side(qty, median_1min_volume, stress)
    return TradeCosts(
        statutory_inr=stt + exchange + sebi + stamp + gst,
        brokerage_inr=brokerage,
        slippage_inr=notional * (entry_slip + exit_slip),
    )


def round_trip_cost_pct(price: float, qty: int, median_1min_volume: float,
                        stress: bool = False, entry_is_quoted: bool = False) -> float:
    """Round-trip cost as a fraction of traded notional (the `c` in EV = δ·W − c)."""
    return round_trip_costs(price, qty, median_1min_volume, stress,
                            entry_is_quoted).total_inr / (price * qty)
