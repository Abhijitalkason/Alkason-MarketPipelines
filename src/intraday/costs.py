"""Indian intraday equity cost stack — single source of truth (PLAN_v3 Section 12.1).

All cost logic lives here. The backtester, paper runner, and gates import from
this module only. Stress mode scales slippage by config multiplier.
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


def round_trip_costs(price: float, qty: int, stress: bool = False) -> TradeCosts:
    """Total round-trip (buy + sell) cost for an intraday equity trade.

    Components (config: costs.*):
      brokerage  — flat per order, two orders
      STT        — sell side only (intraday equity)
      exchange   — both sides
      SEBI fee   — both sides
      stamp duty — buy side only
      GST        — 18% on brokerage + exchange txn charges
      slippage   — modelled, both sides; 2x under stress
    """
    c = load_config()["costs"]
    notional = price * qty
    brokerage = 2 * c["brokerage_per_order_inr"]
    stt = notional * c["stt_sell_pct"]
    exchange = 2 * notional * c["exchange_txn_pct"]
    sebi = 2 * notional * c["sebi_fee_pct"]
    stamp = notional * c["stamp_buy_pct"]
    gst = c["gst_pct"] * (brokerage + exchange)
    slip_mult = c["slippage_stress_multiplier"] if stress else 1.0
    slippage = 2 * notional * c["slippage_pct"] * slip_mult
    return TradeCosts(
        statutory_inr=stt + exchange + sebi + stamp + gst,
        brokerage_inr=brokerage,
        slippage_inr=slippage,
    )


def round_trip_cost_pct(price: float, qty: int, stress: bool = False) -> float:
    """Round-trip cost as a fraction of traded notional (the `c` in EV = δ·W − c)."""
    return round_trip_costs(price, qty, stress).total_inr / (price * qty)
