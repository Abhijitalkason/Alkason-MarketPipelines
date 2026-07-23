"""NSE equity DELIVERY (CNC) cost stack — single source of truth (PLAN_v6 §4.4).

Broker: Upstox (user decision 2026-07-08). Every constant lives in
config_v3.yaml `v6.delivery_costs` with its source cited there
[VERIFIED 2026-07-08 upstox.com/brokerage-charges]. I/O-free by design
(the config dict is injectable) so the math is unit-testable against
hand-computed examples.

Differences from the intraday stack (src/intraday/costs.py):
  - STT on BOTH sides at 0.1% (delivery), not sell-only 0.025%
  - DP charge: flat per scrip per sell day (one exit per round trip here)
  - GST base per the Upstox schedule: brokerage + exchange txn + IPFT + DP
  - slippage is a flat per-side constant, PROVISIONAL until the §4.4
    open-print study (unknown #15) replaces it with the measured
    open-vs-first-15min-VWAP distribution

F21 ordering rule: this module and the c(size) curve it powers are frozen
BEFORE any geometry cell is evaluated, so no cell can be selected that only
works for institutional capital.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.intraday import load_config


@dataclass(frozen=True)
class DeliveryCosts:
    statutory_inr: float   # STT + exchange + SEBI + stamp + IPFT + GST
    brokerage_inr: float   # 2 × flat per-order
    dp_inr: float          # flat per scrip per sell day (GST counted in statutory)
    slippage_inr: float

    @property
    def total_inr(self) -> float:
        return self.statutory_inr + self.brokerage_inr + self.dp_inr + self.slippage_inr


def _cfg() -> dict:
    return load_config()["v6"]["delivery_costs"]


def round_trip_costs(notional_inr: float, stress: bool = False,
                     cfg: dict | None = None) -> DeliveryCosts:
    """Total round-trip (buy + hold overnight + sell) cost for a CNC trade.

    One buy order, one sell order, one scrip-day of DP on the exit.
    `notional_inr` is the per-side traded value; buy and sell notionals are
    treated as equal (the PnL difference is second-order for the cost model).
    """
    c = cfg or _cfg()
    if notional_inr <= 0:
        raise ValueError("notional_inr must be positive")
    brokerage = 2 * c["brokerage_per_order_inr"]
    stt = 2 * notional_inr * c["stt_pct"]
    exchange = 2 * notional_inr * c["exchange_txn_pct"]
    sebi = 2 * notional_inr * c["sebi_fee_pct"]
    ipft = 2 * notional_inr * c["ipft_pct"]
    stamp = notional_inr * c["stamp_buy_pct"]
    dp = c["dp_sell_flat_inr"]
    gst = c["gst_pct"] * (brokerage + exchange + ipft + dp)
    slip_mult = c["slippage_stress_multiplier"] if stress else 1.0
    slippage = 2 * notional_inr * c["open_slippage_pct_per_side"] * slip_mult
    return DeliveryCosts(
        statutory_inr=stt + exchange + sebi + ipft + stamp + gst,
        brokerage_inr=brokerage,
        dp_inr=dp,
        slippage_inr=slippage,
    )


def round_trip_cost_pct(notional_inr: float, stress: bool = False,
                        cfg: dict | None = None) -> float:
    """Round-trip cost as a fraction of one-side notional (the `c` in δ = c/W)."""
    return round_trip_costs(notional_inr, stress, cfg).total_inr / notional_inr


def flat_fee_inr(cfg: dict | None = None) -> float:
    """Size-independent rupee drag per round trip: brokerage + DP + GST on both."""
    c = cfg or _cfg()
    return (2 * c["brokerage_per_order_inr"] + c["dp_sell_flat_inr"]) * (1 + c["gst_pct"])


def variable_cost_pct(stress: bool = False, cfg: dict | None = None) -> float:
    """Asymptotic (size-proportional) round-trip cost fraction, ex flat fees."""
    c = cfg or _cfg()
    slip_mult = c["slippage_stress_multiplier"] if stress else 1.0
    return (
        2 * c["stt_pct"]
        + 2 * c["exchange_txn_pct"] * (1 + c["gst_pct"])
        + 2 * c["sebi_fee_pct"]
        + 2 * c["ipft_pct"] * (1 + c["gst_pct"])
        + c["stamp_buy_pct"]
        + 2 * c["open_slippage_pct_per_side"] * slip_mult
    )


def minimum_viable_size_inr(cfg: dict | None = None) -> float:
    """Pre-registered size floor (§4.4/F21): the size at which the flat-fee
    component equals the ex-slippage variable cost — below it, flat fees
    dominate and the trade is plumbing-test-only, excluded from GO/KILL.

    Slippage is excluded from the variable side so the floor cannot move when
    the provisional slippage constant is replaced by the measured one
    (unknown #15) — the floor stays frozen per F21.
    """
    c = cfg or _cfg()
    variable_ex_slip = (
        2 * c["stt_pct"]
        + 2 * c["exchange_txn_pct"] * (1 + c["gst_pct"])
        + 2 * c["sebi_fee_pct"]
        + 2 * c["ipft_pct"] * (1 + c["gst_pct"])
        + c["stamp_buy_pct"]
    )
    return flat_fee_inr(c) / variable_ex_slip
