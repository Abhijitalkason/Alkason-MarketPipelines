"""Per-horizon cost model for the daily system (Milestone-2 only — the listing
itself ships on predictive quality, costs gate the 'actionable' label).

Two profiles:
  delivery  (swing): STT 0.1% on BOTH sides (delivery equity), exchange/
            SEBI/stamp/GST, a flat DP charge on the sell, and NO slippage/impact —
            we do not model microstructure at daily fills. This is the cheaper
            stack the plan calls for (vs the ~0.26% intraday drag that killed EV).
  intraday  (intraday_open): delegates to src/intraday/costs.py — the FULL stack
            including half-spread + size impact (kept honest for that horizon).

Volume-independent for delivery except the flat brokerage/DP terms, which need a
notional → qty is derived from a capital-per-trade like the intraday backtester.
"""

from __future__ import annotations

from src.daily import horizon_config, load_daily_config


def _delivery_round_trip_pct(price: float, qty: int) -> float:
    c = load_daily_config()["costs"]
    notional = price * qty
    if notional <= 0:
        raise ValueError("notional must be positive")
    brokerage = 2 * c["brokerage_delivery_per_order_inr"]
    stt = 2 * notional * c["stt_delivery_pct"]            # buy + sell (delivery)
    exchange = 2 * notional * c["exchange_txn_pct"]
    sebi = 2 * notional * c["sebi_fee_pct"]
    stamp = notional * c["stamp_buy_pct"]                 # buy side only
    gst = c["gst_pct"] * (brokerage + exchange)
    dp = c["dp_charge_sell_inr"]                          # flat, sell side
    total = brokerage + stt + exchange + sebi + stamp + gst + dp
    return total / notional


def round_trip_cost_pct(horizon: str, price: float, qty: int,
                        median_1min_volume: float | None = None, stress: bool = False) -> float:
    """Round-trip cost as a fraction of notional for `horizon`'s cost profile.
    stress applies the configured multiplier (delivery: scales the whole pct;
    intraday: scales slippage per the intraday model)."""
    profile = horizon_config(horizon)["costs_profile"]
    if profile == "intraday":
        from src.intraday.costs import round_trip_cost_pct as intraday_rt
        if median_1min_volume is None:
            raise ValueError("intraday cost profile requires median_1min_volume")
        return intraday_rt(price, qty, median_1min_volume, stress=stress)
    pct = _delivery_round_trip_pct(price, qty)
    if stress:
        pct *= load_daily_config()["gate"]["milestone2"]["stress_multiplier"]
    return pct
