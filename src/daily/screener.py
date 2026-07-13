"""Daily listing engine — score the PIT universe, rank, emit the top-N watchlist
with direction + calibrated probability + a short SHAP 'why'. This is the product
deliverable (no order execution).

v1 lists the top-N BUY candidates (highest calibrated P(up), direction LONG). The
short side (bottom-N) is a one-line extension once the long list clears the gate.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.daily import horizon_config, load_daily_config, load_universe, today_ist
from src.daily.features import build_matrix, check_schema
from src.daily.panel import load_symbol_daily
from src.daily.trainer import load_bundle

logger = logging.getLogger(__name__)

# Human-readable phrasing for SHAP 'why' (feature → (label, positive-direction word)).
_WHY = {
    "ret_1d": "1-day momentum", "ret_3d": "3-day momentum", "ret_5d": "5-day momentum",
    "ret_10d": "10-day trend", "ret_20d": "20-day trend",
    "vol_10d": "10-day volatility", "vol_20d": "20-day volatility",
    "rsi_14": "RSI(14)", "ma_dist_20": "distance above 20-DMA", "ma_dist_50": "distance above 50-DMA",
    "atr_pct": "volatility (ATR%)", "pos_52w": "52-week range position",
    "gap_prev": "recent gap", "vol_ratio_20": "volume vs 20-day norm",
    "deliv_per": "delivery %", "deliv_z": "delivery-% trend", "turnover_z": "turnover trend",
    "oi_change_z": "OI change", "basis_pct": "futures basis", "fii_5d_z": "FII 5-day flow",
    "overnight_spx": "overnight S&P", "overnight_ndx": "overnight Nasdaq",
    "gift_nifty_ret": "GIFT Nifty", "asia_ret": "Asian markets",
    "india_vix": "India VIX", "india_vix_chg": "India VIX move",
    "dxy_chg": "USD index", "us10y_chg": "US 10Y", "crude_chg": "crude oil",
    "gold_chg": "gold", "usdinr_chg": "USD/INR",
    "days_to_macro": "upcoming macro event", "days_since_macro": "recent macro event",
    "days_to_earnings": "upcoming earnings", "days_since_earnings": "post-earnings drift",
    "monsoon_departure": "monsoon", "sent_mean_5d": "news sentiment",
    "sent_surprise": "sentiment surprise", "sent_coverage": "news coverage",
}


def _humanize(why: dict[str, float]) -> list[str]:
    out = []
    for feat, val in why.items():
        label = _WHY.get(feat, feat)
        out.append(f"{label} ({'+' if val >= 0 else '-'})")
    return out


def screen_day(day: date | None = None, horizon: str = "next_day",
               top_n: int | None = None, df_panels: dict | None = None) -> pd.DataFrame:
    """Top-N daily picks for `horizon`. Returns columns
    [symbol, p_up, direction, prob, why]. df_panels optionally injects panels (live)."""
    cfg = load_daily_config()
    day = day or today_ist()
    top_n = top_n or cfg["listing"]["top_n"]
    k = cfg["listing"]["why_top_k"]

    pm, blender, meta = load_bundle(horizon)
    uni = load_universe(as_of=day)["symbol"].tolist()
    plan = pd.DataFrame({"symbol": uni, "date": [day] * len(uni)})
    matrix = build_matrix(plan, df_panels=df_panels)
    if matrix.empty:
        logger.warning("daily screen %s %s: no feature rows", day, horizon)
        return pd.DataFrame(columns=["symbol", "p_up", "direction", "prob", "why"])
    check_schema(matrix)
    matrix = matrix.assign(p_up=blender.calibrated(pm.predict_proba(matrix)))

    # Rank by calibrated P(win). At/near no-edge the model's scores tie (a constant
    # output is the honest no-signal answer), so break ties TRANSPARENTLY by trailing
    # momentum (ret_20d, then ret_5d) — never by incidental DataFrame order, which
    # would silently produce an alphabetical "top" list.
    ranked = matrix.sort_values(["p_up", "ret_20d", "ret_5d"], ascending=False)
    top = ranked.head(min(top_n, len(ranked))).reset_index(drop=True)
    hc = horizon_config(horizon)
    capital = float(cfg["listing"].get("capital_per_pick_inr", 100_000))
    rows = []
    for _, r in top.iterrows():
        why = pm.shap_top(pd.DataFrame([r]), n=k)
        rows.append({
            "symbol": r["symbol"], "p_up": float(r["p_up"]),
            "direction": "LONG", "prob": float(r["p_up"]),
            "why": _humanize(why),
            **_trade_plan(r["symbol"], day, float(r["atr_pct"]), hc, capital,
                          (df_panels or {}).get(r["symbol"])),
        })
    return pd.DataFrame(rows)


def _trade_plan(symbol: str, day: date, atr_pct: float, hc: dict, capital: float,
                df_panel: pd.DataFrame | None = None) -> dict:
    """Actionable levels for one pick. Reference price = the AS-TRADED close of the
    decision day (the actual entry is the NEXT session's open — levels re-anchor to
    the real fill). target/stop are ± {target,stop}_atr × daily ATR; time exit at
    the close of the time_barrier_days-th session."""
    out = {"ref_close": None, "atr_pct": float(atr_pct), "entry_rule": "next session open",
           "target_price": None, "stop_price": None,
           "target_pct": round(hc["target_atr"] * atr_pct * 100, 2),
           "stop_pct": round(hc["stop_atr"] * atr_pct * 100, 2),
           "max_hold_days": int(hc["time_barrier_days"]), "qty_ref": None,
           "capital_ref_inr": capital}
    try:
        raw = load_symbol_daily(symbol, end=day, adjusted=False, df_panel=df_panel)
        close = float(raw["close"].iloc[-1])
        if close > 0:
            out["ref_close"] = round(close, 2)
            out["target_price"] = round(close * (1 + hc["target_atr"] * atr_pct), 2)
            out["stop_price"] = round(close * (1 - hc["stop_atr"] * atr_pct), 2)
            out["qty_ref"] = max(1, int(capital / close))
    except Exception as e:  # noqa: BLE001 — levels are advisory; the pick still ships
        logger.warning("trade plan %s: %s", symbol, e)
    return out
