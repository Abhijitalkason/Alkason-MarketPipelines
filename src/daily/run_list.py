"""Daily list job — for each enabled horizon: score → rank → top-N → 'why' →
chart → write the list artifact. This is the once-a-day entry point that produces
the watchlist the API serves. Idempotent and point-in-time (decision_ts=now).

Whether a horizon's list is labelled `actionable` comes from its latest backtest
verdict (Milestone 2): predictive-only until the after-cost gate has passed.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from src.daily import daily_path, enabled_horizons, load_daily_config, now_ist, today_ist
from src.daily.charts import render_pick
from src.daily.screener import screen_day

logger = logging.getLogger(__name__)


def _actionable(horizon: str) -> bool:
    """A horizon's list is 'actionable' only if its most recent backtest cleared
    Milestone 2. Absent a report, default to predictive-only (False)."""
    reports = sorted(daily_path("reports").glob(f"backtest_{horizon}_*.json"))
    if not reports:
        return False
    try:
        rep = json.loads(reports[-1].read_text())
        return bool(rep["milestone1_predictive"]["passed"] and rep["milestone2_after_cost"]["passed"])
    except Exception:  # noqa: BLE001
        return False


def run(day: date | None = None, horizons: list[str] | None = None) -> dict:
    cfg = load_daily_config()
    day = day or today_ist()
    horizons = horizons or enabled_horizons()
    if not horizons:
        raise RuntimeError("no enabled horizons in config_daily.yaml")

    out = {"date": str(day), "generated": now_ist().isoformat(timespec="seconds"), "horizons": {}}
    for horizon in horizons:
        picks_df = screen_day(day, horizon)
        actionable = _actionable(horizon)
        picks = []
        trade_cols = ["ref_close", "entry_rule", "target_price", "stop_price",
                      "target_pct", "stop_pct", "max_hold_days", "qty_ref",
                      "capital_ref_inr", "atr_pct"]
        for _, r in picks_df.iterrows():
            chart = render_pick(r["symbol"], day, r["direction"], r["prob"], horizon) \
                if cfg["listing"]["charts"] else None
            picks.append({
                "symbol": r["symbol"], "direction": r["direction"], "prob": round(r["prob"], 4),
                "why": r["why"], "chart_path": chart, "actionable": actionable,
                **{c: r[c] for c in trade_cols if c in picks_df.columns},
            })
        out["horizons"][horizon] = {"actionable": actionable, "picks": picks}
        logger.info("daily list %s %s: %d picks (actionable=%s)", day, horizon, len(picks), actionable)

    lists_dir = daily_path("lists")
    lists_dir.mkdir(parents=True, exist_ok=True)
    (lists_dir / f"list_{day.isoformat()}.json").write_text(json.dumps(out, indent=2, default=str))
    return out
