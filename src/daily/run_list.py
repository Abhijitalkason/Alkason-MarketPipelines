"""Daily list job — for each enabled horizon: score → rank → top-N → 'why' →
chart → write the list artifact. This is the once-a-day entry point that produces
the watchlist the API serves. Idempotent and point-in-time (decision_ts=now).

Whether a horizon's list is labelled `actionable` comes from its latest backtest
verdict (Milestone 2): predictive-only until the after-cost gate has passed.

Two-tier output (design update 2026-07-15, user request):
  tier 2 — TOP PICK: exactly ONE stock per day, the primary horizon's rank #1,
           emitted EVERY day with its true calibrated probability and an honest
           confidence note. "The single name with the highest chances today" —
           a relative statement, tracked daily, never a promise.
  tier 1 — STRONG SIGNAL: any pick whose calibrated P(win) clears
           listing.top_pick.strong_signal_min_prob (0.80 — the frozen selective
           contract). Logged loudly when it fires; expected to be rare/absent
           until a model demonstrates measured edge.
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


def top_pick_block(picks: list[dict], horizon: str, strong_min: float) -> dict | None:
    """Tier assignment for the day's single pick (pure — unit-tested).

    picks are already ranked (rank #1 first, screener tie-breaks applied). The
    pick is promoted to STRONG_SIGNAL only if its calibrated probability clears
    `strong_min`; otherwise it ships as TOP_PICK with an honest note. Returns
    None when there are no picks (e.g. data outage) — an absent pick must never
    be silently substituted."""
    if not picks:
        return None
    p = dict(picks[0])
    strong = p["prob"] >= strong_min
    p["horizon"] = horizon
    p["tier"] = "STRONG_SIGNAL" if strong else "TOP_PICK"
    p["confidence_note"] = (
        f"calibrated P(win)={p['prob']:.3f} >= {strong_min:.2f}: clears the frozen "
        "selective bar — rare by design; verify before acting"
        if strong else
        f"rank #1 of the universe today by model score + momentum tie-break; "
        f"calibrated P(win)={p['prob']:.3f} (below the {strong_min:.2f} strong-signal bar). "
        "This is the day's RELATIVE best, not a high-confidence call — track the "
        "running hit rate before trusting it with capital."
    )
    return p


def run(day: date | None = None, horizons: list[str] | None = None,
        trace=None) -> dict:
    """trace: optional callable(step, item, status, **detail) — threads through
    to screen_day (per-stock scoring events) and journals the final ranking."""
    cfg = load_daily_config()
    day = day or today_ist()
    horizons = horizons or enabled_horizons()
    if not horizons:
        raise RuntimeError("no enabled horizons in config_daily.yaml")

    out = {"date": str(day), "generated": now_ist().isoformat(timespec="seconds"), "horizons": {}}
    tp_cfg = cfg["listing"].get("top_pick", {})
    strong_min = float(tp_cfg.get("strong_signal_min_prob", 0.80))
    strong_signals = []
    for horizon in horizons:
        picks_df = screen_day(day, horizon, trace=trace)
        actionable = _actionable(horizon)
        picks = []
        trade_cols = ["ref_close", "entry_rule", "target_price", "stop_price",
                      "target_pct", "stop_pct", "max_hold_days", "qty_ref",
                      "capital_ref_inr", "atr_pct"]
        for rank, (_, r) in enumerate(picks_df.iterrows(), start=1):
            chart = render_pick(r["symbol"], day, r["direction"], r["prob"], horizon) \
                if cfg["listing"]["charts"] else None
            picks.append({
                "rank": rank,   # rank #1 = the day's top pick; kept per-pick for scoreboards
                "symbol": r["symbol"], "direction": r["direction"], "prob": round(r["prob"], 4),
                "why": r["why"], "chart_path": chart, "actionable": actionable,
                **{c: r[c] for c in trade_cols if c in picks_df.columns},
            })
        out["horizons"][horizon] = {"actionable": actionable, "picks": picks}
        strong_signals += [{**p, "horizon": horizon} for p in picks if p["prob"] >= strong_min]
        if trace:
            for p in picks:
                trace("rank", p["symbol"], "ok", rank=p["rank"], prob=p["prob"],
                      horizon=horizon, why=", ".join(p["why"]),
                      target=p.get("target_price"), stop=p.get("stop_price"))
        logger.info("daily list %s %s: %d picks (actionable=%s)", day, horizon, len(picks), actionable)

    # tier 2 — the day's single TOP PICK, from the PRIMARY (first enabled) horizon
    if tp_cfg.get("enabled", False):
        primary = horizons[0]
        out["top_pick"] = top_pick_block(out["horizons"][primary]["picks"], primary, strong_min)
    # tier 1 — strong signals across horizons: SPEAK when one fires
    out["strong_signals"] = strong_signals
    if strong_signals:
        for s in strong_signals:
            logger.warning("STRONG SIGNAL %s: %s %s P(win)=%.3f >= %.2f — verify before acting",
                           day, s["horizon"], s["symbol"], s["prob"], strong_min)

    lists_dir = daily_path("lists")
    lists_dir.mkdir(parents=True, exist_ok=True)
    (lists_dir / f"list_{day.isoformat()}.json").write_text(json.dumps(out, indent=2, default=str))
    return out
