"""Predict → check → score: how the daily list actually does in the Indian market.

Two views, same question:
  scoreboard_from_oos() — the AT-SCALE, lookahead-free answer from a walk-forward
      backtest's OOS artifact: across thousands of past unseen days, how often were
      the top-N picks right, and what did they return (gross + net of cost)?
  evaluate_day() — the LIVE, day-by-day answer: take a saved daily list and, once
      the outcome has matured (the predicted session has closed and is on disk),
      score each pick against the real move; accumulate a running scoreboard.

Both report hit-rate vs the base up-rate (skill), mean realized return, and a
calibration check (mean predicted P(up) vs realized hit-rate).
"""

from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from src.daily import daily_path, load_daily_config, load_universe, today_ist
from src.daily.costs import round_trip_cost_pct
from src.daily.labeler import label_day, label_symbol

logger = logging.getLogger(__name__)


def _agg(picks: pd.DataFrame, base_rate: float) -> dict:
    """Aggregate scoreboard over a set of scored LONG picks."""
    n = len(picks)
    hit_rate = float(picks["correct"].mean())
    return {
        "n_picks": n,
        "n_days": int(picks["date"].nunique()) if "date" in picks else None,
        "hit_rate": hit_rate,
        "base_up_rate": base_rate,
        "edge_vs_base": hit_rate - base_rate,
        "mean_prob": float(picks["prob"].mean()),
        "calibration_gap": float(abs(picks["prob"].mean() - hit_rate)),
        "mean_return_gross": float(picks["gross"].mean()),
        "mean_return_net": float(picks["net"].mean()),
        "win_rate_money": float((picks["net"] > 0).mean()),
    }


# ── at-scale, lookahead-free (from a backtest OOS artifact) ────────────

def scoreboard_from_oos(horizon: str = "swing_1_5d", top_n: int | None = None,
                        capital_per_trade_inr: float = 100_000) -> dict:
    """Build the predict-vs-actual scoreboard from the most recent OOS artifact.
    Each OOS row is a real out-of-sample prediction (p_cal) paired with what the
    stock actually did (label, pnl_pct) — no lookahead."""
    cfg = load_daily_config()
    top_n = top_n or cfg["gate"]["milestone1"]["top_n"]
    arts = sorted(daily_path("reports").glob(f"oos_{horizon}_*.parquet"))
    if not arts:
        raise FileNotFoundError(f"no OOS artifact for {horizon} — run a backtest first")
    oos = pd.read_parquet(arts[-1])
    if oos.empty:
        return {"horizon": horizon, "source": arts[-1].name, "n_picks": 0,
                "note": "OOS artifact is empty"}
    base_rate = float(oos["label"].mean())          # universe-wide up-rate (the benchmark)

    rows = []
    for d, g in oos.groupby("date"):
        top = g.nlargest(min(top_n, len(g)), "p_cal")
        for _, r in top.iterrows():
            if not (r["entry"] > 0):                 # guard bad/zero/NaN entry price
                continue
            qty = max(1, int(capital_per_trade_inr / r["entry"]))
            cost = round_trip_cost_pct(horizon, r["entry"], qty)
            rows.append({
                "date": d, "symbol": r["symbol"], "prob": float(r["p_cal"]),
                "correct": int(r["label"] == 1),          # picks are LONG
                "gross": float(r["pnl_pct"]), "net": float(r["pnl_pct"] - cost),
            })
    picks = pd.DataFrame(rows)
    if picks.empty:
        return {"horizon": horizon, "source": arts[-1].name, "top_n": top_n,
                "n_picks": 0, "note": "no valid picks in OOS artifact"}
    agg = _agg(picks, base_rate)
    return {"horizon": horizon, "source": arts[-1].name, "top_n": top_n,
            "window": {"start": str(oos["date"].min()), "end": str(oos["date"].max())},
            **agg}


def _universe_base_rate(day: date, horizon: str) -> float:
    """The whole PIT universe's realized up-rate for `day` — the benchmark the
    picks are measured against. Returns NaN if nothing has matured yet."""
    ups = n = 0
    for sym in load_universe(as_of=day)["symbol"]:
        out = label_symbol(sym, [day], horizon).get(day)
        if out is None:
            continue
        n += 1
        ups += int(out.label == 1)
    return ups / n if n else float("nan")


# ── live, day-by-day (score a saved list once it matures) ──────────────

def evaluate_day(day: date, horizon: str = "swing_1_5d",
                 capital_per_trade_inr: float = 100_000) -> dict:
    """Score the saved list for `day` against realized outcomes. 'Matured' means the
    predicted session has closed and is on the daily panel; otherwise status
    not_matured (check again after that session's EOD data lands)."""
    list_path = daily_path("lists") / f"list_{day.isoformat()}.json"
    if not list_path.exists():
        raise FileNotFoundError(f"no saved list for {day} at {list_path} — run daily-list first")
    block = json.loads(list_path.read_text())["horizons"].get(horizon)
    if not block or not block["picks"]:
        return {"day": str(day), "horizon": horizon, "status": "no_picks"}

    rows = []
    for p in block["picks"]:
        out = label_day(p["symbol"], day, horizon)     # None until the outcome matures
        if out is None or not (out.entry > 0):
            continue
        dir_sign = 1 if p["direction"].upper() == "LONG" else -1
        gross = dir_sign * out.pnl_pct
        qty = max(1, int(capital_per_trade_inr / out.entry))
        cost = round_trip_cost_pct(horizon, out.entry, qty)
        rows.append({
            "date": pd.Timestamp(day), "symbol": p["symbol"], "direction": p["direction"],
            "prob": float(p["prob"]), "actual_up": int(out.label == 1),
            "correct": int((out.label == 1) if dir_sign == 1 else (out.label == 0)),
            "actual_return": float(out.pnl_pct), "gross": float(gross),
            "net": float(gross - cost), "barrier": out.barrier,
        })
    if not rows:
        return {"day": str(day), "horizon": horizon, "status": "not_matured",
                "note": "predicted session hasn't closed / isn't on the panel yet — "
                        "refresh the panel after EOD and re-run"}

    picks = pd.DataFrame(rows)
    # base rate = the whole universe's up-rate that day (the benchmark the picks
    # must beat) — NOT the picks' own up-rate, which for LONG-only picks equals the
    # hit-rate and would make edge_vs_base identically zero.
    base_rate = _universe_base_rate(day, horizon)
    agg = _agg(picks, base_rate)
    result = {"day": str(day), "horizon": horizon, "status": "matured",
              "picks": rows, **agg}

    out_dir = daily_path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{horizon}_{day.isoformat()}.json").write_text(json.dumps(result, indent=2, default=str))
    _append_scoreboard(result, horizon)
    logger.info("daily eval %s %s: hit_rate=%.3f net=%.4f (%d picks)",
                day, horizon, agg["hit_rate"], agg["mean_return_net"], agg["n_picks"])
    return result


def _append_scoreboard(result: dict, horizon: str) -> None:
    reg = daily_path("reports") / f"scoreboard_{horizon}.csv"
    line = {k: result[k] for k in ("day", "n_picks", "hit_rate", "base_up_rate",
                                   "edge_vs_base", "mean_return_gross", "mean_return_net",
                                   "calibration_gap")}
    pd.DataFrame([line]).to_csv(reg, mode="a", header=not reg.exists(), index=False)


def scoreboard(horizon: str = "swing_1_5d") -> dict:
    """Aggregate the running live scoreboard (all matured evals to date)."""
    reg = daily_path("reports") / f"scoreboard_{horizon}.csv"
    if not reg.exists():
        return {"horizon": horizon, "n_days": 0, "note": "no live evals yet — run daily-eval"}
    df = pd.read_csv(reg)
    w = df["n_picks"]
    return {
        "horizon": horizon, "n_days": len(df), "n_picks": int(w.sum()),
        "hit_rate": float(np.average(df["hit_rate"], weights=w)),
        "base_up_rate": float(np.average(df["base_up_rate"], weights=w)),
        "edge_vs_base": float(np.average(df["edge_vs_base"], weights=w)),
        "mean_return_net": float(np.average(df["mean_return_net"], weights=w)),
        "since": str(df["day"].min()), "through": str(df["day"].max()),
    }
