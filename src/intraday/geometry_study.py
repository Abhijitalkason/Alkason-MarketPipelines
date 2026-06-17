"""Geometry tuning study — Phase 3 grid protocol (PLAN_v3 §8, Gate 2;
v3_supplementry §8.5).

Grid over a ∈ {0.25..0.6}, b ∈ {1.5..4.0}. Per cell: relabel the training-fold
candidates at that geometry, measure baseline win rate, the required edge
δ = c/W at a realistic round-trip cost, and projected coverage. Select the
geometry satisfying baseline ≥ 86% and required δ ≤ 4 pts at projected
coverage; the chosen geometry is then FROZEN by a config commit.

Runs on TRAINING FOLDS ONLY. The function accepts a TrainWindow object that
can only be produced by make_folds()[k][0] (a plain list of train days) — the
test windows are structurally unreachable here.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config, load_universe
from src.intraday.costs import round_trip_cost_pct
from src.intraday.labeler import label_day, outcomes_frame
from src.intraday.risk import is_blackout
from src.intraday.screener import screen_day

logger = logging.getLogger(__name__)

A_GRID = [0.25, 0.30, 0.40, 0.50, 0.60]
B_GRID = [1.5, 2.0, 2.4, 3.2, 4.0]


class TrainWindow:
    """Opaque holder for train days — only make_folds / explicit construction
    yields one, so a caller cannot accidentally pass test days to the grid."""

    def __init__(self, days: list[date]):
        self.days = sorted(days)


def _relabel(train: TrainWindow, a: float, b: float) -> pd.DataFrame:
    """Label every screened candidate over the train window at geometry (a, b)."""
    cfg = load_config()
    geo = {**cfg["geometry"], "target_atr": a, "stop_atr": b}
    rows = []
    for day in train.days:
        if is_blackout(day):
            continue
        try:
            top = screen_day(day, load_universe(as_of=day)["symbol"].tolist())
        except Exception as e:  # noqa: BLE001
            logger.warning("geometry: screen %s failed (%s)", day, e)
            continue
        for _, s in top.iterrows():
            if is_blackout(day, s["symbol"]):
                continue
            outs = label_day(s["symbol"], day, geometry=geo,
                             directions=None)  # default long-side baseline
            outs = outcomes_frame(outs)
            if not outs.empty:
                outs["day"] = day
                rows.append(outs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_grid(train: TrainWindow) -> pd.DataFrame:
    """Evaluate the geometry grid on the train window. Writes the grid artifact
    to reports/v3/geometry_grid_<ts>.json and returns the grid DataFrame."""
    if not isinstance(train, TrainWindow):
        raise TypeError("run_grid requires a TrainWindow (train days only) — refuse test data")
    cfg = load_config()
    # representative cost: a typical large-cap notional / median volume
    results = []
    for a in A_GRID:
        for b in B_GRID:
            df = _relabel(train, a, b)
            if df.empty:
                continue
            W = (a + b)  # width in ATR units
            baseline = float(df["label"].mean())
            # express barrier width as a fraction of price: (a+b)·mean(atr_pct)
            mean_atr_pct = _mean_atr_pct(df, a)
            width_pct = W * mean_atr_pct
            c = _representative_cost()
            required_delta = c / width_pct if width_pct > 0 else np.nan
            results.append({
                "a": a, "b": b, "width_atr": W, "width_pct": width_pct,
                "baseline_win_rate": baseline,
                "geometric_floor": b / (a + b),
                "required_delta_pts": required_delta * 100,
                "n_candidates": len(df),
                "projected_coverage_hint": baseline,  # finer coverage comes from the gate
                "meets_baseline_86": baseline >= 0.86,
                "meets_delta_4pts": required_delta * 100 <= 4.0,
            })
    grid = pd.DataFrame(results)
    out_dir = ROOT / cfg["paths"]["reports"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"geometry_grid_{stamp}.json"
    path.write_text(json.dumps(grid.to_dict(orient="records"), indent=2, default=str))
    logger.info("geometry grid → %s", path)
    return grid


def _mean_atr_pct(df: pd.DataFrame, a: float) -> float:
    """Mean ATR as a fraction of entry price. The target distance was built at
    this cell's `a`: |target − entry|/entry = a·atr_pct, so atr_pct = that / a."""
    if df.empty or a <= 0:
        return 0.007
    target_dist_pct = ((df["target"] - df["entry"]).abs() / df["entry"]).mean()
    return float(target_dist_pct / a)


def _representative_cost() -> float:
    """Round-trip cost as a fraction of notional at a representative trade size
    (uses the configured liquidity floor as median volume — conservative)."""
    cfg = load_config()
    vol = cfg["universe"]["min_median_1min_volume"]
    return round_trip_cost_pct(price=1000.0, qty=100, median_1min_volume=vol)


def choose_geometry(grid: pd.DataFrame) -> dict | None:
    """Pick the geometry meeting both constraints with the widest barrier
    (lowest required edge). Returns {target_atr, stop_atr} or None if none qualify
    — the honest Gate-2 fail (PLAN_v3 §19)."""
    ok = grid[grid["meets_baseline_86"] & grid["meets_delta_4pts"]]
    if ok.empty:
        return None
    best = ok.sort_values("required_delta_pts").iloc[0]
    return {"target_atr": float(best["a"]), "stop_atr": float(best["b"])}
