"""PLAN_v6 §4.3 — geometry-grid measurement: per-cell mandatory columns and
the frozen §4.7 v2 GO/KILL evaluation.

Empirical required δ (F11): with measured win prob p, mean winning return
r_win, mean losing return r_loss (both from the actual exit mix, so time
exits and gap-throughs are priced), post-cost expectancy at win prob q is
    E(q) = q·r_win + (1−q)·r_loss − c.
δ_required = max(0, q* − p)·100 points, where q* solves E(q*) = 0. This
replaces the closed-form c/W, which understates the requirement when
time-exits exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.v6.labeler import Cell

# §4.7 v2 frozen thresholds (approved 2026-07-08 — do not edit)
BAND_LO, BAND_HI = 0.72, 0.79
HALF_TOL = 0.03
WORST_YEAR_MIN = 0.70
NEIGHBOR_TOL = 0.03
STOP_SHARE_MIN = 0.10
TIME_SHARE_MAX = 0.40
DELTA_MAX_1X = 4.0
DELTA_MAX_2X = 6.0
GAP_LOSS_MAX = 0.25


def required_delta_points(labels: pd.DataFrame, cost_frac: float) -> float:
    """Empirical required δ in probability points for one cell's label set."""
    wins, losses = labels[labels["win"]], labels[~labels["win"]]
    if wins.empty or losses.empty:
        return np.inf
    p = len(wins) / len(labels)
    r_win, r_loss = wins["ret"].mean(), losses["ret"].mean()
    if r_win <= r_loss:
        return np.inf
    q_star = (cost_frac - r_loss) / (r_win - r_loss)
    return max(0.0, (q_star - p)) * 100


def cell_stats(labels: pd.DataFrame, cell: Cell, cost_1x: float,
               cost_2x: float, short_labels: pd.DataFrame | None = None) -> dict:
    """All §4.3 mandatory columns for one cell (long side + short diagnostic)."""
    n = len(labels)
    wr = labels["win"].mean()
    year = pd.to_datetime(labels["signal_date"]).dt.year
    by_year = labels.groupby(year)["win"].agg(["mean", "size"])
    med_date = pd.to_datetime(labels["signal_date"]).median()
    first_half = pd.to_datetime(labels["signal_date"]) <= med_date
    mix = labels["exit_type"].value_counts(normalize=True)
    stop_rows = labels[labels["exit_type"] == "stop"]
    out = {
        "cell": cell.key, "a": cell.a, "b": cell.b, "h": cell.h, "n": n,
        "wr_pooled": wr,
        "wr_half1": labels.loc[first_half, "win"].mean(),
        "wr_half2": labels.loc[~first_half, "win"].mean(),
        "wr_worst_year": by_year.loc[by_year["size"] >= 50, "mean"].min(),
        "wr_by_year": {int(y): round(v, 4) for y, v in by_year["mean"].items()},
        "share_target": mix.get("target", 0.0),
        "share_stop": mix.get("stop", 0.0),
        "share_time": mix.get("time", 0.0),
        "gap_through_share": labels["gap_through"].mean(),
        "dual_touch_share": labels["dual_touch"].mean(),
        "expected_loss_beyond_stop": (stop_rows["loss_beyond_stop_frac"].mean()
                                      if len(stop_rows) else 0.0),
        "mean_ret_win": labels.loc[labels["win"], "ret"].mean(),
        "mean_ret_loss": labels.loc[~labels["win"], "ret"].mean(),
        "expectancy_precost": labels["ret"].mean(),
        "delta_req_1x": required_delta_points(labels, cost_1x),
        "delta_req_2x": required_delta_points(labels, cost_2x),
        "theory_floor": cell.b / (cell.a + cell.b),
        # F19/F25 (review 2): holds whose window straddles a corporate-action
        # ex-date (bonus, split, or detected residual) — disclosed, not excluded.
        "ca_in_hold_share": (labels["ca_in_hold"].mean()
                             if "ca_in_hold" in labels.columns else float("nan")),
        "ca_in_hold_n": (int(labels["ca_in_hold"].sum())
                        if "ca_in_hold" in labels.columns else 0),
    }
    if short_labels is not None and len(short_labels):
        out["wr_short_mirror"] = short_labels["win"].mean()
        out["long_short_asymmetry"] = wr - out["wr_short_mirror"]
    return out


def neighbors(cell_row: pd.Series, grid: pd.DataFrame) -> pd.DataFrame:
    """Existing one-step (a, b) neighbors at the same horizon (F24 grid-edge
    rule: all existing neighbors count; missing ones are neither pass nor fail)."""
    from src.v6.labeler import GRID_A, GRID_B
    ia, ib = GRID_A.index(cell_row["a"]), GRID_B.index(cell_row["b"])
    keys = []
    if ia > 0:
        keys.append((GRID_A[ia - 1], cell_row["b"]))
    if ia < len(GRID_A) - 1:
        keys.append((GRID_A[ia + 1], cell_row["b"]))
    if ib > 0:
        keys.append((cell_row["a"], GRID_B[ib - 1]))
    if ib < len(GRID_B) - 1:
        keys.append((cell_row["a"], GRID_B[ib + 1]))
    return grid[grid.apply(lambda r: (r["a"], r["b"]) in keys
                           and r["h"] == cell_row["h"], axis=1)]


def evaluate_go_kill(grid: pd.DataFrame) -> pd.DataFrame:
    """Frozen §4.7 v2 criteria 1–5 per cell (criterion 6, tie-break bias, is a
    global qualifier evaluated separately). Returns grid with pass columns."""
    g = grid.copy()
    g["c1_band"] = g["wr_pooled"].between(BAND_LO, BAND_HI)
    g["c2_halves"] = ((g["wr_half1"] - g["wr_pooled"]).abs() <= HALF_TOL) & \
                     ((g["wr_half2"] - g["wr_pooled"]).abs() <= HALF_TOL)
    g["c2_worst_year"] = g["wr_worst_year"] >= WORST_YEAR_MIN
    g["c2_plateau"] = [
        (lambda nb: len(nb) > 0 and
         (nb["wr_pooled"] - row["wr_pooled"]).abs().le(NEIGHBOR_TOL).all())
        (neighbors(row, g))
        for _, row in g.iterrows()
    ]
    g["c3_exercised"] = (g["share_stop"] >= STOP_SHARE_MIN) & \
                        (g["share_time"] <= TIME_SHARE_MAX)
    g["c4_delta"] = (g["delta_req_1x"] <= DELTA_MAX_1X) & \
                    (g["delta_req_2x"] <= DELTA_MAX_2X)
    g["c5_gap"] = g["expected_loss_beyond_stop"] <= GAP_LOSS_MAX
    g["go_1_to_4"] = g["c1_band"] & g["c2_halves"] & g["c2_worst_year"] & \
        g["c2_plateau"] & g["c3_exercised"] & g["c4_delta"]
    g["go_all"] = g["go_1_to_4"] & g["c5_gap"]
    # F24 effective-edge disclosure: what the model must actually supply
    g["effective_edge_req"] = np.maximum(
        (0.80 - g["wr_pooled"]) * 100, g["delta_req_1x"])
    return g
