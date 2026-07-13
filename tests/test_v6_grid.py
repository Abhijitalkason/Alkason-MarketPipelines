"""Tests for the PLAN_v6 §4.3/§4.7 v2 GO/KILL evaluator (src/v6/grid.py).

Added in review 2 (2026-07-10) — the gate itself had zero test coverage
before this, the sharpest gap for a module whose numbers decide the project.
"""

import numpy as np
import pandas as pd
import pytest

from src.v6 import grid as gridmod
from src.v6.grid import evaluate_go_kill, neighbors, required_delta_points


def _labels(rets, wins):
    return pd.DataFrame({
        "ret": rets, "win": wins,
        "exit_type": ["target" if w else "stop" for w in wins],
        "signal_date": pd.date_range("2024-01-01", periods=len(rets)),
    })


def test_required_delta_points_hand_computed():
    # 4 wins at +2%, 4 losses at -4%: p=0.5, r_win=0.02, r_loss=-0.04.
    # q* solves q*0.02 + (1-q)*(-0.04) - c = 0, c = 0.004 (0.4%)
    # q*0.06 = 0.044 -> q* = 0.7333; delta = (0.7333 - 0.5)*100 = 23.33 pts
    labels = _labels([0.02] * 4 + [-0.04] * 4, [True] * 4 + [False] * 4)
    delta = required_delta_points(labels, cost_frac=0.004)
    assert delta == pytest.approx(23.333, abs=0.01)


def test_required_delta_points_zero_when_already_profitable():
    # p=0.9 far above breakeven -> delta clamped to 0, not negative.
    labels = _labels([0.02] * 9 + [-0.04], [True] * 9 + [False])
    delta = required_delta_points(labels, cost_frac=0.001)
    assert delta == 0.0


def test_required_delta_points_inf_when_win_not_bigger_than_loss():
    # r_win <= r_loss (degenerate/pathological mix) -> no win rate saves it.
    labels = _labels([0.01, 0.01, 0.02, 0.02], [True, False, True, False])
    assert required_delta_points(labels, cost_frac=0.001) == np.inf


def test_frozen_thresholds_match_plan_v6_section_4_7_v2():
    """§4.7 v2 was approved and frozen 2026-07-08 — this is the tripwire: if
    anyone edits these constants, this test fails and must be consciously
    updated (and the plan's amendment log re-checked), rather than the gate
    silently drifting from the document that governs it."""
    assert (gridmod.BAND_LO, gridmod.BAND_HI) == (0.72, 0.79)
    assert gridmod.HALF_TOL == 0.03
    assert gridmod.WORST_YEAR_MIN == 0.70
    assert gridmod.NEIGHBOR_TOL == 0.03
    assert gridmod.STOP_SHARE_MIN == 0.10
    assert gridmod.TIME_SHARE_MAX == 0.40
    assert gridmod.DELTA_MAX_1X == 4.0
    assert gridmod.DELTA_MAX_2X == 6.0
    assert gridmod.GAP_LOSS_MAX == 0.25


def _grid_row(a, b, h, wr):
    return {"cell": f"a{a}_b{b}_h{h}", "a": a, "b": b, "h": h, "wr_pooled": wr}


def test_neighbors_interior_cell_has_four():
    # a=0.75 (index 1 of 4), b=3 (index 1 of 4) — both sides exist.
    from src.v6.labeler import GRID_A, GRID_B
    rows = [_grid_row(a, b, 5, 0.7) for a in GRID_A for b in GRID_B]
    grid = pd.DataFrame(rows)
    row = grid[(grid.a == 0.75) & (grid.b == 3)].iloc[0]
    nb = neighbors(row, grid)
    assert len(nb) == 4
    assert set(zip(nb["a"], nb["b"])) == {(0.5, 3), (1.0, 3), (0.75, 2), (0.75, 4)}


def test_neighbors_corner_cell_has_two():
    # a=0.5 (grid edge, index 0), b=2 (grid edge, index 0) — F24 grid-edge rule.
    from src.v6.labeler import GRID_A, GRID_B
    rows = [_grid_row(a, b, 5, 0.7) for a in GRID_A for b in GRID_B]
    grid = pd.DataFrame(rows)
    row = grid[(grid.a == 0.5) & (grid.b == 2)].iloc[0]
    nb = neighbors(row, grid)
    assert len(nb) == 2
    assert set(zip(nb["a"], nb["b"])) == {(0.75, 2), (0.5, 3)}


def _synthetic_cell(wr, delta_1x=2.0, delta_2x=3.0, stop_share=0.15,
                    time_share=0.20, gap_loss=0.10, worst_year=None,
                    half1=None, half2=None):
    half1 = wr if half1 is None else half1
    half2 = wr if half2 is None else half2
    worst_year = wr if worst_year is None else worst_year
    return {
        "wr_pooled": wr, "wr_half1": half1, "wr_half2": half2,
        "wr_worst_year": worst_year, "share_target": 1 - stop_share - time_share,
        "share_stop": stop_share, "share_time": time_share,
        "delta_req_1x": delta_1x, "delta_req_2x": delta_2x,
        "expected_loss_beyond_stop": gap_loss,
    }


def test_evaluate_go_kill_end_to_end_one_passing_cell():
    """Synthetic 6-cell grid on the real (a, b) axis with one cell engineered
    to pass every criterion; the rest fail one criterion each — confirms the
    evaluator finds the GO cell and correctly rejects the near-misses."""
    from src.v6.labeler import GRID_A, GRID_B
    rows = []
    for a, b in zip(GRID_A[:3], GRID_B[:3]):
        rows.append({**_grid_row(a, b, 5, 0.75), **_synthetic_cell(0.75)})
    # a fail on band (criterion 1): WR outside [0.72, 0.79]
    rows[0].update(_synthetic_cell(0.90))
    rows[0]["cell"], rows[0]["a"], rows[0]["b"] = "a0.5_b2_h5", 0.5, 2
    # a fail on delta (criterion 4)
    rows[1].update(_synthetic_cell(0.75, delta_1x=8.0))
    rows[1]["cell"], rows[1]["a"], rows[1]["b"] = "a0.75_b2_h5", 0.75, 2
    # the one GO cell: passes 1-5
    rows[2].update(_synthetic_cell(0.75, delta_1x=2.0, delta_2x=3.0,
                                   stop_share=0.15, time_share=0.20, gap_loss=0.10))
    rows[2]["cell"], rows[2]["a"], rows[2]["b"] = "a1.0_b2_h5", 1.0, 2
    for r in rows:
        r["h"] = 5

    grid = pd.DataFrame(rows)
    out = evaluate_go_kill(grid)

    assert out.loc[out["cell"] == "a1.0_b2_h5", "go_1_to_4"].iloc[0]
    assert not out.loc[out["cell"] == "a0.5_b2_h5", "c1_band"].iloc[0]
    assert not out.loc[out["cell"] == "a0.75_b2_h5", "c4_delta"].iloc[0]
    assert int(out["go_1_to_4"].sum()) == 1


def test_evaluate_go_kill_zero_cells_is_kill():
    from src.v6.labeler import GRID_A, GRID_B
    rows = [{**_grid_row(a, b, 5, 0.60), **_synthetic_cell(0.60)}
           for a, b in zip(GRID_A, GRID_B)]
    for r in rows:
        r["h"] = 5
    grid = pd.DataFrame(rows)
    out = evaluate_go_kill(grid)
    assert int(out["go_1_to_4"].sum()) == 0
