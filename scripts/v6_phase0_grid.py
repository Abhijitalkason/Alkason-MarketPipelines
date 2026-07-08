"""PLAN_v6 Phase 0 — §4.1/4.2/4.3/4.5 measurement run.

Builds the panel + PIT universe, labels all 48 cells (long primary + short
mirror diagnostic), measures gap risk, evaluates the frozen §4.7 v2 criteria
1–5, and persists everything for the report assembler:

    reports/v6/grid_long.parquet      per-cell stats + criteria columns
    reports/v6/labels_sample.parquet  the qualifying cells' raw labels
    reports/v6/gap_risk.json          §4.5 gap + circuit-lock measurements
    reports/v6/grid_run_meta.json     run metadata + data-sanity counts

Run: python -m scripts.v6_phase0_grid
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.intraday import ROOT
from src.v6 import grid as gridmod
from src.v6.labeler import all_cells, label_universe
from src.v6.panel import load_panel
from src.v6.universe import in_universe_mask, monthly_pit_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("v6.phase0")

OUT = ROOT / "reports" / "v6"
ATR_SANITY_MAX = 0.10   # signal days with trailing ATR > 10% are data-suspect


def main() -> None:
    t0 = time.time()
    costs = json.loads((OUT / "cost_curve.json").read_text())
    c1x = costs["cost_pct_at_reference"] / 100
    c2x = costs["cost_pct_at_reference_2x"] / 100

    panel = load_panel()
    universe = monthly_pit_universe(panel)
    mask = in_universe_mask(panel, universe)

    # data-sanity guard: unadjusted corporate actions masquerade as huge ATR
    suspect = mask & (panel["atr_pct"] > ATR_SANITY_MAX)
    mask = mask & ~(panel["atr_pct"] > ATR_SANITY_MAX)
    logger.info("in-universe signal rows: %d (excluded %d ATR-sanity rows)",
                mask.sum(), suspect.sum())

    rows, label_frames = [], []
    for i, cell in enumerate(all_cells()):
        labels = label_universe(panel, mask, cell, side="long")
        shorts = label_universe(panel, mask, cell, side="short")
        rows.append(gridmod.cell_stats(labels, cell, c1x, c2x, shorts))
        label_frames.append(labels)
        logger.info("[%02d/48] %s: n=%d wr=%.3f stop=%.2f time=%.2f d1x=%.1f",
                    i + 1, cell.key, len(labels), rows[-1]["wr_pooled"],
                    rows[-1]["share_stop"], rows[-1]["share_time"],
                    rows[-1]["delta_req_1x"])

    grid = pd.DataFrame(rows)
    grid = gridmod.evaluate_go_kill(grid)
    by_year_col = grid.pop("wr_by_year")
    grid["wr_by_year"] = [json.dumps(d) for d in by_year_col]
    grid.to_parquet(OUT / "grid_long.parquet", index=False)

    qualifying = grid[grid["go_1_to_4"]]
    keep = pd.concat([f for f, k in zip(label_frames, grid["cell"])
                      if k in set(qualifying["cell"])], ignore_index=True) \
        if len(qualifying) else pd.concat(label_frames[:1], ignore_index=True)
    keep.to_parquet(OUT / "labels_sample.parquet", index=False)

    gap = measure_gap_risk(panel, mask, pd.concat(label_frames, ignore_index=True))
    (OUT / "gap_risk.json").write_text(json.dumps(gap, indent=2))

    meta = {
        "generated": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t0),
        "panel_rows": int(len(panel)), "symbols": int(panel["symbol"].nunique()),
        "universe_months": int(universe["month"].nunique()),
        "universe_unique_symbols": int(universe["symbol"].nunique()),
        "signal_rows": int(mask.sum()), "atr_sanity_excluded": int(suspect.sum()),
        "atr_pct_median": round(float(panel.loc[mask, "atr_pct"].median()), 4),
        "atr_pct_p10": round(float(panel.loc[mask, "atr_pct"].quantile(.1)), 4),
        "atr_pct_p90": round(float(panel.loc[mask, "atr_pct"].quantile(.9)), 4),
        "cost_1x_pct": costs["cost_pct_at_reference"],
        "cost_2x_pct": costs["cost_pct_at_reference_2x"],
        "cells_passing_1_to_4": int(grid["go_1_to_4"].sum()),
        "cells_passing_all": int(grid["go_all"].sum()),
    }
    (OUT / "grid_run_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("done in %ds — cells passing criteria 1-4: %d, all: %d",
                meta["runtime_s"], meta["cells_passing_1_to_4"], meta["cells_passing_all"])


def measure_gap_risk(panel: pd.DataFrame, mask: pd.Series,
                     all_labels: pd.DataFrame) -> dict:
    """§4.5 — overnight gap tails on the universe + per-label stop overshoot,
    split 1-night vs weekend/multi-night (F17), plus the circuit-lock tail (F22)."""
    p = panel[mask & panel["prev_close_adj"].notna() & (panel["atr_pct"] > 0)]
    gap_atr = ((p["open"] / p["prev_close_adj"] - 1).abs() / p["atr_pct"])
    stops = all_labels[all_labels["exit_type"] == "stop"]
    gt = stops[stops["gap_through"]]
    one_night = gt[~gt["multi_night_gap"]]
    multi = gt[gt["multi_night_gap"]]
    locked_stops = stops[stops["exit_locked"]]
    return {
        "gap_beyond_1atr_share": round(float((gap_atr > 1).mean()), 5),
        "gap_beyond_2atr_share": round(float((gap_atr > 2).mean()), 5),
        "gap_beyond_3atr_share": round(float((gap_atr > 3).mean()), 5),
        "stop_exits": int(len(stops)),
        "gap_through_share_of_stops": round(float(len(gt) / max(len(stops), 1)), 4),
        "loss_beyond_stop_frac_mean": round(float(gt["loss_beyond_stop_frac"].mean())
                                            if len(gt) else 0.0, 4),
        "loss_beyond_stop_frac_p95": round(float(gt["loss_beyond_stop_frac"].quantile(.95))
                                           if len(gt) else 0.0, 4),
        "one_night": {"n": int(len(one_night)),
                      "loss_beyond_mean": round(float(one_night["loss_beyond_stop_frac"].mean())
                                                if len(one_night) else 0.0, 4)},
        "multi_night": {"n": int(len(multi)),
                        "loss_beyond_mean": round(float(multi["loss_beyond_stop_frac"].mean())
                                                  if len(multi) else 0.0, 4)},
        "circuit_lock": {
            "stop_exits_on_locked_day": int(len(locked_stops)),
            "share_of_stops": round(float(len(locked_stops) / max(len(stops), 1)), 5),
        },
    }


if __name__ == "__main__":
    main()
