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
from src.v6.panel import load_ca_events, load_panel
from src.v6.universe import in_universe_mask, monthly_pit_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("v6.phase0")

OUT = ROOT / "reports" / "v6"
ATR_SANITY_MAX = 0.10   # signal days with trailing ATR > 10% are data-suspect
EXTREME_LOSS_TRIPWIRE = -0.30   # F25/R1: single-trade loss beyond this is not
                                # organic for top-100-liquidity names — any
                                # survivor is an undetected data artifact


def main() -> None:
    t0 = time.time()
    costs = json.loads((OUT / "cost_curve.json").read_text())
    c1x = costs["cost_pct_at_reference"] / 100
    c2x = costs["cost_pct_at_reference_2x"] / 100

    panel = load_panel()
    ca_events = load_ca_events()
    universe = monthly_pit_universe(panel)
    mask = in_universe_mask(panel, universe)

    # data-sanity guard: unadjusted corporate actions masquerade as huge ATR
    suspect = mask & (panel["atr_pct"] > ATR_SANITY_MAX)
    mask = mask & ~(panel["atr_pct"] > ATR_SANITY_MAX)
    logger.info("in-universe signal rows: %d (excluded %d ATR-sanity rows)",
                mask.sum(), suspect.sum())

    rows, label_frames = [], []
    for i, cell in enumerate(all_cells()):
        labels = label_universe(panel, mask, cell, side="long", ca_events=ca_events)
        shorts = label_universe(panel, mask, cell, side="short", ca_events=ca_events)
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

    # F25/R4 (review 2): persist the BINDING cells' raw labels on a KILL (the
    # cell nearest the WR band and the cell nearest the delta ceiling), not
    # just the grid's first cell — an evidence file should carry the audit
    # trail for the numbers it actually leans on.
    qualifying = grid[grid["go_1_to_4"]]
    if len(qualifying):
        keep_cells = set(qualifying["cell"])
    else:
        keep_cells = {grid.loc[grid["wr_pooled"].idxmax(), "cell"],
                     grid.loc[grid["delta_req_1x"].idxmin(), "cell"]}
    keep = pd.concat([f for f, k in zip(label_frames, grid["cell"]) if k in keep_cells],
                     ignore_index=True)
    keep.to_parquet(OUT / "labels_sample.parquet", index=False)

    all_labels = pd.concat(label_frames, ignore_index=True)
    gap = measure_gap_risk(panel, mask, all_labels)
    (OUT / "gap_risk.json").write_text(json.dumps(gap, indent=2))

    # F25 tripwire: any surviving single-trade loss beyond EXTREME_LOSS_TRIPWIRE
    # is not an organic move for this universe — it flags residual data
    # artifacts the CA detector's [0.5, 2.0] band didn't catch (disclosed
    # known gap: demergers/splits whose ratio lands inside that band, e.g. a
    # clean 2:1 split with a small same-day return — see PLAN_v6 F25).
    extreme = all_labels[all_labels["ret"] < EXTREME_LOSS_TRIPWIRE]
    extreme_detail = sorted(set(zip(extreme["symbol"], extreme["exit_date"].astype(str))))
    logger.info("extreme-loss tripwire (<%.0f%%): %d label rows, %d unique (symbol, exit_date)",
                EXTREME_LOSS_TRIPWIRE * 100, len(extreme), len(extreme_detail))

    delisting = measure_delisting_tail(panel, universe)

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
        "ca_events_csv_bonus": int((ca_events["purpose"] != "detected_residual").sum()),
        "ca_events_detected_residual": int((ca_events["purpose"] == "detected_residual").sum()),
        "ca_in_hold_share_mean_across_cells": round(float(grid["ca_in_hold_share"].mean()), 4),
        "extreme_loss_tripwire_threshold": EXTREME_LOSS_TRIPWIRE,
        "extreme_loss_tripwire_label_rows": int(len(extreme)),
        "extreme_loss_tripwire_unique_events": len(extreme_detail),
        "extreme_loss_tripwire_events": extreme_detail[:20],
        "delisting_tail": delisting,
    }
    (OUT / "grid_run_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("done in %ds — cells passing criteria 1-4: %d, all: %d",
                meta["runtime_s"], meta["cells_passing_1_to_4"], meta["cells_passing_all"])


def measure_delisting_tail(panel: pd.DataFrame, universe: pd.DataFrame) -> dict:
    """R4 (review 2) — the delisting/merger tail count PLAN_v6 §4.1 promised
    ("delistings simply truncate the series and are counted upstream as tail
    events") but the original implementation never computed: universe
    symbols whose panel series ends materially before the panel's own end
    date (mergers like HDFC->HDFCBANK 2023, delistings, suspensions)."""
    global_end = panel["date"].max()
    last_date = panel.groupby("symbol")["date"].max()
    uni_syms = set(universe["symbol"].unique())
    truncated = last_date[last_date.index.isin(uni_syms)
                          & (last_date < global_end - pd.Timedelta(days=30))]
    return {
        "universe_symbols_with_truncated_series": int(len(truncated)),
        "symbols_sample": sorted(truncated.index.tolist())[:20],
    }


def measure_gap_risk(panel: pd.DataFrame, mask: pd.Series,
                     all_labels: pd.DataFrame) -> dict:
    """§4.5 — overnight gap tails on the universe + per-label stop overshoot,
    split 1-night vs weekend/multi-night (F17), plus the circuit-lock tail (F22).

    F25 (review 2): the one_night/multi_night SPLIT is deduped by
    (symbol, exit_date) — the same overnight gap otherwise repeats once per
    cell (up to 48x) and the printed "n" would overstate distinct events. The
    pooled gap_through_share_of_stops / loss_beyond_stop_frac stats are left
    per-(trade,cell) intentionally: a stop's gap-through-ness is a genuine
    property of that cell's specific stop distance, so pooling those IS
    correct — only the F17 event COUNT was the overcounting bug."""
    p = panel[mask & panel["prev_close_adj"].notna() & (panel["atr_pct"] > 0)]
    gap_atr = ((p["open"] / p["prev_close_adj"] - 1).abs() / p["atr_pct"])
    stops = all_labels[all_labels["exit_type"] == "stop"]
    gt = stops[stops["gap_through"]]
    gt_events = gt.drop_duplicates(subset=["symbol", "exit_date"])
    one_night = gt_events[~gt_events["multi_night_gap"]]
    multi = gt_events[gt_events["multi_night_gap"]]
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
