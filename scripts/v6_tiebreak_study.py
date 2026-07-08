"""PLAN_v6 §4.2 — tie-break bias study: daily-OHLC dual-touch resolution vs
1-minute truth on the 46 bar-store names.

Ambiguity in daily-bar resolution exists ONLY where the labeler collapsed a
dual-touch (both barriers inside one day's range) to a conservative stop —
on any other day the untouched barrier was outside the day's range and no
minute ordering can change the outcome. So this study re-resolves exactly
the dual-touch exit days on 1-min paths and reports:

    disagreement_rate  — share of dual-touch exits where the minute path hit
                         the TARGET first (daily said stop);
    tie_break_bias_pts — win-rate points the daily grid loses to the
                         conservative collapse, per §4.7 criterion 6
                         (<= 5 points, else re-base on the 1-min subset).

Adjustment-seam guard: Upstox minute bars are back-adjusted at source; the
daily panel is back-adjusted by our factor table. Exit days where the two
closes disagree by > 2% are skipped and counted (factor-table gaps must not
masquerade as tie-break disagreements).

Run: python -m scripts.v6_tiebreak_study
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.intraday import ROOT
from src.v6.labeler import all_cells, label_universe
from src.v6.panel import load_panel
from src.v6.universe import in_universe_mask, monthly_pit_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("v6.tiebreak")

BARS_DIR = ROOT / "data" / "bars"
OUT = ROOT / "reports" / "v6" / "tiebreak_study.json"
ATR_SANITY_MAX = 0.10
CLOSE_MISMATCH_MAX = 0.02


def _minute_day(symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    f = BARS_DIR / symbol / f"{day.strftime('%Y-%m')}.parquet"
    if not f.exists():
        return pd.DataFrame()
    bars = pd.read_parquet(f, columns=["ts", "high", "low", "close"])
    return bars[bars["ts"].dt.normalize() == day.normalize()]


def _first_hit_minute(bars: pd.DataFrame, target: float, stop: float) -> str:
    """Walk one exit day's minute bars; same-bar dual-touch stays a stop."""
    for h, l in zip(bars["high"], bars["low"]):  # noqa: E741
        hit_t, hit_s = h >= target, l <= stop
        if hit_s:            # includes same-minute dual-touch => stop
            return "stop"
        if hit_t:
            return "target"
    return "none"            # barriers not reached on minute path (bar gaps)


def main() -> None:
    panel = load_panel()
    universe = monthly_pit_universe(panel)
    mask = in_universe_mask(panel, universe) & (panel["atr_pct"] <= ATR_SANITY_MAX)
    bar_syms = sorted(p.name for p in BARS_DIR.iterdir() if p.is_dir())
    mask &= panel["symbol"].isin(bar_syms)
    logger.info("%d bar-store symbols in universe rows: %d", len(bar_syms), mask.sum())

    close_by_key = panel.set_index(["symbol", "date"])["close"]
    total, dual_total, flipped, skipped_seam, unresolved = 0, 0, 0, 0, 0
    per_cell = {}
    for cell in all_cells():
        labels = label_universe(panel, mask, cell, side="long")
        if labels.empty:
            continue
        total += len(labels)
        dual = labels[labels["dual_touch"]]
        cell_flips = 0
        for r in dual.itertuples(index=False):
            sig_close = close_by_key.get((r.symbol, pd.Timestamp(r.signal_date)))
            if sig_close is None:
                continue
            atr_abs = r.atr_pct * sig_close
            target = r.entry + cell.a * atr_abs
            stop = r.entry - cell.b * atr_abs
            bars = _minute_day(r.symbol, pd.Timestamp(r.exit_date))
            if bars.empty:
                unresolved += 1
                continue
            daily_close = close_by_key.get((r.symbol, pd.Timestamp(r.exit_date)))
            if daily_close and abs(bars["close"].iloc[-1] / daily_close - 1) > CLOSE_MISMATCH_MAX:
                skipped_seam += 1
                continue
            if _first_hit_minute(bars, target, stop) == "target":
                cell_flips += 1
        dual_total += len(dual)
        flipped += cell_flips
        per_cell[cell.key] = {"n": int(len(labels)), "dual": int(len(dual)),
                              "flips": int(cell_flips),
                              "bias_pts": round(100 * cell_flips / max(len(labels), 1), 3)}
        logger.info("%s: n=%d dual=%d flips=%d", cell.key, len(labels), len(dual), cell_flips)

    summary = {
        "generated": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "trades_total": total, "dual_touch_total": dual_total,
        "flipped_to_target_on_1min": flipped,
        "disagreement_rate_of_dual": round(flipped / max(dual_total, 1), 4),
        "tie_break_bias_pts_pooled": round(100 * flipped / max(total, 1), 3),
        "skipped_adjustment_seam": skipped_seam,
        "unresolved_no_bars": unresolved,
        "per_cell": per_cell,
        "criterion_6_threshold_pts": 5.0,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    logger.info("tie-break bias pooled: %.2f pts (criterion 6: <= 5) -> %s",
                summary["tie_break_bias_pts_pooled"], OUT)


if __name__ == "__main__":
    main()
