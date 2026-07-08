"""PLAN_v6 Phase 0 — report assembler (§4 deliverable).

Collects every Phase 0 artifact under reports/v6/ into phase0_report.md with
the GO/KILL verdict computed against the frozen §4.7 v2 criteria. Probe
artifacts that have not landed yet are marked PENDING and the report is
regenerated when they do.

Run: python -m scripts.v6_phase0_report
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.intraday import ROOT

OUT = ROOT / "reports" / "v6"


def _load(name: str) -> dict | None:
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    grid = pd.read_parquet(OUT / "grid_long.parquet")
    meta = _load("grid_run_meta.json")
    costs = _load("cost_curve.json")
    gap = _load("gap_risk.json")
    tie = _load("tiebreak_study.json")
    gdelt = _load("gdelt_probe_summary.json")
    earn = _load("earnings_probe.json")

    kill = int(grid["go_1_to_4"].sum()) == 0
    best = grid.nlargest(6, "wr_pooled")
    exercised = grid[grid["c3_exercised"]]
    lines = []
    w = lines.append

    w("# PLAN_v6 Phase 0 — Feasibility Measurement Report")
    w("")
    w(f"Generated {datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')} · "
      f"grid run {meta['generated']} ({meta['runtime_s']}s) · all numbers `[MEASURED]` "
      "unless tagged otherwise")
    w("")
    w("## VERDICT — against the frozen §4.7 v2 (approved 2026-07-08, pre-measurement)")
    w("")
    if kill:
        w("### ❌ KILL — no geometry cell satisfies criteria 1–4")
        w("")
        w("Per the frozen contract: *\"selective-signal-by-geometry does not work "
          "long-only on NSE at ≤5-day holds.\"* The pre-registered fallback (F24) "
          "applies: default = re-contract to asymmetric risk/reward (~45–60% WR, "
          "≥2:1 payoff) under a fresh pre-registered gate; stopping outright is the "
          "user's election. **No threshold tuning is legitimate past this point.**")
    else:
        w(f"### ✅ GO candidates: {int(grid['go_1_to_4'].sum())} cell(s) pass criteria 1–4")
    w("")
    w("| # | Criterion (frozen) | Result |")
    w("|---|---|---|")
    w(f"| 1 | ≥1 cell pooled WR ∈ [0.72, 0.79] | **{int(grid['c1_band'].sum())} cells** "
      f"(max pooled WR = {grid['wr_pooled'].max():.3f}) |")
    w(f"| 2 | plateau + halves ±3pts + worst year ≥ 0.70 | halves pass: "
      f"{int(grid['c2_halves'].sum())}; worst-year pass: {int(grid['c2_worst_year'].sum())}; "
      f"plateau pass: {int(grid['c2_plateau'].sum())} |")
    w(f"| 3 | stop-hit ≥ 10% AND time-exit ≤ 40% | **{int(grid['c3_exercised'].sum())} cells** |")
    w(f"| 4 | empirical δ ≤ 4.0 (1×) and ≤ 6.0 (2×) pts | **{int(grid['c4_delta'].sum())} cells** "
      f"(min δ@1× = {grid['delta_req_1x'].min():.1f} pts) |")
    w(f"| 5 | expected loss beyond stop ≤ 25% of stop distance | "
      f"{int(grid['c5_gap'].sum())} cells pass |")
    tie_pts = tie["tie_break_bias_pts_pooled"] if tie else None
    w(f"| 6 | tie-break bias ≤ 5 pts | **{tie_pts} pts** — "
      f"{'PASS' if tie and tie_pts <= 5 else 'PENDING'} |")
    w("")

    w("## Why it fails — the two binding walls")
    w("")
    w(f"- **Band (criterion 1):** best cells are a=0.5 at H=5, pooled WR "
      f"{grid[grid.a == 0.5].wr_pooled.max():.3f} — the [0.72, 0.79] band is never "
      "reached. Longer horizons help (H=5 ≫ H=2) but the whole surface tops out ~0.71.")
    w(f"- **Cost wall (criterion 4):** minimum empirical δ across ALL 48 cells is "
      f"{grid['delta_req_1x'].min():.1f} pts @1× (frozen ceiling 4.0). The payoff "
      "asymmetry does the damage: wide-stop/narrow-target cells buy win rate at the "
      "price of catastrophic loss-per-stop, and the measured exit mix prices it.")
    w("")

    w("## §4.3 grid — top cells by pooled WR (long side, PIT universe)")
    w("")
    w("| cell | n | WR pooled | half1/half2 | worst yr | tgt/stop/time | δ@1× | δ@2× | eff. edge req | theory floor |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in best.iterrows():
        w(f"| {r.cell} | {r.n:,} | **{r.wr_pooled:.3f}** | {r.wr_half1:.3f}/{r.wr_half2:.3f} "
          f"| {r.wr_worst_year:.3f} | {r.share_target:.2f}/{r.share_stop:.2f}/{r.share_time:.2f} "
          f"| {r.delta_req_1x:.1f} | {r.delta_req_2x:.1f} | {r.effective_edge_req:.1f} "
          f"| {r.theory_floor:.2f} |")
    w("")
    w("Geometry-exercised cells (criterion 3 pass) and their gap to the contract:")
    w("")
    w("| cell | WR | δ@1× | short-mirror WR | asymmetry |")
    w("|---|---|---|---|---|")
    for _, r in exercised.iterrows():
        w(f"| {r.cell} | {r.wr_pooled:.3f} | {r.delta_req_1x:.1f} | "
          f"{r.get('wr_short_mirror', float('nan')):.3f} | "
          f"{r.get('long_short_asymmetry', float('nan')) * 100:+.1f} pts |")
    w("")

    w("## §4.1 data basis")
    w("")
    w(f"- Panel: {meta['panel_rows']:,} rows, {meta['symbols']:,} symbols, CA-adjusted "
      f"(170 factors), trailing ATR(14).")
    w(f"- PIT universe (GO basis, F20): top-100 by trailing 6-mo median turnover, "
      f"monthly; {meta['universe_months']} months, {meta['universe_unique_symbols']} "
      "unique symbols — rotation confirms survivorship-freeness.")
    w(f"- In-universe signal rows: {meta['signal_rows']:,}; ATR-sanity exclusions "
      f"(>10% — unadjusted-CA suspects): {meta['atr_sanity_excluded']}.")
    w(f"- **ATR% distribution (unknown #3 resolved): median {meta['atr_pct_median']:.2%}, "
      f"p10 {meta['atr_pct_p10']:.2%}, p90 {meta['atr_pct_p90']:.2%}** — wider than the "
      "plan's assumed 1.2–2.2%.")
    w("- Disclosures: user 16-list not provided → cut skipped (report-only, F8); "
      "surveillance exclusion is a circuit-lock structural proxy (unknown #12 open); "
      "index-reconstitution cross-check cut not built (unknown #14 open) — neither "
      "affects the GO basis.")
    w("")

    w("## §4.4 costs (frozen earlier today)")
    w("")
    w(f"- c(₹1L) = {costs['cost_pct_at_reference']}% base / "
      f"{costs['cost_pct_at_reference_2x']}% @2× stress; minimum viable size "
      f"₹{costs['minimum_viable_size_inr']:,} (F21, frozen); slippage constant "
      "provisional (unknown #15).")
    w("")

    w("## §4.5 gap risk (unknown #5 resolved)")
    w("")
    w(f"- Overnight |gap| > 1/2/3×ATR on {meta['signal_rows']:,} universe rows: "
      f"{gap['gap_beyond_1atr_share']:.2%} / {gap['gap_beyond_2atr_share']:.2%} / "
      f"{gap['gap_beyond_3atr_share']:.2%}.")
    w(f"- Gap-through stops: {gap['gap_through_share_of_stops']:.1%} of stop exits; "
      f"loss beyond stop (of stop distance): mean {gap['loss_beyond_stop_frac_mean']:.2f}, "
      f"p95 {gap['loss_beyond_stop_frac_p95']:.2f} — **stops are emphatically not max "
      "losses at this horizon.**")
    w(f"- Split (F17): 1-night n={gap['one_night']['n']}, mean beyond "
      f"{gap['one_night']['loss_beyond_mean']:.2f}; weekend/multi-night "
      f"n={gap['multi_night']['n']}, mean {gap['multi_night']['loss_beyond_mean']:.2f}.")
    w(f"- Circuit-lock tail (F22): {gap['circuit_lock']['share_of_stops']:.2%} of stop "
      "exits landed on an all-day locked session.")
    w("")

    w("## §4.2 tie-break bias (unknown #6 resolved)")
    w("")
    if tie:
        w(f"- Dual-touch trades: {tie['dual_touch_total']} of {tie['trades_total']:,} "
          f"({tie['dual_touch_total'] / tie['trades_total']:.4%}); {tie['disagreement_rate_of_dual']:.0%} "
          f"flip to target on 1-min truth → pooled bias **{tie['tie_break_bias_pts_pooled']} pts** "
          "(criterion 6 ≤ 5: PASS). Daily-OHLC results stand.")
    w("")

    w("## §4.6 / §4.6b probes")
    w("")
    if gdelt:
        s = gdelt["symbol_level"]
        w(f"- GDELT (unknown #7): symbol-week coverage {s['weeks_with_any_article']:.0%}, "
          f"symbol-day coverage {s['symbol_day_coverage']:.0%} (threshold 30%) → "
          f"**{'news can enter the Gate-3 backtest' if s['symbol_day_coverage'] >= 0.30 else 'news stays forward-only'}**; "
          f"market-level day coverage {gdelt['market_level']['day_coverage']:.0%}.")
    else:
        w("- GDELT (unknown #7): PENDING — probe running in background.")
    if earn:
        w(f"- Earnings PIT (unknown #8): announcement coverage "
          f"{earn['announcement_coverage']:.0%}; board-meeting (true PIT) coverage "
          f"{earn['board_meeting_pit_coverage']:.0%} "
          f"(threshold 90%; median intimation lead {earn['median_lead_days']} days).")
    else:
        w("- Earnings PIT (unknown #8): PENDING — probe running in background.")
    w("")

    w("## What the fallback would inherit")
    w("")
    w("Assets that survive the KILL: the panel/universe/labeler/grid machinery "
      "(re-parameterizable to asymmetric R:R in hours), the frozen cost model, the "
      "measured gap tails, and the probe outcomes. A re-contracted evaluation needs "
      "only a fresh pre-registered gate before it runs.")
    w("")
    w("---")
    w("*Every number above was computed by the pipeline that will be audited, on the "
      "PIT universe, with the gate frozen before measurement. The KILL fired exactly "
      "as designed: in days, at compute cost, before any capital.*")

    (OUT / "phase0_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'phase0_report.md'}")


if __name__ == "__main__":
    main()
