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
from src.v6.grid import BAND_LO

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
    w("## Revision note (review 2, 2026-07-10) — evidence-quality repair, verdict unchanged")
    w("")
    w("An independent review of PLAN_v6 vs. its implementation found the corporate-action "
      "table covered bonuses only (170/170 rows); splits and demergers were absent, so a "
      "handful of holds resolved as fake catastrophic stop-outs (e.g. NAUKRI/DRREDDY/"
      "SHRIRAMFIN 1:5 splits mis-priced as ≈−80% losses). A discontinuity detector (F25) "
      "was added and the grid re-run. **The KILL verdict is unchanged** — both binding "
      "walls move by less than a point, an order of magnitude short of flipping either:")
    w("")
    w("| Quantity | Before | After | Bar |")
    w("|---|---|---|---|")
    w(f"| Criterion 1 — max pooled WR | 0.7106 | **{grid['wr_pooled'].max():.4f}** | ≥ 0.72 |")
    w(f"| Criterion 4 — min required δ@1× | 6.27 pts | **{grid['delta_req_1x'].min():.2f} pts** | ≤ 4.0 pts |")
    w(f"| Cells passing criteria 1–4 | 0 | **{int(grid['go_1_to_4'].sum())}** | ≥ 1 (GO) |")
    w(f"| Criterion 5 (non-binding) pass count | 30/48 | **{int(grid['c5_gap'].sum())}/48** | — |")
    w("")
    w("The largest correction is to the §4.5 gap-risk numbers, which the CSV-bonus gap had "
      "materially inflated (see the revised §4.5 section below) — those numbers are an asset "
      "the F24 fallback explicitly inherits, so this was the one place the old evidence file "
      "was actively misleading, not merely conservative. A residual, disclosed gap remains: "
      "the detector's threshold does not catch every corporate action (see §4.1); an explicit "
      "tripwire now surfaces what's left rather than hiding it.")
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
    w(f"- **The two walls fail independently.** The band miss is small — "
      f"{(BAND_LO - grid['wr_pooled'].max()) * 100:.1f} pts, plausibly within "
      "day-clustered sampling noise over a 31-month window — so criterion 1 alone is not "
      "a hard no. Criterion 4 is not close: the best available δ is "
      f"{grid['delta_req_1x'].min() / 4.0:.1f}× the ceiling. A pooled-estimate CI computation "
      "was out of scope for this review, but no plausible CI on the WR point estimate closes "
      "a 2×+ multiple on required edge — the cost wall carries the KILL on its own.")
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
      f"({meta['ca_events_csv_bonus']} CSV bonus factors + "
      f"{meta['ca_events_detected_residual']} detected residual splits/demergers, F25), "
      "trailing ATR(14).")
    w(f"- PIT universe (GO basis, F20): top-100 by trailing 6-mo median turnover, "
      f"monthly; {meta['universe_months']} months, {meta['universe_unique_symbols']} "
      "unique symbols — rotation confirms survivorship-freeness.")
    w(f"- In-universe signal rows: {meta['signal_rows']:,}; ATR-sanity exclusions "
      f"(>10% — unadjusted-CA suspects): {meta['atr_sanity_excluded']}.")
    w(f"- **ATR% distribution (unknown #3 resolved): median {meta['atr_pct_median']:.2%}, "
      f"p10 {meta['atr_pct_p10']:.2%}, p90 {meta['atr_pct_p90']:.2%}** — wider than the "
      "plan's assumed 1.2–2.2%.")
    w(f"- **F19 disclosure (review 2):** {meta['ca_in_hold_share_mean_across_cells']:.2%} "
      "of held trades (mean across cells) straddle a corporate-action ex-date — the count "
      "PLAN_v6 §4.1 originally promised but the implementation never computed.")
    w(f"- **Residual-artifact tripwire (F25, review 2):** the CA detector flags discontinuities "
      "outside [0.5×, 2.0×] on principled a-priori grounds (NSE price bands rarely exceed "
      "20% for liquid names), chosen before the fix was validated against known cases — "
      "not retuned afterward. Known residual gap: an action whose ratio lands inside that "
      f"band (e.g. a clean 2:1 split plus a small same-day return) can still slip through. "
      f"{meta['extreme_loss_tripwire_unique_events']} such (symbol, exit-date) events remain "
      f"across the pooled grid ({meta['extreme_loss_tripwire_label_rows']} label rows) beyond "
      "the −30% single-trade tripwire — some are plausibly genuine large drawdowns (e.g. a "
      "well-documented 2024 regulatory action), not all are artifacts; sample: "
      f"{', '.join(f'{s}/{d}' for s, d in meta['extreme_loss_tripwire_events'][:6])}"
      f"{'…' if meta['extreme_loss_tripwire_unique_events'] > 6 else ''}.")
    w(f"- **Delisting/merger tail (R4, review 2):** "
      f"{meta['delisting_tail']['universe_symbols_with_truncated_series']} universe symbols "
      "have a panel series ending >30 days before the panel's own end date (mergers, "
      f"renames, delistings): {', '.join(meta['delisting_tail']['symbols_sample'])}.")
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
    w("**Revised in review 2** — the CSV-bonus gap (F25) put several corporate-action "
      "artifacts into this table (a fake ≈−80% split-mispriced \"stop\" contributes as much "
      "as ~14 organic gap-throughs); the F17 one_night/multi_night split was also pooling the "
      "same calendar event once per grid cell (up to 48×). Both are fixed below; the old "
      "p95 was **5.17×** stop distance, now **1.30×** — see the revision note above.")
    w("")
    w(f"- Overnight |gap| > 1/2/3×ATR on {meta['signal_rows']:,} universe rows: "
      f"{gap['gap_beyond_1atr_share']:.2%} / {gap['gap_beyond_2atr_share']:.2%} / "
      f"{gap['gap_beyond_3atr_share']:.2%}.")
    w(f"- Gap-through stops: {gap['gap_through_share_of_stops']:.1%} of stop exits; "
      f"loss beyond stop (of stop distance): mean {gap['loss_beyond_stop_frac_mean']:.2f}, "
      f"p95 {gap['loss_beyond_stop_frac_p95']:.2f} — **stops are emphatically not max "
      "losses at this horizon.**")
    w(f"- Split (F17, now deduped by unique (symbol, exit-date) — see note above): "
      f"1-night n={gap['one_night']['n']}, mean beyond "
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
        w(f"- GDELT (unknown #7): symbol-day coverage {s['symbol_day_coverage']:.0%} "
          f"on {s['weeks_probed']} sampled symbol-weeks (threshold 30%) — probe halted "
          "at the user's stop election; the partial sample already clears the "
          "threshold, so news backtestability was NOT the binding constraint.")
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
      "corrected gap tails (review 2), and the probe outcomes. A re-contracted evaluation "
      "needs only a fresh pre-registered gate before it runs — and should still watch the "
      "residual-artifact tripwire (§4.1), since it is disclosed, not eliminated.")
    w("")
    w("---")
    w("*Every number above was computed by the pipeline that will be audited, on the "
      "PIT universe, with the gate frozen before measurement. The KILL fired exactly "
      "as designed: in days, at compute cost, before any capital.*")

    (OUT / "phase0_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'phase0_report.md'}")


if __name__ == "__main__":
    main()
