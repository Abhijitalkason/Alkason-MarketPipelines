# PLAN_v6 Phase 0 — Feasibility Measurement Report

Generated 2026-07-10T11:58:23+05:30 · grid run 2026-07-10T11:53:44+05:30 (141s) · all numbers `[MEASURED]` unless tagged otherwise

## Revision note (review 2, 2026-07-10) — evidence-quality repair, verdict unchanged

An independent review of PLAN_v6 vs. its implementation found the corporate-action table covered bonuses only (170/170 rows); splits and demergers were absent, so a handful of holds resolved as fake catastrophic stop-outs (e.g. NAUKRI/DRREDDY/SHRIRAMFIN 1:5 splits mis-priced as ≈−80% losses). A discontinuity detector (F25) was added and the grid re-run. **The KILL verdict is unchanged** — both binding walls move by less than a point, an order of magnitude short of flipping either:

| Quantity | Before | After | Bar |
|---|---|---|---|
| Criterion 1 — max pooled WR | 0.7106 | **0.7108** | ≥ 0.72 |
| Criterion 4 — min required δ@1× | 6.27 pts | **5.55 pts** | ≤ 4.0 pts |
| Cells passing criteria 1–4 | 0 | **0** | ≥ 1 (GO) |
| Criterion 5 (non-binding) pass count | 30/48 | **48/48** | — |

The largest correction is to the §4.5 gap-risk numbers, which the CSV-bonus gap had materially inflated (see the revised §4.5 section below) — those numbers are an asset the F24 fallback explicitly inherits, so this was the one place the old evidence file was actively misleading, not merely conservative. A residual, disclosed gap remains: the detector's threshold does not catch every corporate action (see §4.1); an explicit tripwire now surfaces what's left rather than hiding it.

## VERDICT — against the frozen §4.7 v2 (approved 2026-07-08, pre-measurement)

### ❌ KILL — no geometry cell satisfies criteria 1–4

Per the frozen contract: *"selective-signal-by-geometry does not work long-only on NSE at ≤5-day holds."* The pre-registered fallback (F24) applies: default = re-contract to asymmetric risk/reward (~45–60% WR, ≥2:1 payoff) under a fresh pre-registered gate; stopping outright is the user's election. **No threshold tuning is legitimate past this point.**

| # | Criterion (frozen) | Result |
|---|---|---|
| 1 | ≥1 cell pooled WR ∈ [0.72, 0.79] | **0 cells** (max pooled WR = 0.711) |
| 2 | plateau + halves ±3pts + worst year ≥ 0.70 | halves pass: 48; worst-year pass: 3; plateau pass: 12 |
| 3 | stop-hit ≥ 10% AND time-exit ≤ 40% | **3 cells** |
| 4 | empirical δ ≤ 4.0 (1×) and ≤ 6.0 (2×) pts | **0 cells** (min δ@1× = 5.5 pts) |
| 5 | expected loss beyond stop ≤ 25% of stop distance | 48 cells pass |
| 6 | tie-break bias ≤ 5 pts | **0.013 pts** — PASS |

## Why it fails — the two binding walls

- **Band (criterion 1):** best cells are a=0.5 at H=5, pooled WR 0.711 — the [0.72, 0.79] band is never reached. Longer horizons help (H=5 ≫ H=2) but the whole surface tops out ~0.71.
- **Cost wall (criterion 4):** minimum empirical δ across ALL 48 cells is 5.5 pts @1× (frozen ceiling 4.0). The payoff asymmetry does the damage: wide-stop/narrow-target cells buy win rate at the price of catastrophic loss-per-stop, and the measured exit mix prices it.
- **The two walls fail independently.** The band miss is small — 0.9 pts, plausibly within day-clustered sampling noise over a 31-month window — so criterion 1 alone is not a hard no. Criterion 4 is not close: the best available δ is 1.4× the ceiling. A pooled-estimate CI computation was out of scope for this review, but no plausible CI on the WR point estimate closes a 2×+ multiple on required edge — the cost wall carries the KILL on its own.

## §4.3 grid — top cells by pooled WR (long side, PIT universe)

| cell | n | WR pooled | half1/half2 | worst yr | tgt/stop/time | δ@1× | δ@2× | eff. edge req | theory floor |
|---|---|---|---|---|---|---|---|---|---|
| a0.5_b5_h5 | 62,456 | **0.711** | 0.702/0.720 | 0.707 | 0.69/0.01/0.30 | 8.7 | 10.6 | 8.9 | 0.91 |
| a0.5_b4_h5 | 62,456 | **0.710** | 0.701/0.719 | 0.706 | 0.69/0.02/0.29 | 8.9 | 10.7 | 9.0 | 0.89 |
| a0.5_b3_h5 | 62,456 | **0.708** | 0.699/0.718 | 0.703 | 0.69/0.05/0.26 | 9.2 | 11.0 | 9.2 | 0.86 |
| a0.5_b2_h5 | 62,456 | **0.698** | 0.687/0.710 | 0.691 | 0.68/0.14/0.18 | 9.7 | 11.6 | 10.2 | 0.80 |
| a0.5_b5_h3 | 62,660 | **0.645** | 0.632/0.658 | 0.639 | 0.60/0.00/0.39 | 11.1 | 13.4 | 15.5 | 0.91 |
| a0.5_b4_h3 | 62,660 | **0.645** | 0.632/0.657 | 0.638 | 0.60/0.01/0.39 | 11.2 | 13.5 | 15.5 | 0.89 |

Geometry-exercised cells (criterion 3 pass) and their gap to the contract:

| cell | WR | δ@1× | short-mirror WR | asymmetry |
|---|---|---|---|---|
| a0.5_b2_h5 | 0.698 | 9.7 | 0.727 | -2.9 pts |
| a0.75_b2_h5 | 0.612 | 8.8 | 0.639 | -2.6 pts |
| a1.0_b2_h5 | 0.560 | 7.8 | 0.582 | -2.2 pts |

## §4.1 data basis

- Panel: 1,543,046 rows, 2,906 symbols, CA-adjusted (170 CSV bonus factors + 254 detected residual splits/demergers, F25), trailing ATR(14).
- PIT universe (GO basis, F20): top-100 by trailing 6-mo median turnover, monthly; 31 months, 177 unique symbols — rotation confirms survivorship-freeness.
- In-universe signal rows: 62,966; ATR-sanity exclusions (>10% — unadjusted-CA suspects): 114.
- **ATR% distribution (unknown #3 resolved): median 2.71%, p10 1.73%, p90 4.46%** — wider than the plan's assumed 1.2–2.2%.
- **F19 disclosure (review 2):** 0.12% of held trades (mean across cells) straddle a corporate-action ex-date — the count PLAN_v6 §4.1 originally promised but the implementation never computed.
- **Residual-artifact tripwire (F25, review 2):** the CA detector flags discontinuities outside [0.5×, 2.0×] on principled a-priori grounds (NSE price bands rarely exceed 20% for liquid names), chosen before the fix was validated against known cases — not retuned afterward. Known residual gap: an action whose ratio lands inside that band (e.g. a clean 2:1 split plus a small same-day return) can still slip through. 9 such (symbol, exit-date) events remain across the pooled grid (441 label rows) beyond the −30% single-trade tripwire — some are plausibly genuine large drawdowns (e.g. a well-documented 2024 regulatory action), not all are artifacts; sample: BAJFINANCE/2025-06-16, COCHINSHIP/2024-01-10, MAZDOCK/2024-12-27, PAYTM/2024-02-14, PERSISTENT/2024-03-28, SILVERBEES/2026-02-02….
- **Delisting/merger tail (R4, review 2):** 7 universe symbols have a panel series ending >30 days before the panel's own end date (mergers, renames, delistings): GMRINFRA, IBULHSGFIN, LTIM, PEL, SWANENERGY, TATAMOTORS, ZOMATO.
- Disclosures: user 16-list not provided → cut skipped (report-only, F8); surveillance exclusion is a circuit-lock structural proxy (unknown #12 open); index-reconstitution cross-check cut not built (unknown #14 open) — neither affects the GO basis.

## §4.4 costs (frozen earlier today)

- c(₹1L) = 0.3932% base / 0.4932% @2× stress; minimum viable size ₹31,828 (F21, frozen); slippage constant provisional (unknown #15).

## §4.5 gap risk (unknown #5 resolved)

**Revised in review 2** — the CSV-bonus gap (F25) put several corporate-action artifacts into this table (a fake ≈−80% split-mispriced "stop" contributes as much as ~14 organic gap-throughs); the F17 one_night/multi_night split was also pooling the same calendar event once per grid cell (up to 48×). Both are fixed below; the old p95 was **5.17×** stop distance, now **1.30×** — see the revision note above.

- Overnight |gap| > 1/2/3×ATR on 62,966 universe rows: 2.16% / 0.34% / 0.05%.
- Gap-through stops: 13.1% of stop exits; loss beyond stop (of stop distance): mean 0.41, p95 1.30 — **stops are emphatically not max losses at this horizon.**
- Split (F17, now deduped by unique (symbol, exit-date) — see note above): 1-night n=502, mean beyond 0.24; weekend/multi-night n=421, mean 0.31.
- Circuit-lock tail (F22): 0.11% of stop exits landed on an all-day locked session.

## §4.2 tie-break bias (unknown #6 resolved)

- Dual-touch trades: 214 of 1,207,024 (0.0177%); 74% flip to target on 1-min truth → pooled bias **0.013 pts** (criterion 6 ≤ 5: PASS). Daily-OHLC results stand.

## §4.6 / §4.6b probes

- GDELT (unknown #7): symbol-day coverage 94% on 17 sampled symbol-weeks (threshold 30%) — probe halted at the user's stop election; the partial sample already clears the threshold, so news backtestability was NOT the binding constraint.
- Earnings PIT (unknown #8): announcement coverage 49%; board-meeting (true PIT) coverage 80% (threshold 90%; median intimation lead 14.0 days).

## What the fallback would inherit

Assets that survive the KILL: the panel/universe/labeler/grid machinery (re-parameterizable to asymmetric R:R in hours), the frozen cost model, the corrected gap tails (review 2), and the probe outcomes. A re-contracted evaluation needs only a fresh pre-registered gate before it runs — and should still watch the residual-artifact tripwire (§4.1), since it is disclosed, not eliminated.

---
*Every number above was computed by the pipeline that will be audited, on the PIT universe, with the gate frozen before measurement. The KILL fired exactly as designed: in days, at compute cost, before any capital.*
