# PLAN_v6 Phase 0 — Feasibility Measurement Report

Generated 2026-07-08T11:55:31+05:30 · grid run 2026-07-08T11:49:09+05:30 (153s) · all numbers `[MEASURED]` unless tagged otherwise

## VERDICT — against the frozen §4.7 v2 (approved 2026-07-08, pre-measurement)

### ❌ KILL — no geometry cell satisfies criteria 1–4

Per the frozen contract: *"selective-signal-by-geometry does not work long-only on NSE at ≤5-day holds."* The pre-registered fallback (F24) applies: default = re-contract to asymmetric risk/reward (~45–60% WR, ≥2:1 payoff) under a fresh pre-registered gate; stopping outright is the user's election. **No threshold tuning is legitimate past this point.**

| # | Criterion (frozen) | Result |
|---|---|---|
| 1 | ≥1 cell pooled WR ∈ [0.72, 0.79] | **0 cells** (max pooled WR = 0.711) |
| 2 | plateau + halves ±3pts + worst year ≥ 0.70 | halves pass: 48; worst-year pass: 3; plateau pass: 12 |
| 3 | stop-hit ≥ 10% AND time-exit ≤ 40% | **3 cells** |
| 4 | empirical δ ≤ 4.0 (1×) and ≤ 6.0 (2×) pts | **0 cells** (min δ@1× = 6.3 pts) |
| 5 | expected loss beyond stop ≤ 25% of stop distance | 30 cells pass |
| 6 | tie-break bias ≤ 5 pts | **0.012 pts** — PASS |

## Why it fails — the two binding walls

- **Band (criterion 1):** best cells are a=0.5 at H=5, pooled WR 0.711 — the [0.72, 0.79] band is never reached. Longer horizons help (H=5 ≫ H=2) but the whole surface tops out ~0.71.
- **Cost wall (criterion 4):** minimum empirical δ across ALL 48 cells is 6.3 pts @1× (frozen ceiling 4.0). The payoff asymmetry does the damage: wide-stop/narrow-target cells buy win rate at the price of catastrophic loss-per-stop, and the measured exit mix prices it.

## §4.3 grid — top cells by pooled WR (long side, PIT universe)

| cell | n | WR pooled | half1/half2 | worst yr | tgt/stop/time | δ@1× | δ@2× | eff. edge req | theory floor |
|---|---|---|---|---|---|---|---|---|---|
| a0.5_b5_h5 | 62,174 | **0.711** | 0.702/0.719 | 0.707 | 0.69/0.01/0.30 | 9.1 | 10.9 | 9.1 | 0.91 |
| a0.5_b4_h5 | 62,174 | **0.710** | 0.701/0.719 | 0.706 | 0.69/0.02/0.29 | 9.2 | 11.0 | 9.2 | 0.89 |
| a0.5_b3_h5 | 62,174 | **0.708** | 0.699/0.717 | 0.704 | 0.69/0.05/0.26 | 9.5 | 11.3 | 9.5 | 0.86 |
| a0.5_b2_h5 | 62,174 | **0.698** | 0.687/0.709 | 0.691 | 0.68/0.14/0.18 | 10.0 | 11.8 | 10.2 | 0.80 |
| a0.5_b5_h3 | 62,377 | **0.645** | 0.633/0.657 | 0.639 | 0.60/0.00/0.39 | 11.3 | 13.6 | 15.5 | 0.91 |
| a0.5_b4_h3 | 62,377 | **0.644** | 0.632/0.657 | 0.638 | 0.60/0.01/0.39 | 11.4 | 13.7 | 15.6 | 0.89 |

Geometry-exercised cells (criterion 3 pass) and their gap to the contract:

| cell | WR | δ@1× | short-mirror WR | asymmetry |
|---|---|---|---|---|
| a0.5_b2_h5 | 0.698 | 10.0 | 0.727 | -2.9 pts |
| a0.75_b2_h5 | 0.612 | 9.3 | 0.639 | -2.7 pts |
| a1.0_b2_h5 | 0.560 | 8.4 | 0.582 | -2.3 pts |

## §4.1 data basis

- Panel: 1,543,046 rows, 2,906 symbols, CA-adjusted (170 factors), trailing ATR(14).
- PIT universe (GO basis, F20): top-100 by trailing 6-mo median turnover, monthly; 31 months, 177 unique symbols — rotation confirms survivorship-freeness.
- In-universe signal rows: 62,785; ATR-sanity exclusions (>10% — unadjusted-CA suspects): 295.
- **ATR% distribution (unknown #3 resolved): median 2.71%, p10 1.73%, p90 4.45%** — wider than the plan's assumed 1.2–2.2%.
- Disclosures: user 16-list not provided → cut skipped (report-only, F8); surveillance exclusion is a circuit-lock structural proxy (unknown #12 open); index-reconstitution cross-check cut not built (unknown #14 open) — neither affects the GO basis.

## §4.4 costs (frozen earlier today)

- c(₹1L) = 0.3932% base / 0.4932% @2× stress; minimum viable size ₹31,828 (F21, frozen); slippage constant provisional (unknown #15).

## §4.5 gap risk (unknown #5 resolved)

- Overnight |gap| > 1/2/3×ATR on 62,785 universe rows: 2.16% / 0.34% / 0.05%.
- Gap-through stops: 13.8% of stop exits; loss beyond stop (of stop distance): mean 0.94, p95 5.17 — **stops are emphatically not max losses at this horizon.**
- Split (F17): 1-night n=8093, mean beyond 1.27; weekend/multi-night n=7972, mean 0.60.
- Circuit-lock tail (F22): 0.11% of stop exits landed on an all-day locked session.

## §4.2 tie-break bias (unknown #6 resolved)

- Dual-touch trades: 211 of 1,203,328 (0.0175%); 67% flip to target on 1-min truth → pooled bias **0.012 pts** (criterion 6 ≤ 5: PASS). Daily-OHLC results stand.

## §4.6 / §4.6b probes

- GDELT (unknown #7): PENDING — probe running in background.
- Earnings PIT (unknown #8): announcement coverage 49%; board-meeting (true PIT) coverage 80% (threshold 90%; median intimation lead 14.0 days).

## What the fallback would inherit

Assets that survive the KILL: the panel/universe/labeler/grid machinery (re-parameterizable to asymmetric R:R in hours), the frozen cost model, the measured gap tails, and the probe outcomes. A re-contracted evaluation needs only a fresh pre-registered gate before it runs.

---
*Every number above was computed by the pipeline that will be audited, on the PIT universe, with the gate frozen before measurement. The KILL fired exactly as designed: in days, at compute cost, before any capital.*
