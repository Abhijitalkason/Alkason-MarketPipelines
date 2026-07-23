# Phase 0 §4.4 — Delivery cost curve c(size) — broker: Upstox

Generated: 2026-07-13T09:58:15+05:30 · constants: upstox.com/brokerage-charges [VERIFIED 2026-07-08]

| Size (₹) | Cost (₹) | c(size) % | ex-slippage % | 2× stress % | flat-fee share |
|---|---|---|---|---|---|
| 1,000 | 74.02 | 7.4024 | 7.3024 | 7.5024 | 95.6% |
| 10,000 | 103.04 | 1.0304 | 0.9304 | 1.1304 | 68.7% |
| 25,000 | 151.41 | 0.6056 | 0.5056 | 0.7056 | 46.8% |
| 50,000 | 232.02 | 0.464 | 0.364 | 0.564 | 30.5% |
| 100,000 | 393.25 | 0.3932 | 0.2932 | 0.4932 | 18.0% |
| 1,000,000 | 3,295.25 | 0.3295 | 0.2295 | 0.4295 | 2.1% |
| 10,000,000 | 32,315.34 | 0.3232 | 0.2232 | 0.4232 | 0.2% |

**Minimum viable size (FROZEN, F21): ₹31,828**
— rule: smallest size where flat fees (brokerage+DP+GST) <= ex-slippage variable cost; frozen per F21, slippage-independent by construction. Sizes below it are plumbing-test-only and
excluded from GO/KILL (§4.4).

**Reference size for gate arithmetic: ₹100,000** —
c = **0.3932%** base, **0.4932%** at 2× slippage stress.
These are the `c` inputs to the §4.3 empirical-δ columns.

Slippage: PROVISIONAL 5bps/side [MEASURE — unknown #15 open-print study] — the floor rule excludes slippage
so the measured constant cannot move the frozen floor.

Per-symbol maximum size (pre-open auction participation, F18): pending
auction-volume data `[MEASURE — unknown #13]`; not needed for the floor.
