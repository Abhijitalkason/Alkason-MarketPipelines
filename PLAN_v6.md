# PLAN v6 — Daily/Swing Selective Signal System (1–5 Day Holds, NSE) — **v2.1**

> **Project:** Selective trading-signal system for NSE stocks at the 1–5 day (swing) horizon — **long-only in the cash/delivery segment** (see §2, review finding F3)
> **Goal contract:** ≥ **80% win rate on emitted signals** AND **positive post-cost expectancy per trade** — the win rate engineered by barrier geometry **verified to materialize on real paths before anything is built**, the expectancy delivered by model edge and proven through pre-registered gates.
> **Lineage:** PLAN.md (v1, killed) → PLAN_v3 (intraday, killed) → PLAN_v4 (intraday retarget, kill fired at Gate 2) → v6 (horizon change, the PLAN_v4 §6.5 response)
> **Status:** v2 — REVISED 2026-07-07 after adversarial deep review (19 findings, §10 amendment log). The v1 §4.7 thresholds approved earlier that day were found **arithmetically unsatisfiable** (F1/F2) and are superseded below. **§4.7 v2 APPROVED by user 2026-07-08** (after end-to-end re-verification, F24) — **thresholds are now FROZEN; Phase 0 may start.** Position size config-driven ₹1k→₹1Cr; universe = 16-list (report-only) + PIT (GO basis) + liquidity tier (robustness); repo stays uncommitted per user instruction.
> **Plan version:** 2026-07-08 (v2.2 — §4.7 v2 approved & frozen; F24 clarifications: effective-edge disclosure, grid-edge neighbor rule, pre-registered KILL fallback)
> **PHASE 0 OUTCOME `[MEASURED 2026-07-08]`: KILL** — 0/48 cells pass frozen criteria 1–4; see §4.7 outcome note and `reports/v6/phase0_report.md`.
> **PROJECT STATUS: STOPPED — user election 2026-07-08.** The pre-registered fallback (asymmetric R:R) was evaluated against the measured grid and declined: every cell's pre-cost expectancy is negative (best −0.026%/trade vs 0.393% costs) and realized payoff asymmetry collapses under 52–84% time-exit shares at H ≤ 5. Established conclusion: **barrier geometry cannot engineer an edge in NSE equities at any horizon from minutes to a week with this information set.** Data layers, cost model, and measurement machinery remain committed for any future thesis built on a different information source.

---

## 0. The No-Excuses Contract

Four binding rules, adopted because v1/v3/v4 each violated one:

1. **Measure before build.** Phase 0 (2–3 days, compute only) measures the load-bearing
   question — *does the geometry floor materialize at this horizon?* — before system work.
2. **Every number is tagged** `[MEASURED]` / `[VERIFY]` / `[MEASURE]`. No untagged estimates.
3. **Pre-registered GO/KILL, frozen before measurement.** The v1→v2 revision of §4.7
   happened *pre-measurement* under adversarial review — the only window in which
   changing thresholds is legitimate. Once Phase 0 runs, §4.7 v2 is immutable.
4. **Metric contract frozen:** win rate on emitted signals + post-cost expectancy,
   day-clustered statistics. Full-coverage "accuracy" stays banned.

**Honest priors (disclosed):** ~90% the geometry floor is real at this horizon; ~35–50%
that a positive post-cost edge exists — now with the required size made explicit: the
model must add **≈ 3–6 probability points on its fired subset** (corrected arithmetic,
§2), against an intraday-measured edge of ≈ 0; **~25–40% the full system passes every
gate.** No guarantee is claimed. What is guaranteed: every "no" arrives in days-to-weeks,
with evidence, at compute cost, before capital.

---

## 1. Lessons Register

| Plan | Failure | v6 response |
|---|---|---|
| v1 | "Maximum achievable accuracy" target; 22–40% ≈ baseline; yfinance data; dead FinBERT pipeline | Accuracy banned; selective contract; NSE-official data; timestamped news only |
| v3 | 89%-by-geometry premise never measured; contract never evaluated; 0.50 misread panic | Executable gates; contract metric computed by the pipeline itself |
| v4 | Built 4 weeks before a 1-day measurement killed the premise `[MEASURED 2026-07-03: baseline 0.51–0.575 vs theory 0.67–0.91; 75/98 time exits; model logloss 0.69 ≈ no edge; WR 51%, expectancy −0.27%]` | **Measure first (Phase 0).** Horizon change per v4's own §6.5 |
| v6 v1 (same day) | Deep review found: §2 required-edge arithmetic off 10×; §4.7 GO criterion unsatisfiable by any grid cell; shorts not executable in cash segment; Gates 3/4 statistically underpowered | This v2 — all 19 findings dispositioned in §10 |

**Carried assets (~70% reuse):** PIT bar/bhavcopy stores (bhavcopy = full NSE market,
~2,400 symbols × 763 sessions with delivery % `[MEASURED on disk]`), regime channel
(785 days backfilled), news capture (RSS+GDELT, live-verified), labeler/backtester/
gates/risk/serving, 27-test regime suite.

---

## 2. Why 1–5 Day Holds — and Every Constraint It Brings

**For it:**
- The measured intraday killer (time-barrier truncation) weakens: barriers become
  reachable — **but not uniformly**: an H-day range scales ≈ √H·ATR (1.4/1.7/2.2 ATR for
  H=2/3/5), so 4–5×ATR stops at H=2–3 recreate the v4 truncation trap. The genuinely
  testable region is roughly a ∈ {0.75, 1.0}, b ∈ {2, 3}, H = 5 (F13). Phase 0 measures
  stop-hit share per cell to prove the geometry is exercised, not truncated.
- **Cost-to-width, corrected (F1):** W = (a+b)·ATR_daily with ATR ≈ 1.2–2.2% `[MEASURE]`
  → W ≈ 3–14%; delivery round-trip c ≈ 0.28–0.35% at ≥₹1L `[VERIFY §4.4]` → required
  edge δ = c/W ≈ **2–6 probability points** for viable cells (NOT the "0.3–0.6" of v6
  v1 — that was a 10× arithmetic error). Still ~5–10× better than the measured 15–31
  points intraday, and in the range documented anomalies can plausibly supply — but it
  must be said plainly: **the model has to produce 3–6 points of real fired-subset edge,
  and the intraday measurement of this model family was ≈ zero.** That is the honest
  central risk of this plan (F10). Its payoff-asymmetry face: at b ≫ a, one stop-out
  erases ≈ b/a winners plus their costs — the empirical-δ criterion (F11) prices exactly
  this from the measured exit mix, which is why it replaces the closed-form c/W.
- The information we ingest lives here: post-event/news drift (documented at days-scale),
  delivery %, FII/DII, OI — daily series; the v4 regime channel plugs in unchanged.
- Competition is not speed-decided; data breadth is full-market.

**Against it (owned):**
- **LONG-ONLY (F3):** retail cannot hold overnight shorts in the NSE cash segment
  (intraday square-off is forced; SLB is institutional; stock futures have ₹5–15L+ lot
  values and a different cost model — incompatible with the ₹1k–₹1L size range). The
  tradeable system is long-only; short-side labeling exists **only as a drift
  diagnostic** (§4.3). A long-only system is regime-dependent by construction —
  the no-trade regime veto becomes load-bearing, and Phase 0 measures per-year floors.
- **Overnight gap risk:** stops are not maximum losses (§4.5 measures the tail,
  split 1-night vs weekend/multi-night, F17); sizing is gap-stressed (§5.4).
- **Event risk in-window:** no entry if a scheduled event falls inside the hold window;
  requires a PIT historical results calendar — availability is itself `[MEASURE]`
  (§4.6b, F9).
- **Market microstructure (F15/F16/F18):** circuit-band/ASM/GSM/T2T names excluded from
  the tradeable universe (a lower-circuit lock makes stops unexecutable); T+1 settlement
  caps capital rotation and BTST carries auction risk; execution at next-open must be
  capped by *pre-open auction* volume, not daily turnover.
- Fewer signals (~2–5/week `[MEASURE]`) → slower evidence; Gate 4 takes months, not
  weeks (§6, F5).

---

## 3. Goal Contract (frozen)

| Quantity | Contract |
|---|---|
| Win rate on emitted signals | ≥ 0.80 point estimate AND day-clustered 95% CI lower bound ≥ the **measured** floor of the frozen geometry |
| Post-cost expectancy | point ≥ +0.10%/trade at the reference size AND day-clustered CI lower bound > 0 `[gate; c(size) from §4.4]` |
| Signals | ≥ 2/week minimum viability; expected 2–5/week `[MEASURE]` |
| Direction | Long-only (F3) |
| What this is NOT | Daily up/down calls on every stock; the system is silent unless calibrated P(win) clears the gate |
| Kill-switch metric | Expectancy |

---

## 4. PHASE 0 — Feasibility Measurement (2–3 days, compute only) — THE DECISION GATE

**Deliverable:** `reports/v6/phase0_report.md`, every table filled, GO/KILL verdict
against §4.7 v2. No system building. User reviews before Phase 1 exists.

### 4.1 Data assembly (½ day)
- Daily panel from the existing bhavcopy store (2023-06→2026-06). **Adjusted paths:**
  barrier resolution runs on corporate-action-adjusted prices; holds containing an
  ex-date are counted and reported (F19).
- Universe cuts (F20 — external review 2026-07-07): the on-disk membership file was
  verified to be a **static starter, not PIT** (all 47 names stamped from 2019-04-01,
  zero removals `[MEASURED 2026-07-07]`) — using it as the GO basis would inject the
  exact survivorship bias it was meant to prevent. Therefore:
  **(a)** user's 16-stock list — report-only, excluded from GO/KILL (F8);
  **(b) THE GO/KILL BASIS — a PIT-by-construction liquidity universe:** top-100 symbols
  by trailing 6-month median daily turnover, **recomputed monthly from the full
  bhavcopy market using strictly-trailing data only**. Survivorship-free by
  construction: a name that later fell was liquid *then* and is included *then*; names
  entering/exiting mid-window do so on information available at the time. Delistings
  mid-hold are counted and reported as tail events;
  **(c)** true index-membership ranges rebuilt from NSE/niftyindices reconstitution
  archives `[VERIFY availability — unknown #14]` as a cross-check cut; the static
  starter file is demoted to a legacy artifact and never used for evaluation.
  Circuit-band/ASM/GSM/T2T names excluded from (b) and (c) `[VERIFY lists]` (F15).
- ATR_daily(14), strictly trailing.

### 4.2 Daily triple-barrier labeler (adaptation)
- Entry = next day's open; target/stop = entry ± a/b × ATR_daily; time barrier = H
  trading days (exit at close of day H); conservative dual-touch ⇒ stop.
- **Tie-break bias quantified:** re-resolve the 46 bar-store names on 1-min paths;
  report the disagreement rate `[MEASURE — qualifies every other result]`.

### 4.3 Geometry grid — long-only primary (48 cells: a ∈ {0.5, 0.75, 1.0, 1.5} × b ∈ {2, 3, 4, 5} × H ∈ {2, 3, 5})
Per cell, mandatory columns:
- measured baseline win rate (pooled) · **per calendar year** · **per sample half** (F6/F7)
- exit mix: target-hit / stop-hit / time-exit shares — **stop-hit share is a GO input** (F13)
- **empirical required δ** from the measured exit-type PnL mix (the closed-form c/W
  understates the requirement when time-exits exist, F11), at 1× and 2× slippage stress,
  at the §4.4 reference size
- **short-side mirror baseline as drift diagnostic** (not tradeable): long/short floor
  asymmetry > 8 points flags drift contamination of the measured floor (F6)
- gap-through-stop share; n; theoretical floor for reference.

### 4.4 Delivery cost model `[VERIFY every constant in code, cite source]` — **broker: Upstox (user decision 2026-07-08)**
- **Ordering rule (F21):** the c(size) curve is computed FIRST — it needs no labeling —
  and the minimum-viable-size floor is frozen **before any geometry cell is evaluated**,
  so a cell can never be selected that only works for institutional capital.
- STT 0.1% buy AND sell · stamp 0.015% buy · exchange txn + SEBI · GST · brokerage
  (₹0–20/order) · DP ≈ ₹15 flat per sell scrip-day · next-open slippage capped by
  **pre-open auction volume** participation (not daily turnover, F18).
- **Open-print slippage measured, not assumed (F23):** the opening print is chaotic;
  the slippage constant is set from the empirical distribution of (next-open fill vs
  first-15-min VWAP) on the 46 names holding 1-min bars `[MEASURE — unknown #15]`;
  Gate 4 later compares realized paper fills against this model.
- **Position size config-driven ₹1k → ₹1Cr** (user requirement): output the full cost
  curve c(size) at ₹1k / ₹10k / ₹1L / ₹10L / ₹1Cr; report the **minimum viable size**
  (flat fees dominate below it — **`[MEASURED 2026-07-08]` ₹31,828**, inside the
  indicative ₹25–50k band; rule: flat fees = ex-slippage variable cost, frozen per F21;
  smaller sizes are plumbing-test-only and excluded from GO/KILL) and the **per-symbol
  maximum** from auction-volume participation `[MEASURE — unknown #13]`. Reference size
  for all gate arithmetic: **₹1L** (revisited at Gate 3 per the curve) —
  **`[MEASURED 2026-07-08]` c(₹1L) = 0.393% base / 0.493% at 2× stress (0.293%
  ex-slippage, in the predicted 0.28–0.35 band)**; slippage constant provisional 5bps/side
  pending the open-print study (unknown #15). Full curve:
  `reports/v6/phase0_cost_curve.md`; code `src/v6/costs.py` (7 hand-computed tests).
- Config keys (Phase 1): `v6.position_size_inr`, `v6.max_auction_participation_pct`.

### 4.5 Gap-risk measurement
Overnight gap distribution beyond k·ATR; expected loss beyond stop per cell;
**1-night vs weekend/multi-night split** (F17). Feeds §5.4 sizing. `[MEASURE]`
- **Circuit-lock tail factor (F22):** measure how often an adverse overnight gap
  coincides with a lower-circuit lock (stop unexecutable — the position cannot exit
  at any price that day), especially in the liquidity-tier mid-caps; model the
  multi-day liquidation penalty and fold it into the gap-stressed sizing factor.

### 4.6 News backtestability probe (GDELT historical) — **sampled design** (F14)
8 symbols × 12 random weeks per quarter, 2023→2026, at 1 request/5s (compliant), run as
a background job; extrapolate coverage. Market-level and symbol-level coverage reported
**separately** (market-level headlines exist ~daily; the 30% symbol-day threshold
applies to symbol-level only). Decides whether news enters the Gate-3 backtest or stays
forward-only. `[MEASURE — live pull verified 2026-07-07; archive depth unverified]`

### 4.6b Earnings-calendar PIT probe (F9) — NEW
Gate 3's event blackout needs historical results dates *as known before the event* for
the full universe. Probe availability/coverage (NSE corporate-announcement archives;
`scripts/fetch_results_dates.py` exists) for 2023→2026. If PIT coverage < 90% of
universe-quarters, the blackout is applied conservatively (exclude the scheduled-month
window) and the impact on signal rate is reported. `[MEASURE]`

### 4.7 GO / KILL — **v2 — FROZEN (approved by user 2026-07-08, pre-measurement, after end-to-end re-verification; see F24). No change is legitimate from this point on.**

**GO to Phase 1 requires ALL of the following, evaluated on the PIT universe cut only, long side:**
1. ≥ 1 geometry cell with pooled measured baseline ∈ **[0.72, 0.79]** (strictly below the
   0.80 target so the model's edge closes the gap — the v1 band's top of 0.85 contradicted
   the target, F12);
2. **robustness of that cell (F6/F7):** each sample half within ±3 points of the pooled
   baseline; worst calendar year ≥ 0.70; the nearest neighbor cells (a±1, b±1 step)
   within ±3 points — a plateau, not a lottery cell. **Grid-edge rule (F24):** for cells
   on the grid boundary, ALL existing neighbors (2–4 of them) must satisfy the ±3-point
   band; a missing neighbor is neither a pass nor a fail;
3. **geometry genuinely exercised (F13):** stop-hit share ≥ 10% AND time-exit share ≤ 40%;
4. **empirical required δ** (from the measured exit mix, F11) ≤ **4.0 points** at 1× costs
   and ≤ **6.0 points** at 2× slippage stress, at the ₹1L reference size;
5. gap risk bounded: expected loss beyond stop ≤ 25% of stop distance (weekend split
   reported);
6. tie-break bias (4.2) ≤ 5 points, else all daily-OHLC results re-based on the
   1-min-verified subset.

**Effective-edge disclosure (F24):** the model's required fired-subset lift is
**max(0.80 − cell baseline, empirical δ)** — at the band bottom (0.72) that is up to
**8 points**, exceeding the 3–6-point range stated in §0/§2 (which is the cost-driven
component only). GO cells near the bottom of the band therefore demand more model edge;
the Phase 0 report must print this effective requirement per qualifying cell so the GO
decision is made with the true number in view.

**KILL:** no cell satisfies 1–4 ⇒ *"selective-signal-by-geometry does not work long-only
on NSE at ≤5-day holds."* **Pre-registered fallback (F24, recorded at approval time so it
cannot be invented after a kill):** the default next step on a Phase 0 or Gate 3 KILL is
**re-contract to asymmetric risk/reward at realistic win rates (~45–60% WR, payoff ≥ 2:1),
evaluated on the same measured data with a freshly pre-registered gate before any build**;
stopping outright remains available at the user's election. **No threshold tuning past
this point.**

> **OUTCOME `[MEASURED 2026-07-08]`: KILL FIRED.** 0 of 48 cells satisfy criteria 1–4
> (max pooled WR 0.711 vs band floor 0.72; min empirical δ 6.3 pts vs ceiling 4.0).
> Full evidence: `reports/v6/phase0_report.md`. Same-day cost: ~3 hours of compute,
> ₹0 of capital.
> **ELECTION `[2026-07-08]`: the user elected STOP.** The F24 fallback was assessed
> against the same measured data before the election: pre-cost expectancy negative in
> all 48 cells; realized win/loss asymmetry at a=1.5,b=2 collapses to ≈0.9:1 (nominal
> 0.75:1 target never realized) under 52–84% time-exits; a ≥2:1 realized payoff needs
> multi-week holds — a fourth horizon change, declined as sunk-cost escalation.
> Project closed with the evidence file complete.

---

## 5. PHASE 1 — Conditional Build (2–3 weeks) — only after the Phase 0 report is approved

### 5.0 Prerequisite (upgraded from optional, F4): history extension
Extend bhavcopy to **≥ 2019** with per-year legacy-format verification + cross-checks
against broker daily candles (20 random symbol-days/year) `[VERIFY]`. Rationale: Gate 3
needs ≥ 300 pooled OOS trades (§5.5); 3 years of data cannot supply that at 2–5
signals/week. If extension fails verification, Gate 3's evaluation window must grow
forward in time instead — stated now, not renegotiated later.

### 5.1 Reuse map
bars/bhavcopy/corporate_actions unchanged · regime channel as-is · news per §4.6 outcome ·
labeler daily mode from Phase 0 · features: daily set (momentum ladder, delivery z, FII z,
OI Δ z, basis, 52w position, vol regime) + the 13 regime/news features, schema v6.0, same
`_present` policy · LightGBM only · blend + ACI gate (α = 0.20; τ₀ = measured floor + 4pts)
· backtester fold machinery + day-clustered CI + registry unchanged · EOD decision runner
replaces the intraday loop · API/Swagger/journal/drift unchanged.

### 5.2 Daily ranker
Post-close ranking on volume/turnover surge z, delivery z, news/regime alignment,
momentum quality; per-fold weight fitting; surveillance-list exclusions live here (F15).

### 5.3 Models
LightGBM on the daily row. No new model families.

### 5.4 Risk layer (overnight-specific)
- **Gap-stressed sizing:** size = (capital × risk%) / (stop distance × (1 + measured
  gap-tail factor)). A stop is never treated as a max loss.
- **Event blackout** per §4.6b outcome — and the post-event entry is a **named
  "post-event drift" sub-strategy (F23)**: entries allowed only *after* a scheduled
  event's gap (never held into one), reported as its own cut in the Gate-3 fairness
  tables so its contribution is visible, never averaged away.
- **T+1 capital rotation** modeled in the backtester's concurrency accounting; BTST
  exits flagged with auction risk noted (F16).
- Concurrency/sector caps, kill switches carried from v4.

### 5.5 Walk-forward & Gate 3 (pre-registered)
- Expanding folds; with §5.0 history: train ≥ 3y, OOS 6m per fold, ≥ 5 folds.
- **Gate 3 is scoreable only at pooled OOS n ≥ 300 matured signals** (F4 power
  arithmetic: distinguishing WR 0.82 from a 0.76 floor day-clustered needs ~300+);
  if n < 300 the OOS window extends — pre-stated, not negotiated.
- **Acceptance (base AND 2× stress):** WR point ≥ 0.80 ∧ day-clustered CI lower ≥
  measured floor ∧ expectancy point ≥ +0.10% at reference size ∧ expectancy CI lower > 0
  ∧ ≥ 2 signals/week ∧ calibration gap ≤ 0.03 ∧ no leakage alarm (> floor + 10pts at
  > 10% coverage) ∧ no subsidy breach.
- **Kill criterion:** floor inside the WR CI ∧ expectancy ≤ 0 ⇒ "no exploitable edge at
  the daily horizon with this information set." Final; §4.7 KILL options apply.

## 6. PHASE 2 — Paper & Capital Gates (honest timelines, F5)
- **Gate 4:** ≥ **60 matured signals** (n=30's ±14pt noise made the old 5pt-gap test a
  coin flip). Criteria: live WR not below backtest WR by more than the one-sided 95%
  binomial bound at realized n (≈ 8.5pts at n=60) ∧ live expectancy > 0 ∧ fills within
  modeled next-open slippage. **Duration: 12–30 weeks at 2–5 signals/week (3–7 months)**
  — stated plainly; the earlier "6–8 weeks" held only at the optimistic rate.
- **Gate 5:** unchanged (compliance checklist, broker orders, smallest viable size,
  kill switches clean).

## 7. Timeline & Decision Tree
```
Phase 0 (2–3 days) ─► user reviews report
  ├─ KILL ─► stop / re-contract — cost: days
  └─ GO ─► Phase 1 (2–3 wks incl. history extension) ─► Gate 3 (n ≥ 300 pooled OOS)
              ├─ KILL ─► stop; data layers remain — cost: ~3 wks
              └─ PASS ─► Phase 2 paper (3–7 months, Gate 4) ─► Gate 5 ─► capital
```

## 8. Register of Every Unknown
| # | Quantity | Tag | Resolved by |
|---|---|---|---|
| 1 | Measured daily baseline per cell (pooled/yearly/halves) | `[MEASURE]` | 4.3 |
| 2 | Exit mix incl. stop-hit share per cell | `[MEASURE]` | 4.3 |
| 3 | Daily ATR% distribution / widths | `[MEASURE]` | 4.1 |
| 4 | Delivery cost constants + DP flat + c(size) curve | `[VERIFY]` | 4.4 |
| 5 | Gap tail (1-night vs multi-night) | `[MEASURE]` | 4.5 |
| 6 | Daily-OHLC tie-break bias vs 1-min truth | `[MEASURE]` | 4.2 |
| 7 | GDELT historical coverage (market vs symbol level) | `[MEASURE]` | 4.6 |
| 8 | PIT earnings-calendar coverage | `[MEASURE]` | 4.6b |
| 9 | Signal rate at fire threshold | `[MEASURE]` | Gate 3 |
| 10 | Model fired-subset edge δ vs required 3–6pts | `[MEASURE]` | Gate 3 |
| 11 | Pre-2023 bhavcopy legacy format | `[VERIFY]` | 5.0 |
| 12 | Surveillance/circuit-band lists (PIT) | `[VERIFY]` | 4.1/5.2 |
| 13 | Pre-open auction volume by symbol | `[MEASURE]` | 4.4 |
| 14 | Index reconstitution archives 2019→2026 (cross-check cut) | `[VERIFY]` | 4.1c |
| 15 | Open-print slippage distribution (open vs first-15min VWAP) | `[MEASURE]` | 4.4 |

## 9. What You Get at Each Checkpoint / Never Claimed
- After Phase 0 (days): the measured answer to "can 80% be engineered long-only at this
  horizon?", with drift, truncation, and selection guards — the question no prior plan
  asked first. After Gate 3 (~3–4 wks): the edge verdict at honest statistical power.
  After Gate 4 (3–7 months): live-conditions evidence.
- Never claimed: a guarantee. The kill criteria are the product — "no" in days, not months.

## 10. Amendment Log — deep review 2026-07-07 (19 findings, all dispositioned)
| F# | Severity | Finding → disposition |
|---|---|---|
| F1 | CRITICAL | §2 required-δ off 10× ("0.3–0.6pts") → corrected to 2–6pts throughout |
| F2 | CRITICAL | v1 GO "δ ≤ 1.5pts" unsatisfiable by any cell → v2 criterion 4: empirical δ ≤ 4.0/6.0pts |
| F3 | CRITICAL | Overnight cash shorts impossible → system long-only; shorts = drift diagnostic only |
| F4 | CRITICAL | Gate 3 underpowered ~3–7× → n ≥ 300 pooled; history extension now a prerequisite (§5.0) |
| F5 | MAJOR | Gate 4 n=30/5pt = coin flip; timeline wrong → n ≥ 60, binomial bound, 3–7 months stated |
| F6 | MAJOR | Long-only baseline contaminated by 2023–26 drift → per-year/half tables, short-mirror diagnostic, worst-year ≥ 0.70 |
| F7 | MAJOR | 48×3 cells, "≥1 in band" = winner's curse → plateau + split-half + PIT-only evaluation |
| F8 | MAJOR | 16-list survivorship-selected → report-only, excluded from GO/KILL |
| F9 | MAJOR | Earnings blackout unbacktestable/unregistered → §4.6b probe + unknown #8 |
| F10 | MAJOR | Required model edge never stated vs measured ≈0 → disclosed in §0/§2; unknown #10 |
| F11 | MAJOR | c/W invalid with time-exits → empirical per-cell required δ |
| F12 | MAJOR | Band top 0.85 > 0.80 target → band [0.72, 0.79] |
| F13 | MAJOR | √H scaling recreates truncation at H=2–3, b=4–5 → stop-hit ≥ 10% GO criterion |
| F14 | MINOR | GDELT probe ~54h at rate limit → sampled design |
| F15 | MINOR | Circuit/ASM/GSM/T2T unmodeled → excluded from universe; unknown #12 |
| F16 | MINOR | T+1/BTST unaddressed → §5.4 + backtester rotation accounting |
| F17 | MINOR | Weekend gaps pooled → split reporting in 4.5 |
| F18 | MINOR | ₹1Cr cap on daily turnover wrong basis → pre-open auction volume basis |
| F19 | MINOR | Corp actions mid-hold → adjusted paths + ex-date counts in 4.2 |
| F20 | MAJOR | *(external review 2026-07-07)* membership file verified static (47 names, from 2019-04-01, zero removals) → GO basis replaced by PIT-by-construction liquidity universe; index-archive rebuild as cross-check (unknown #14) |
| F21 | MINOR | *(external)* c(size) curve computed + viable-size floor frozen BEFORE cell evaluation |
| F22 | MINOR | *(external)* circuit-lock tail factor added to §4.5 gap model |
| F23 | MINOR | *(external)* post-event drift named as sub-strategy with own fairness cut; open-print slippage set empirically (unknown #15) |
| F24 | — | *(approval review 2026-07-08)* §4.7 v2 re-verified end-to-end and **approved by user; thresholds frozen**. Three pre-measurement clarifications recorded: (a) effective required edge = max(0.80 − baseline, empirical δ), up to 8pts at band bottom — must be printed per qualifying cell in the Phase 0 report; (b) grid-edge cells evaluated on all existing neighbors (2–4); (c) KILL fallback pre-registered: default = re-contract to asymmetric R:R (~45–60% WR, ≥2:1 payoff) under a fresh pre-registered gate; stop remains the user's election |

---

*PLAN v6 v2.2 — 2026-07-08. Reviewed adversarially before a single measurement ran; the
frozen gate that would have auto-killed the project by arithmetic was caught on paper,
which is precisely the discipline this plan exists to enforce. §4.7 v2 approved and
frozen 2026-07-08 (F24); the pre-registration window is now closed — Phase 0 may run.*
