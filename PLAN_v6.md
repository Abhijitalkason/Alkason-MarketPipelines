# PLAN v6 — Daily/Swing Selective Signal System (1–5 Day Holds, NSE)

> **Project:** Selective trading-signal system for NSE stocks at the 1–5 day (swing) horizon
> **Goal contract:** ≥ **80% win rate on emitted signals** AND **positive post-cost expectancy per trade** — the win rate engineered by barrier geometry **verified to materialize on real paths before anything is built**, the expectancy delivered by model edge and proven through pre-registered gates.
> **Lineage:** PLAN.md (v1, killed) → PLAN_v3 (intraday, killed) → PLAN_v4 (intraday retarget, kill criterion fired at Gate 2) → **v6 (horizon change — the response PLAN_v4 §6.5 itself prescribed)**
> **Status:** PLAN ONLY — no implementation until Phase 0 is approved, and no Phase 1 until Phase 0's measured evidence is reviewed.
> **Plan version:** 2026-07-07

---

## 0. The No-Excuses Contract (how v6 is different from every prior plan)

This plan is written under four binding rules, adopted because v1/v3/v4 each failed by
violating one of them:

1. **Measure before build.** No module is written on top of an unmeasured assumption.
   Phase 0 (2–3 days, compute only) measures the load-bearing question — *does the
   geometry floor actually materialize at this horizon?* — before any system work.
   (v4's error: 4 weeks of build before a 1-day measurement that killed the premise.)
2. **Every number is tagged.** `[MEASURED]` = from our own data, cited. `[VERIFY]` = a
   public constant to be confirmed in code. `[MEASURE]` = unknown until Phase 0/1 —
   and nothing downstream may treat it as known. There are **no untagged estimates**.
3. **Pre-registered GO/KILL at every checkpoint,** written here, before results are
   seen. The user reviews evidence at each checkpoint and explicitly approves the next
   phase. No tuning past a kill signal (the run registry makes tune-until-green visible).
4. **The metric contract is frozen:** win rate on **emitted** signals + post-cost
   expectancy, with day-clustered confidence intervals. Full-coverage "accuracy"
   remains banned (retired to `scripts/diagnostics/`, PLAN_v4 §9).

**Honest prior, disclosed up front (from the 2026-07-07 discussion):** ~90% confidence
the geometry floor is real at this horizon (mechanical, testable); ~35–50% that a
positive post-cost edge exists; **~25–40% that the full system passes every gate.**
Phase 0 exists to replace these priors with measurements. There is no guarantee, and
this plan never claims one — what it guarantees is that every "no" arrives in days,
with evidence, at compute cost only, before any capital is exposed.

---

## 1. Lessons Register — every prior plan, and what v6 does about it

| Plan | What it promised | What actually happened | v6 design response |
|---|---|---|---|
| **v1** (PLAN.md) | 5-class next-day signals, "maximum achievable accuracy" (73–85%) | 22–40% ≈ majority baseline; dead FinBERT pipeline; 6-model stack; yfinance adjustment errors | Accuracy banned; selective contract; one model family; NSE-official/broker data only; news only with real timestamps |
| **v3** (PLAN_v3) | 89% win rate purchased by geometry at 1–2h; edge δ ≥ 3–4pts | Contract never evaluated (no backtest artifacts); user measured banned metric → 0.50 panic | Executable gates with evidence files; the contract metric is computed by the pipeline itself, not a side script |
| **v4** (PLAN_v4) | 80% floor via wide geometry + regime/news channel | **Built first, measured after.** Measured truth `[MEASURED 2026-07-03]`: real baseline 0.51–0.575 across all 25 geometry cells (theory said 0.67–0.91) because the 2h time barrier truncates paths (75/98 trades were time exits); models ≈ 0 edge (logloss 0.69); opened-gate run: WR 51%, expectancy −0.27%/trade | **Horizon change** — the exact §6.5 response. And the build order inverted: Phase 0 measures the new horizon's floor before one line of system code |
| v4 carry-forward assets | — | Production-grade & reusable: PIT bar/bhavcopy stores, regime channel (785 days backfilled), news capture (RSS+GDELT live-verified), labeler/backtester/gates/risk/serving, 27-test regime suite | §5 reuse map — ~70% of v6 is already built |

**The one structural fact that motivates v6** `[MEASURED]`: the geometry theorem
`P(win)=b/(a+b)` failed intraday *only* because stops were unreachable within 2h —
the measured baseline was flat in `b`. At 1–5 day holds, a 2–5×ATR stop is genuinely
reachable, so the mechanism that buys the 80% has room to function. **That this
actually happens on NSE daily paths is Phase 0's question — not an assumption.**

---

## 2. Why 1–5 Day Holds (evidence, and the new risks it introduces)

**For it:**
- Time-barrier truncation — the measured killer of v3/v4 — weakens: barriers are
  reachable before the horizon expires. `[MEASURE: time-exit share per cell, Phase 0.3]`
- Cost-to-width ratio collapses: daily ATR ≈ 1.5–2.5% `[MEASURE]` → total width
  W = (a+b)·ATR ≈ 6–12%, against delivery round-trip costs ≈ 0.30–0.35% `[VERIFY §4.4]`
  → required edge δ = c/W ≈ **0.3–0.6 points**, versus 15–31 points measured intraday.
  This is the single largest structural improvement.
- The information we already ingest lives at this horizon: post-event/news drift plays
  out over days (the academically documented edge), delivery %, FII/DII flows, OI
  positioning are all daily series. The v4 regime channel plugs in unchanged.
- Competition at this horizon is not decided by co-location speed.
- Data breadth: bhavcopy on disk covers the **entire NSE market** (~2,400 symbols ×
  763 sessions, 2023-06-01 → 2026-06-19, with delivery %) `[MEASURED on disk]` — Phase 0
  is not limited to the 46 bar-store names.

**Against it (owned, not hidden):**
- **Overnight gap risk:** a stop is no longer a maximum loss — price can gap through
  it. Sizing must use gap-stressed loss, not stop distance (§5.4). Gap-tail
  distribution is measured in Phase 0.5, not assumed.
- **Event risk inside the hold window:** earnings/RBI/budget can land mid-trade.
  Policy (§5.4): trade *post*-event drift; never hold a position into a scheduled
  event (blackout uses the existing events calendar).
- Fewer signals: selectivity at this horizon means **~2–5 signals/week** `[MEASURE]`,
  not per day. Slower feedback; Gate-4 paper takes ~6–8 weeks to mature ≥30 signals.
- Only 3 years of on-disk history constrains walk-forward folds; §5.6 extends history
  with verification steps (no source is trusted uncross-checked).

---

## 3. Goal Contract (frozen)

| Quantity | Contract | Tag |
|---|---|---|
| Win rate on emitted signals | ≥ 0.80, with the **day-clustered 95% CI lower bound above the measured geometric floor** | gate |
| Post-cost expectancy per trade | ≥ +0.10% (floor raised vs v4's +0.05% because delivery costs and gap risk are larger) | gate `[VERIFY after 4.4]` |
| Signals | ~2–5/week portfolio-wide; ≥ 2/week minimum for the system to be worth operating | `[MEASURE]` |
| What this is NOT | Daily up/down predictions for every stock on a list. The system is silent unless its calibrated P(win) clears the conformal gate. | frozen |
| Kill-switch metric | Expectancy (win rate stays high right until a high-floor system dies) | frozen |

---

## 4. PHASE 0 — Feasibility Measurement (2–3 days, compute only) — THE DECISION GATE

**Deliverable:** `reports/v6/phase0_report.md` — every table below filled with measured
numbers, ending in a GO / KILL verdict against the pre-registered criteria in §4.7.
**No system building.** The only code written is a daily-labeling adaptation script and
the report generator. User reviews the report before Phase 1 exists.

### 4.1 Data assembly (½ day)
- Daily panel from the **existing** bhavcopy store (no new downloads needed for the
  core question): OHLC, volume, turnover, delivery % per symbol-day, 2023-06→2026-06.
- Universe for measurement: the existing PIT membership file (51 names) **plus** a
  liquidity tier from the full bhavcopy market (top-N by median turnover, membership
  checked per-day against the PIT file where covered). *Survivorship caveat recorded:*
  conclusions are strongest on the PIT-covered names; the wide tier is a robustness
  check, not primary evidence.
- ATR_daily(14) from daily OHLC, strictly trailing.

### 4.2 Daily triple-barrier labeler (adaptation, not new architecture)
- Entry = next day's **open** (realistic; decisions made post-close or pre-open).
- Target = entry + a·ATR_daily; Stop = entry − b·ATR_daily (long side; short side mirrored).
- Time barrier = H trading days, exit at close of day H.
- Path resolution on **daily OHLC** with the conservative dual-touch tie-break
  (both barriers inside one day's range ⇒ stop) — same pessimistic convention as v4.
- **Tie-break bias quantified, not assumed:** on the 46 symbols with 1-min bars,
  re-resolve the same trades on true intraday paths and report how often the
  conservative daily tie-break was wrong. `[MEASURE — this number qualifies every
  other Phase 0 result]`

### 4.3 Geometry grid on real daily paths (the core measurement)
- Grid: a ∈ {0.5, 0.75, 1.0, 1.5} × b ∈ {2.0, 3.0, 4.0, 5.0} × H ∈ {2, 3, 5} days
  (48 cells). Per cell, report: **measured baseline win rate** (the number that came
  out 0.51–0.575 intraday), time-exit share, gap-through-stop share, n, theoretical
  floor, required δ = c/W at the §4.4 cost model, 1× and 2× cost stress.
- Long side default; direction-hinted variant (prior-day momentum sign) as a
  robustness column.

### 4.4 Delivery cost model `[VERIFY every constant in code, cite source]`
- STT 0.1% buy AND sell (delivery) · stamp 0.015% buy · exchange txn + SEBI · GST ·
  brokerage (₹0–20/order) · **DP charge ~₹15 flat per sell scrip-day** (matters at
  small size — report cost as f(position size)) · slippage at next-open execution
  (measure open-auction participation feasibility on bhavcopy volumes).
- Output: round-trip c% at ₹1L / ₹5L / ₹20L position sizes.

### 4.5 Gap-risk measurement
- Distribution of overnight gaps beyond k·ATR for the universe; **expected loss beyond
  stop** per geometry cell; worst-decile scenario. Feeds §5.4 sizing. `[MEASURE]`

### 4.6 News backtestability probe (GDELT historical)
- The v4 finding "news can't be backfilled" was an **intraday** constraint (pre-09:15
  precision). At daily horizon, GDELT's timestamped archive may be usable. Live pull
  verified 2026-07-07 (66 pre-open headlines); **historical depth/coverage unverified**
  (API rate-limited during probing). Measure: % of universe symbol-days with ≥1 matched
  headline, per quarter 2023→2026, at a compliant 1-request/5s rate. `[MEASURE]`
- Outcome decides whether news sentiment enters the Phase-1 decision backtest
  (coverage ≥ 30% of symbol-days) or stays forward-only as in v4.

### 4.7 Pre-registered GO / KILL (written before any measurement runs)
**GO to Phase 1 requires ALL of:**
1. ≥ 1 geometry cell with measured baseline ∈ **[0.75, 0.85]** (floor below target so
   the model adds the edge; above it and the v3 trap returns);
2. that cell's required δ ≤ **1.5 points** at 1× costs and ≤ 3 points at 2× stress;
3. time-exit share ≤ **40%** in that cell (the intraday failure signature was 77%);
4. gap risk bounded: expected loss beyond stop ≤ **25%** of the stop distance;
5. tie-break bias (4.2) ≤ **5 points** of win rate, else daily-OHLC results are
   re-based on the 1-min-verified subset.

**KILL:** no cell satisfies 1–3 ⇒ the finding is *"selective-signal-by-geometry does
not work on NSE at ≤5-day holds either."* Response: stop this design; the remaining
honest options are (a) re-contract to asymmetric risk/reward at ~55–65% win rate, or
(b) stop the project. **No threshold tuning past this point.**

---

## 5. PHASE 1 — Conditional Build (2–3 weeks) — only after user approves the Phase 0 report

### 5.1 Reuse map (what is NOT rebuilt — ~70% carries over)
| Module | Disposition |
|---|---|
| bars.py / bhavcopy.py / corporate_actions.py | unchanged (bhavcopy becomes the primary price source at daily) |
| regime_data.py + 785-day backfill + capture cron | **as-is** — same pre-open regime features, now matched to their natural horizon |
| news_regime.py (RSS + GDELT + FinBERT) | as-is for forward capture; + GDELT historical backfill iff 4.6 passes |
| labeler.py | + daily mode from Phase 0 (entry/barriers/H parameterized; same tie-break discipline) |
| features.py | daily feature set: momentum ladder (1/3/5/10/20d), delivery z, FII 5d z, OI Δ z, basis, 52w position, vol regime + the 13 regime/news features unchanged; schema v6.0, same `_present` missingness policy |
| price_model / blend / conformal ACI gate | unchanged (α = 0.20, τ₀ set from Phase 0 floor + 4pts) |
| backtester.py | fold machinery, leakage alarms, day-clustered CI, fairness tables, run registry — unchanged; label horizon parameterized |
| risk.py | + gap-aware sizing (5.4); blackout calendar reused |
| paper_runner.py | **simplified** to an EOD decision runner (decide post-close, order next open) — no intraday loop |
| API / Swagger / journal / drift / DVC | unchanged |

### 5.2 Screener → daily ranker
Rank the PIT universe post-close on: volume/turnover surge z, delivery z, news/regime
alignment, momentum quality. Same per-fold weight-fitting discipline as v3/v4 (§7).

### 5.3 Models
LightGBM on the daily feature row (regime + news included per 4.6). Flow channel
staging rule carried from v4. No new model families — the v1 lesson stands.

### 5.4 Risk layer additions (the overnight-specific controls)
- **Sizing on gap-stressed loss:** size = (capital × risk%) / (stop distance ×
  (1 + measured gap-tail factor from 4.5)) — a stop is not a max loss and is never
  treated as one.
- **Event blackout:** no entry if a scheduled event (earnings, RBI, budget, Fed) falls
  inside the hold window; existing events.csv + hold-horizon lookahead. Post-event
  entries are allowed — that *is* the drift trade.
- Concurrency/sector caps and kill switches carried from v4 unchanged.

### 5.5 Walk-forward protocol & Gate 3 (pre-registered)
- With on-disk history: train 18m → OOS 3m, expanding, ≥ 4 folds. If 5.6 extends
  history to ≥ 2019: ≥ 6 folds (preferred).
- **Acceptance (base AND 2× cost stress):** fired WR ≥ 0.80 ∧ day-clustered CI lower
  bound > measured floor ∧ expectancy ≥ +0.10% ∧ ≥ 2 signals/week ∧ calibration gap
  ≤ 0.03 ∧ no leakage alarm (> floor+10pts at >10% coverage) ∧ no subsidy breach.
- **Kill criterion:** floor inside the CI ∧ expectancy ≤ 0 ⇒ "no exploitable edge at
  the daily horizon with this information set" ⇒ see §4.7 KILL options. Final.

### 5.6 History extension (parallel task, verification-first)
- NSE bhavcopy archives predate 2023; the pre-UDiFF legacy format differs. Extend
  `bhavcopy.py` to ≥ 2019 **only with** per-year format verification + cross-checks
  against a second source (broker daily candles) on 20 random symbol-days/year.
  `[VERIFY — no source trusted uncross-checked; the v1 yfinance lesson]`

## 6. PHASE 2 — Paper & Capital Gates
- **Gate 4:** ≥ 30 matured signals (≈ 6–8 weeks at the measured signal rate) via the
  EOD paper runner; WR gap vs backtest ≤ 5pts (wider than v4's 3 — small-n honesty);
  positive expectancy; fills within the modeled next-open slippage.
- **Gate 5:** unchanged from v3/v4 (compliance checklist, broker orders, smallest
  size, kill switches clean). Not scheduled until Gate 4 evidence exists.

## 7. Timeline & Decision Tree

```
Phase 0  (2–3 days, compute)  ──► user reviews phase0_report.md
   ├─ KILL: no qualifying geometry ──► stop / re-contract (§4.7) — cost: days
   └─ GO ──► Phase 1 build (2–3 wks) ──► Gate 3 backtest (base+stress)
                 ├─ KILL: no edge over floor ──► stop; data layers remain — cost: ~3 wks
                 └─ PASS ──► Phase 2 paper (6–8 wks, Gate 4) ──► Gate 5 ──► capital
```
Checkpoints are user-reviewed; nothing auto-proceeds.

## 8. Register of Every Unknown (nothing here is assumed)

| # | Quantity | Tag | Resolved by |
|---|---|---|---|
| 1 | Measured daily-horizon baseline per geometry cell | `[MEASURE]` | 4.3 |
| 2 | Time-exit share at H ∈ {2,3,5} | `[MEASURE]` | 4.3 |
| 3 | Daily ATR% distribution / width W | `[MEASURE]` | 4.1 |
| 4 | Delivery cost stack constants + DP flat fee | `[VERIFY]` | 4.4 |
| 5 | Overnight gap-tail beyond stop | `[MEASURE]` | 4.5 |
| 6 | Daily-OHLC tie-break bias vs 1-min truth | `[MEASURE]` | 4.2 |
| 7 | GDELT historical coverage of the universe | `[MEASURE]` | 4.6 |
| 8 | Signal rate at the fire threshold | `[MEASURE]` | Gate 3 |
| 9 | Model edge δ over the floor | `[MEASURE]` | Gate 3 |
| 10 | Legacy bhavcopy format pre-2023 | `[VERIFY]` | 5.6 |

## 9. What You Get at Each Checkpoint / What Is Never Claimed

- **After Phase 0 (days):** a measured answer to "can 80% be engineered at this
  horizon on NSE?" — the question v3/v4 never asked first. Either a GO with numbers,
  or a KILL that cost almost nothing.
- **After Gate 3 (~3 weeks):** the edge verdict on the system's real contract, with
  day-clustered statistics, leakage discipline, and an append-only run registry.
- **After Gate 4 (~2 months):** live-conditions evidence.
- **Never claimed:** a guarantee. The pre-registered kill criteria are the product —
  they convert "months of hope" into "days of evidence," which is the difference
  between this plan and the three before it.

---

*PLAN v6 — 2026-07-07. The horizon where the geometry can work, the information we
already collect has its documented edge, and — for the first time in this project —
the measurement comes before the build.*
