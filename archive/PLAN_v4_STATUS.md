> ⚠️ **EXPIRED — this plan is no longer active and has been superseded. Please check the current plan: [Aug26Plan.md](../Aug26Plan.md).**

# PLAN v4 — Implementation Status & Runbook

> Companion to PLAN_v4.md. Records what is implemented, how it was verified, and
> the exact commands to run the decision phase on the real data machine.
> Last updated: 2026-07-03.

---

## 1. Implementation status (code complete)

| Milestone | Status | Key files |
|---|---|---|
| M0 — Gate 1 fix | ✅ done | `src/intraday/gates.py` (`gate1` uses `gates.min_data_names_frac` × PIT-universe size) |
| M1 — Geometry retarget (80% floor) | ✅ done | `config/config_v3.yaml` (geometry/gate/gates), `src/intraday/labeler.py` (geo-override fix), `src/intraday/geometry_study.py` (grid + config thresholds) |
| M2 — Global/macro regime layer | ✅ done + **executed** | `src/intraday/regime_data.py`, `scripts/backfill_regime.py`, `main.py` (`regime-capture`, `regime-backfill`) |
| M3 — Features v4 + backtester | ✅ done | `src/intraday/features.py` (schema **v4.0**), `src/intraday/backtester.py` (regime_bucket, day-clustered CI, Run-D `symbols`) |
| M4 — Regime veto | ✅ done | `src/intraday/regime.py`, backtester + `paper_runner.py` fire-time plug-ins |
| M5 — News sentiment channel | ✅ done | `src/intraday/news_regime.py` (FinBERT, lazy), `main.py` (`news-capture`) |
| M6 — Retire accuracy metric | ✅ done | `scripts/diagnostics/prediction_accuracy.py` (banner: not a decision metric) |
| Swagger | ✅ done | `src/api/app.py` (v4, `/docs` + `/redoc`, `APIKeyHeader` Authorize) |

### Config changes (the 80% contract)
- `geometry`: `target_atr 0.75`, `stop_atr 3.00`, `time_barrier_hours 2` → geometric floor `3.0/3.75 = 0.80`, wide width to lower required edge `δ = c/W`.
- `gate`: `fire_threshold_init 0.84`, `target_win_rate 0.80` (α = 0.20).
- `gates`: `min_win_rate 0.80`, `leakage_alarm_win_rate 0.90`, `calibration_max_gap 0.03`, `min_data_names_frac 0.90`.
- new `geometry_study` (grid band), `regime` (veto), `data.regime_path` sections.

### Verification (tests)
- Full `pytest` suite green (see §4 for the new tests).
- New: `tests/test_regime.py` (PIT lookahead exclusion, veto logic, news degradation), `tests/test_geometry_v4.py` (labeler override honored, 80% floor).

---

## 2. Executed on this machine

- **M2 global/macro backfill**: `python scripts/backfill_regime.py --years 3`
  → **5,979 rows across 785 day-files** (SPX, NDX, Nikkei, HangSeng, USDINR, crude, NSEI, India VIX), 2023-07-04 → 2026-07-03, in `data/regime/global/`.
- **M2 live snapshot**: `python main.py --mode regime-capture` → 8 rows written for today. Verified.
- **Regime channel is LIVE in features**: a 1-month `build_dataset` slice shows global
  regime features at `present=1.0` (spx/ndx/asia/usdinr/crude/india_vix). So a normal
  backtest now genuinely exercises **Run B**.

### 2a. Scoped real backtests (16 symbols, 2025-06→2026-06, 6-mo train, 2 folds)
These are compute-feasible *demonstrations*, NOT the Gate-3 acceptance run (which is
46 symbols × 36 months × 6 folds — run on the data machine per §3).

- **At the real fire threshold τ₀=0.84**: **0 trades emitted** — models trained to
  `binary_logloss ≈ 0.69` (≈ random, no edge), so nothing cleared the gate. The system
  correctly stays silent rather than firing edgeless trades ("Gate 3 fails honestly").
- **Mechanism check (gate forced open, τ₀=0.50, floor=0.50)** — exercises the v4 report
  on real fired trades:
  - n_trades 98 · **win_rate 0.510** · expectancy_net **−0.267%/trade**
  - day-clustered 95% CI **[0.398, 0.626]** — entirely **below** the geometric floor
  - all gates evaluate; `all_pass=false`, `ci_above_floor_ok=false`
  - report JSON + trades parquet + `run_registry.csv` all written correctly.

### 2b. ⚠️ Important finding — the geometric floor is NOT free on real paths
The theoretical floor `b/(a+b)=0.80` at 0.75/3.0 assumes driftless, symmetric paths.
The **realized** triple-barrier win rate on real 1-min paths over this window is **~0.51**,
far below 0.80 — because the 2-hour time barrier truncates many trades (time exits split
~50/50 by PnL sign) and paths carry drift. **Consequence:** the 0.75/3.0 geometry in
config is a placeholder that the **geometry study (Gate 2) must confirm or replace** —
it will search for a geometry whose *measured* baseline lands in the [0.76, 0.84] band,
exactly as PLAN_v4 §4 says ("the grid on real 1-min paths decides, not the table").
This is the #1 next action on the data machine.

### 2c. 🔴 Gate 2 result — geometry sweep on real 1-min paths (8 symbols, ~4 months)
Measured REAL triple-barrier baseline win rate for all 25 grid cells (long side,
all decision bars — same basis as `geometry_study._relabel`):

| a (target) | REAL baseline | theoretical `b/(a+b)` range | required δ |
|---|---|---|---|
| 0.40 | **0.575** | 0.833–0.909 | 17–31 pts |
| 0.50 | 0.545 | 0.800–0.889 | 17–30 pts |
| 0.60 | 0.528 | 0.769–0.870 | 16–29 pts |
| 0.75 | 0.517 | 0.727–0.842 | 16–27 pts |
| 1.00 | 0.512 | 0.667–0.800 | 15–25 pts |

**QUALIFYING cells in [0.76, 0.84]: NONE.** Two structural facts:

1. **The real baseline depends almost only on `a`, not on `b`** — it is flat as the
   stop widens (0.4/2.0 and 0.4/4.0 both give 0.575). The geometric theorem
   `P(win)=b/(a+b)` assumes the stop is actually *hit*; at a 2-hour horizon a
   3–4×ATR stop is almost never reached, so widening it buys **zero** extra wins.
   Most trades resolve as target-hit or as a ~50/50 time-barrier exit, dragging the
   win rate toward 0.5, not 0.8.
2. **The best achievable real baseline is ~0.575** (tightest target), and even that
   needs a **17–31 point** model edge to beat costs — versus the ~0 edge every model
   showed (logloss ≈ 0.69).

**Conclusion (PLAN_v4 §4 / §6.5 kill criterion, at the geometry level):** the 80%+
win-rate-by-geometry contract is **not achievable at the 1–2 hour horizon** on these
NSE large caps. The premise that both PLAN_v3 (89%) and PLAN_v4 (80%) rest on —
"purchase the win rate with barrier geometry" — breaks because the intraday time
barrier truncates paths before wide stops are hit. This is the honest, evidence-
backed finding, not a tuning problem. **Do not loosen the gates.**

*Caveat:* scoped run (8 symbols, 4 months, long-only). Confirm with the full
`--mode geometry-study` on the data machine — but the flat-across-`b` pattern and
the ~0.51–0.58 ceiling are structural (a property of the 2h barrier), so the
conclusion is expected to hold.

### 2d. Recommended pivot (per PLAN_v4 §6.5)
The response to "edge below costs at this horizon" is a horizon change, not gate
loosening: move the signal to a **daily/swing (1–5 day hold)** system, where (a) the
geometry theorem works because multi-day stops actually get hit, and (b) news /
macro sentiment has *proven* post-event drift edge (its natural home — exactly why
PLAN_v4 keeps the regime/news data layers, which carry forward unchanged). The
intraday code, data moat, and PIT discipline are all reusable; only the horizon
(labeler barriers + hold window) changes.

---

## 3. Decision phase — commands to run on the data machine

Global/macro regime data is now on disk, so a standard backtest already exercises
the regime channel (this is **Run B**). Steps:

```bash
# (once) confirm Gate 1 now passes on the ~50-name universe
python main.py --mode gates

# M1 — confirm/adjust the frozen 0.75/3.0 geometry on real history (train folds only)
python main.py --mode geometry-study

# M3 decision backtests (fired-signal win rate + expectancy + day-clustered CI):
#   Run B — full universe, global regime live (news still present=0, forward-only)
python main.py --mode backtest
python main.py --mode backtest --stress          # 2× slippage stress (Gate 3 needs both)

#   Run D — restricted to the pre-decided 16-symbol list (coverage/WR cost check)
python main.py --mode backtest --symbols RELIANCE HDFCBANK INFY TCS WIPRO BPCL \
  ICICIBANK SBIN BHARTIARTL ITC LT KOTAKBANK AXISBANK BAJFINANCE MARUTI TITAN

#   Run A — pure ablation (regime present=0): temporarily point data.regime_path at
#   an empty dir (or `mv data/regime data/regime.bak`) and re-run backtest.
```

Read `reports/v3/backtest_<ts>.json` and `reports/v3/run_registry.csv`. The
report now carries `winrate_ci_dayclustered` and per-`regime_bucket` fairness.

### Acceptance (Gate 3, v4) — Run B, base AND 2× stress
- fired win rate ≥ 0.80 · expectancy ≥ +0.05% net · ≥1 signal/day
- calibration gap ≤ 0.03 · no fold >0.90 WR at >0.10 coverage · no subsidy
- fired-WR 95% **day-clustered** CI lower bound above the geometric floor (`gates.ci_above_floor_ok`)

### Kill criterion (pre-registered — no tuning past this)
If Run B's win rate is statistically indistinguishable from the 0.80 floor (floor
inside the day-clustered CI) AND expectancy ≤ 0 → "no exploitable edge at this
horizon with this information set." Keep the data layers; move the news/regime
channel to a daily/swing system (new plan). Do not loosen the gates.

---

## 4. Forward-only channels (start now, mature over weeks)

```bash
# global/macro morning snapshot + news — schedule 08:45–09:10 IST daily (cron)
python main.py --mode regime-capture
python main.py --mode news-capture     # needs: pip install feedparser transformers torch
```
News features are historically `present=0` by design; they earn a value only on
days the capture ran, and are judged in paper trading (news-present vs -absent),
never in the decision backtest (PLAN_v4 §8 honesty clause).

---

## 5. Serving / Swagger

```bash
export V3_API_KEY=<key>
python main.py --mode serve      # then open http://localhost:8000/docs
```
Click **Authorize** in Swagger UI and paste the key; it is injected as `X-API-Key`
on every "Try it out" call. ReDoc at `/redoc`, raw schema at `/openapi.json`.

---

## 5a. Audit-gap closure (2026-07-07)

The six gaps found by the post-implementation audit are closed:

1. **GDELT DOC 2.0 secondary news source** — `news_regime._gdelt_headlines()`
   pulls the [prior close → 09:15 IST] window (live-verified: 66 pre-open
   headlines); merged into `capture()` with `source="gdelt"`, fail-open.
2. **Per-day regime cache** — `features._market_regime_for_day()` caches the
   market-level regime features + verdict per (regime-dir, day); symbols on the
   same day share one read (bounded, test-root-safe).
3. **Vetoed-candidate counterfactual scaffold** — the paper runner now evaluates
   the veto AFTER features/probabilities; a vetoed signal's journal line carries
   direction, p_price/p_flow/p_cal, τ, and the full feature dict, so it can be
   labeled and scored once the day's bars mature.
4. **README/RUNBOOK** — rewritten for v4 (contract, gates table, new CLI modes,
   regime/news channels, retired-diagnostic note, measured-baseline caveat).
5. **Cron** — `scripts/preopen_capture.sh` (runs both captures, fail-open) +
   exact crontab entries documented in RUNBOOK ("Pre-open regime & news capture").
6. **Explicit regime lookahead test** —
   `test_regime.py::test_regime_features_future_data_immunity` (future content
   dates + post-09:30 captures leave the row bit-identical) and
   `test_regime_day_cache_consistency`.

## 6. Deliberate deviations from PLAN_v4

- **Report/state paths kept at `reports/v3` / `models/v3`** (plan suggested a v4
  bump). The v4 schema change invalidates the old bundle anyway (retrain
  overwrites it), and keeping the paths avoids destabilizing the test fixtures —
  functionally equivalent. Change the `paths.*` block if hard separation is wanted.
- **GIFT-Nifty pre-open** is wired (feature + `_present` flag) but the fetch is a
  no-op placeholder — no reliable free source. It stays `present=0` until a source
  is added; the missingness policy handles this by design.
