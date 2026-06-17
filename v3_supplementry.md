# PLAN v3 — Supplementary Implementation Specification (Production)

> **Project:** Intraday (2–3 hour window) selective trading-signal system for NSE large caps
> **Parent document:** PLAN_v3.md (2026-06-11). This document is the **complete, file-by-file production implementation specification** for v3. It incorporates every finding of the 2026-06-12 audit (IDs B-x/H-x/M-x/L-x) and specifies the system to the level of module contracts, data schemas, API contracts, error behavior, tests, and deployment — executing this spec yields the production system, not scaffolding.
> **Scope rule:** when this document and PLAN_v3.md conflict, PLAN_v3's *targets and gates* win; this document's *implementation detail* wins.
> **Spec version:** 2026-06-12

---

## Table of Contents

0. [Non-Negotiables](#0-non-negotiables)
1. [Final Repository Layout](#1-final-repository-layout)
2. [Configuration — final `config_v3.yaml`](#2-configuration)
3. [Data Layer Specification](#3-data-layer)
4. [Feature & Label Pipeline Specification](#4-feature--label-pipeline)
5. [Screener Specification](#5-screener)
6. [Models, Blend, Conformal Gate](#6-models-blend-gate)
7. [Cost Model](#7-cost-model)
8. [Backtester & Experimentation Protocol](#8-backtester)
9. [Training Pipeline (`trainer.py`) & Model Registry](#9-training-pipeline)
10. [Live Runtime: Recorder & Paper Runner](#10-live-runtime)
11. [Risk Layer & Kill Switches](#11-risk-layer)
12. [Governance: Journal, MLflow, DVC, Drift](#12-governance)
13. [Serving API (`src/api/app.py`)](#13-serving-api)
14. [Explainability](#14-explainability)
15. [CLI (`main.py`) — final mode set](#15-cli)
16. [v1 Decommission](#16-v1-decommission)
17. [Deployment: Docker, env, secrets](#17-deployment)
18. [Test Plan (full suite)](#18-test-plan)
19. [Documentation Deliverables](#19-documentation)
20. [Acceptance Gates incl. Gate 0](#20-acceptance-gates)
21. [Execution Order & Milestones](#21-execution-order)
22. [Definition of Production-Ready (checklist)](#22-definition-of-production-ready)

---

## 0. Non-Negotiables

These carry the force of PLAN_v3's honesty clause and bind every PR:

1. **No training run until B-4 (FII lookahead) and B-5 (flow-channel imputation) are fixed.** Metrics from a contaminated pipeline are void.
2. **A control without a call site does not exist.** Every control ships with (a) its invocation in the live path and (b) a test that fails if the invocation is removed. (v1 and the current v3 skeleton both died of dead controls: `check_kill_switches`, `cross_check_bhavcopy`, `PriceModel.tune`, the entire validation module.)
3. **No silent degradation.** Exceptions either raise or are counted against an explicit, configured tolerance that raises when exceeded. `fillna(0)`, `except: pass`, "log and continue" without a counter are all banned patterns, enforced in review.
4. **One config (`config_v3.yaml`), one clock (IST), one feature pipeline (train == backtest == paper == serve, same module, same functions).**
5. **Documentation trails reality.** No commit message, README line, or STATUS entry may claim a phase done before its gate test passes in CI.
6. **Append-only evidence.** Run records, journals, gate-state history, and gate evaluations are written once and never overwritten.

---

## 1. Final Repository Layout

```
AI-MLOps-Solution/
├── config/
│   └── config_v3.yaml                # THE config (Section 2). config.yaml deleted with v1.
├── data/
│   ├── bars/<SYMBOL>/<YYYY-MM>.parquet
│   ├── bhavcopy/eq_<date>.parquet, fo_<date>.parquet, fii_dii.parquet
│   ├── preopen/<date>.parquet
│   ├── depth/<date>/<HHMM>.parquet
│   └── reference/
│       ├── universe_membership.csv   # point-in-time membership (Section 3.4)
│       ├── corporate_actions.csv     # ex-date factors (Section 3.5)
│       ├── events.csv                # blackout calendar (populated, Section 11)
│       ├── nse_holidays.csv          # exchange holidays (Section 3.3)
│       └── upstox_instruments.json   # cached instrument map
├── src/intraday/
│   ├── __init__.py                   # config, IST clock, PIT universe loader
│   ├── data_feed.py                  # Upstox client, backfill, reconcile      [MODIFY]
│   ├── bhavcopy.py                   # NSE EOD files, raise-on-miss            [MODIFY]
│   ├── corporate_actions.py          # factor table build + apply              [NEW]
│   ├── bars.py                       # store, resample, ATR, validation        [MODIFY]
│   ├── screener.py                   # 09:30 screen + per-fold weight fit      [MODIFY]
│   ├── labeler.py                    # triple-barrier                          [KEEP]
│   ├── features.py                   # feature matrix, schema v3.1             [MODIFY]
│   ├── price_model.py                # LightGBM (+ monotonicity)               [MODIFY]
│   ├── flow_model.py                 # CatBoost                                [KEEP]
│   ├── blend.py                      # blend + isotonic                        [KEEP]
│   ├── conformal_gate.py             # ACI + state history                     [MODIFY]
│   ├── costs.py                      # cost stack + impact term                [MODIFY]
│   ├── backtester.py                 # purged walk-forward + metric contract   [MODIFY]
│   ├── trainer.py                    # production training pipeline            [NEW]
│   ├── geometry_study.py             # Phase-3 grid protocol (Gate 2)          [NEW]
│   ├── risk.py                       # limits + kill switches (wired)          [MODIFY]
│   ├── recorder.py                   # live session recorder + live bars       [MODIFY]
│   ├── paper_runner.py               # Gate-4 loop, restart-safe               [MODIFY]
│   ├── drift.py                      # Evidently → ACI recalib → halt          [NEW]
│   ├── journal.py                    # hash-chained append-only decision log   [NEW]
│   ├── tracking.py                   # MLflow wrapper + run registry           [NEW]
│   ├── dvc_utils.py                  # DVC ops, failures raise                 [NEW]
│   └── gates.py                      # executable Gates 0–5                    [NEW]
├── src/slm/explainer.py              # Granite narration of fired signals      [MODIFY]
├── src/api/app.py                    # v3 serving API (Section 13)             [REWRITE]
├── main.py                           # v3-only CLI (Section 15)                [REWRITE]
├── tests/                            # Section 18                              [NEW]
├── docker/                           # Section 17                              [MODIFY]
├── RUNBOOK.md, COMPLIANCE.md, README.md (rewritten)
├── PLAN_v3.md, v3_supplementry.md
└── PLAN.md, STATUS_2026-06-03.md     # historical record only
```

**Deleted with v1 (Section 16):** `src/models/` (all six files), `src/slm/news_scraper.py`, `src/slm/sentiment.py`, `src/training/`, `src/evaluation/`, `src/data/` (after import sweep), `config/config.yaml`, v1 modes in `main.py`, v1 model artifacts (after DVC freeze tag `v1-archive`).

---

## 2. Configuration

Final `config_v3.yaml` — existing keys unchanged; **additions** marked. Every key below must be consumed by code; a CI test (`test_config.py`) asserts no dead keys and no missing keys.

```yaml
universe:
  membership_file: data/reference/universe_membership.csv     # ADDED (replaces file:)
  index_symbol: "NIFTY 50"
  min_median_1min_volume: 5000          # NOW ENFORCED in screener (H-1)
  max_median_spread_bps: 5              # NOW ENFORCED in screener (H-1)

data:
  provider: upstox
  bars_path: data/bars
  bhavcopy_path: data/bhavcopy
  preopen_path: data/preopen
  depth_path: data/depth
  reference_path: data/reference
  backfill_years: 3
  record_depth_levels: 5
  bar_freq_decision: 15min
  bar_freq_feature: 5min
  session_open: "09:15"
  session_close: "15:30"
  timezone: Asia/Kolkata                # ADDED — all session logic pinned (M-7)
  live_poll_seconds: 10                 # ADDED — screened-name poll cadence (H-3)
  max_flow_missing_share: 0.40          # ADDED — loud threshold (B-5)
  max_skip_share: 0.05                  # ADDED — dataset-thinning tolerance (M-1)

screen:
  time: "09:30:05"
  top_n: 12
  gap_band: [0.003, 0.025]
  lookback_days: 20
  weights: { ... unchanged ... }        # live DEFAULTS; per-fold fitted weights override (Section 5.3)

geometry: { ... unchanged ... }

gate:
  fire_threshold_init: 0.92
  aci_gamma: 0.01
  target_win_rate: 0.89
  model_agreement_floor: 0.75
  calibration_window_days: 60           # NOW CONSUMED (Section 9.4)
  flow_model_active: false              # ADDED — staged activation (B-5); flips to true
                                        # only when flow-feature real-data share ≥ 0.60

training:
  train_min_months: 18
  test_fold_months: 3
  embargo_days: 1
  optuna_trials: 30                     # NOW CONSUMED (Section 6.1)
  blend_weight_init: 0.6
  random_state: 42
  inner_calib_days: 60                  # ADDED — day-boundary inner split (F.1/M-3)

costs:
  brokerage_per_order_inr: 20
  stt_sell_pct: 0.00025
  exchange_txn_pct: 0.0000297
  sebi_fee_pct: 0.000001
  stamp_buy_pct: 0.00003
  gst_pct: 0.18
  slippage_half_spread_pct: 0.0002      # ADDED — replaces flat slippage_pct
  impact_coeff: 0.10                    # ADDED — impact = coeff·(qty/median_1min_vol) (H-7)
  slippage_stress_multiplier: 2.0

risk: { ... unchanged ... }

drift:                                  # ADDED (B-7 / D.5)
  reference_artifact: feature_matrix    # MLflow artifact name of production model's train matrix
  current_window_sessions: 10
  share_drifted_columns_alert: 0.30
  consecutive_breaches_to_halt: 2

gates: { ... unchanged ... }

mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: intraday-selective-v3
  registry_model_name: intraday-v3-signal   # ADDED

api:                                    # ADDED (Section 13)
  host: 0.0.0.0
  port: 8000
  api_key_env: V3_API_KEY               # name of env var holding the key

paths:
  reports: reports/v3
  signals_log: reports/v3/signals
  journal: reports/v3/journal           # ADDED
  alerts: reports/v3/alerts             # ADDED
  gates: reports/v3/gates               # ADDED
  run_registry: reports/v3/run_registry.csv   # ADDED
  state: models/v3
```

---

## 3. Data Layer

### 3.1 `src/intraday/__init__.py` — core utilities

```python
ROOT, CONFIG_PATH                      # unchanged
load_config() -> dict                  # lru_cache(1); load_config.cache_clear() exposed for tests
now_ist() -> datetime                  # zoneinfo("Asia/Kolkata"); ALL session logic uses this (M-7)
today_ist() -> date
load_universe(as_of: date) -> pd.DataFrame   # point-in-time (Section 3.4); `as_of` REQUIRED —
                                             # no default arg, so survivorship can't sneak back in
```

Startup assertion (recorder/paper/serve): if `datetime.now()` differs from `now_ist()` by > 60 s, log the offset at ERROR once.

### 3.2 Bar store — `bars.py`

**File schema** `data/bars/<SYMBOL>/<YYYY-MM>.parquet`:

| column | type | notes |
|---|---|---|
| ts | datetime64[ns] (tz-naive, IST) | bar start, 1-min |
| open/high/low/close | float64 | **raw, as-traded** — storage is never adjusted (H-2) |
| volume | int64 | |
| capture_ts | datetime64[ns] | when this row was captured |
| provisional | bool | ADDED — true for poll-built bars; false for official candles (H-3) |

**Public contract:**

```python
load_1min(symbol, start, end, adjusted=True) -> pd.DataFrame
    # session-filtered, ts-indexed, validated (raises BarValidationError)
    # adjusted=True: applies corporate-action factors ON READ as explicit columns:
    #   adds adj_factor; open/high/low/close become adjusted; close_raw retained
    #   (PLAN_v3 §6.2 "non-destructive by construction"; storage untouched)
    # EXCLUDES provisional bars unless include_provisional=True (training never sees them)
validate_1min(symbol, df)              # unchanged checks; stays called inside load_1min
cross_check_bhavcopy(symbol, day, df_1min)   # unchanged check; CALLERS in 3.6 (fixes B-9)
resample(df, freq)                     # unchanged (left-label/left-closed)
atr_2h(df_5min) -> pd.Series           # CHANGED: true range masked across session boundaries —
                                       # first bar of each day uses high-low only (M-4)
session_dates(symbol, start, end)
```

### 3.3 NSE EOD files — `bhavcopy.py`

Changes (all else unchanged):

- `fetch_fii_dii()` output gains a parsed **`date` column** (from NSE payload's date field, `dd-Mon-yyyy`), and named flow columns (`fii_net_cr`, `dii_net_cr`). Existing file is migrated once by a `scripts/migrate_fii.py` one-shot. **(B-4 prerequisite)**
- `backfill_bhavcopies(start, end)` **raises `BhavcopyError`** at the end if any business day not present in `data/reference/nse_holidays.csv` is missing *both* eq and fo files (M-1). Holiday file: `date,description`, sourced from NSE trading-holiday list, one row per holiday, maintained yearly.
- `fetch_preopen()` unchanged (correctly capture-stamped). It remains true that pre-open history only accumulates forward — the *consequence* is handled by the missingness policy (Section 4.3), not hidden.

### 3.4 Point-in-time universe — `universe_membership.csv` (H-1)

```
symbol,name,sector,from_date,to_date        # to_date empty = current member
RELIANCE,Reliance Industries,Energy,2019-04-01,
...
```

Built from monthly NIFTY-200 constituent archives (niftyindices.com); a helper `scripts/build_membership.py` converts the monthly files into ranges. `load_universe(as_of)` filters `from_date ≤ as_of < to_date`. The backtester, screener, and trainer pass the replay/training day; the live path passes `today_ist()`.

### 3.5 Corporate actions — `corporate_actions.py` (NEW, H-2)

```python
build_table(start, end) -> pd.DataFrame      # parse NSE CA file → symbol, ex_date, factor
                                             # (split/bonus ratios → price factor)
load_factors(symbol) -> pd.DataFrame         # cached per-symbol factor series
apply(df_1min, symbol) -> pd.DataFrame       # cumulative adj_factor column; price cols × factor;
                                             # close_raw preserved; volume ÷ factor
```

Called only from `load_1min(adjusted=True)`. Tested with a synthetic 1:2 split fixture asserting `gap_pct`, ATR, and barrier placement continuity across the ex-date.

### 3.6 Broker feed — `data_feed.py`

Changes:

- `backfill(...)`: after each symbol's save, run `cross_check_bhavcopy` on **5 random session dates** of the downloaded span; any failure adds the symbol to `failures` (which already raises at the end). **First caller of the dead control (B-9).**
- `reconcile_day(symbol, day)` (NEW): fetch the official 1-min candles for `day` via `historical_1min`, overwrite that day's poll-built provisional rows in the monthly store (`provisional=False`), then `cross_check_bhavcopy`. Called by the recorder post-close for every recorded symbol (H-3). Logs per-day mean |provisional close − official close| to the journal; > 5 bps emits an alert file (this is the live skew detector PLAN_v3 §14 wanted).
- `save_monthly` keeps `keep="first"` **only for `provisional=False` rows**; official rows always replace provisional rows for the same ts.

---

## 4. Feature & Label Pipeline

### 4.1 Schema v3.1 — `features.py`

```python
PRICE_FEATURES   # unchanged 21 columns
FLOW_FEATURES  = ["oi_change_z", "basis_pct", "delivery_z", "preopen_imbalance", "fii_5d_z"]
FLOW_PRESENT   = [f + "_present" for f in FLOW_FEATURES]      # ADDED (B-5)
FEATURE_ORDER  = PRICE_FEATURES + FLOW_FEATURES + FLOW_PRESENT + ["direction"]
SCHEMA_VERSION = "v3.1"
check_schema(df)            # unchanged contract; NOW CALLED at every serve-time predict
                            # (paper runner + API) — fixes H-10
```

### 4.2 Lookahead fixes (B-4)

`_flow_row(symbol, day)`:

- FII/DII: `df = df[df["date"] < pd.Timestamp(day)]` **before** any aggregation; z-score from named column `fii_net_cr` over the trailing 20 rows. The fragile `num.columns[-1]` selection is deleted.
- All other flow inputs already use strictly-prior files (verified); each gains its `_present` flag.
- The FII parse `except Exception → warn` is replaced by: parse failure **raises** unless the file is absent (absence sets `fii_5d_z=NaN`, `fii_5d_z_present=0`).

### 4.3 Missingness policy (B-5) — `build_matrix`

```
1. flow NaN → 0.0 ONLY alongside its _present flag = 0      (information preserved)
2. flow_real_share = mean over rows of mean(FLOW_PRESENT)   (logged on every build)
3. if flow_real_share < 1 - data.max_flow_missing_share → RAISE FlowDataError
   (training on majority-imputed flow is the v1 FinBERT failure; it must be impossible)
4. price-feature NaN rows: dropped as today, but counted; skip_share > data.max_skip_share → RAISE
```

Per-row feature exceptions in `build_matrix` and per-day screen exceptions in `build_dataset`: kept as skips but **counted**; over-threshold raises (M-1).

### 4.4 Lookahead property test (the load-bearing test)

For a fixture symbol-day and every decision bar T: compute `features_for_day` on data truncated at T+15min (decision-bar close) and on the full day; assert the row for T is **identical**. Runs over all features including flow. This test would have caught B-4 and is the regression guard for every future feature. (`tests/test_features.py::test_no_lookahead_property`)

### 4.5 Labeler — unchanged

`labeler.py` passed the audit (entry next-bar-open, conservative dual-touch tie-break, squareoff cap, time-barrier sign). It gains only tests (Section 18).

---

## 5. Screener

### 5.1 Liquidity floor (H-1) — now enforced

In `screen_day`, before scoring: drop symbols whose trailing 20-session median 1-min volume < `universe.min_median_1min_volume`; where recorded depth exists, drop symbols whose median spread > `max_median_spread_bps` (no depth → spread leg skipped and counted in the day's journal line).

### 5.2 Point-in-time inputs

- `symbols = load_universe(as_of=day)` (H-1).
- Backtest replay filters every capture-stamped input (`bhavcopy`, `preopen`, `fii_dii`) on `capture_ts ≤ decision time` where the file's capture_ts predates the replay decision; for backfilled files (capture after the fact) the **content date** rule (strictly-prior files) applies — both rules live in one helper `point_in_time(df, day, decision_ts)` used by screener and features (H-8).

### 5.3 Per-fold weight fitting (H-8 / PLAN_v3 §7)

```python
fit_weights(screen_rows: pd.DataFrame, day_won: pd.Series) -> dict
    # logistic regression of "symbol produced ≥1 triple-barrier win that day"
    # on the six score components; returns weight dict, L2-regularized, signs reported
```

- **Backtester:** `build_dataset` retains the screen component columns per (day, symbol). Per fold, `fit_weights` runs on the **train days only**; the fitted weights re-rank each test day's candidate set and take `top_n` (a subset of the default-weight screen's candidates — the approximation is logged per fold as `screen_refit_overlap`; if overlap < 0.7 the fold report flags that the dataset should be rebuilt with fold weights).
- **Trainer:** fits weights on the production train window, saves to `models/v3/screen_weights.json`; the live `screen_day` loads them when present, else config defaults.

---

## 6. Models, Blend, Gate

### 6.1 Price model — `price_model.py`

- `tune()` gets its caller (M-5): `trainer.py` and `backtester.fit_fold` call it **once per fold/run** on the inner train split, `training.optuna_trials` trials, params frozen before any OOF prediction; study summary logged to MLflow. (Config flag `training.tune: true|false` to keep CI fast.)
- Monotonicity constraints (PLAN_v3 §10.1): `monotone_constraints` for economically-signed features — `atr_pct` unconstrained, `cumvol_vs_norm` ≥ 0 monotone… maintained as an explicit dict in the module with a comment per constraint; absent justification = no constraint.

### 6.2 Flow model — `flow_model.py`

Unchanged, **but**: instantiated/used only when `gate.flow_model_active: true`. Activation criterion (checked by `trainer.py`, logged): `flow_real_share ≥ 0.60` over the training window. Until then the system is price-model-only (Section 6.3). This is the production answer to B-5: the flow channel turns on when its data is real, not before.

### 6.3 Blend — `blend.py`

- When flow inactive: `weight = 1.0` pinned, calibrator fit on price-model OOF alone; `fire()` agreement floor applies to `p_price` only.
- Calibration-consistency rule (H-4): the calibrator persisted with a model bundle must have been fit on **out-of-fold predictions of training runs with identical hyperparameters**, produced by a chronological day-boundary inner split with 1-day embargo (`training.inner_calib_days` tail) — never on predictions of models that saw the calibration rows (M-3 inner-split purge included).

### 6.4 Conformal gate — `conformal_gate.py`

- Decision/update logic unchanged (audit-verified correct).
- `save()` additionally **appends** `{ts, tau, alpha, days_updated}` to `models/v3/gate_state_history.jsonl` (G.1); the JSON state file remains the resume point.
- New `coverage_report(trades) -> dict`: per-symbol realized win rate + exact binomial 95% CI for symbols with ≥ 30 fired signals — consumed by backtester (Section 8.3) and `/performance` (Section 13).

---

## 7. Cost Model

`costs.py` — statutory stack unchanged; slippage model replaced (H-7):

```python
round_trip_costs(price, qty, median_1min_volume, stress=False,
                 entry_is_quoted=False) -> TradeCosts
    half_spread = costs.slippage_half_spread_pct
    impact      = costs.impact_coeff * (qty / median_1min_volume)
    per_side    = half_spread + impact          # × stress multiplier under stress
    # entry_is_quoted=True (paper/live entries filled at quoted bid/ask): the entry side
    # charges impact only — the spread is already paid in the fill price (M-1 double-count fix)
round_trip_cost_pct(...)                        # same, as fraction of notional
```

`median_1min_volume` comes from the bar store (trailing 20 sessions), computed in `build_dataset`/paper runner and passed in — `costs.py` stays I/O-free (testable). Backtest report includes the cost-sensitivity table: expectancy at 1× / 1.5× / 2× slippage.

---

## 8. Backtester

### 8.1 Fold machinery (unchanged where audit-verified)

Expanding-window day-level folds, ≥ `train_min_months`, 3-month test, 1-day embargo — kept. **Inner split changed** (M-3/H-4): blend/calibration tail = last `inner_calib_days` **calendar days** of the train window with 1-day embargo, never a row-index cut.

`fit_fold(train_df)` final form:

```
1. head = train days except calib tail (+embargo);  tail = calib days
2. (optional per config) PriceModel.tune(head)          # params frozen now
3. price/flow fit on head → predict tail (true OOF)
4. Blender.fit_weight + fit_calibration on tail OOF
5. price/flow refit on FULL train window with the frozen params
6. return models + blender; per-fold record: params, weight, calib Brier
```

Step 5/4 pairing is legitimate OOF-stacking practice; the *forbidden* pattern (calibrator fit on predictions of models that trained on those rows) is structurally excluded and unit-tested.

### 8.2 Metric contract (PLAN_v3 §12.3 — fully implemented)

Logged to MLflow per fold and aggregate (Section 12.2), and written to `reports/v3/backtest_<ts>.json`:

win rate · post-cost expectancy/trade · coverage (corrected denominator = **all candidate rows in all test windows**, M-4) · signals/day · realized δ vs geometric baseline · max drawdown · worst day · profit factor · daily-P&L Sharpe · per-fold stability table · **Brier score** · artifacts: win-rate-vs-coverage frontier curve (PNG), calibration reliability diagram (PNG), trades parquet. "Accuracy" appears nowhere.

**Calibration gate addition:** |expected − realized win rate| in the fired region ≤ 2 pts on the reliability diagram, else `gates.calibration_ok = false`.

### 8.3 Distributional fairness (H-5) — §12.5 of the metric contract

`_finalize` emits grouped tables (JSON + MLflow artifacts): per **symbol**, **sector**, **volatility regime** (20-day realized-vol tercile), **time-of-day bucket** (09:30–10:00/10:00–10:30/10:30–11:00): n, win rate, expectancy, mean p_cal.

Executable rules:

- **Subsidy alarm:** any symbol/sector with ≥ 30 trades and negative expectancy while the aggregate passes ⇒ `gates.no_subsidy_ok = false` ⇒ Gate 3 fails.
- **Per-symbol coverage:** any symbol (≥ 30 fired) whose exact binomial 95% CI upper bound < `target_win_rate` ⇒ symbol written to `models/v3/excluded_symbols.json` (consumed by screener live) and flagged in the report — excluded explicitly, never averaged away.

### 8.4 Experiment discipline (M-5)

- Every `run_backtest` invocation appends one line to `paths.run_registry` (CSV, append-only): timestamp, git SHA, config hash, args, win rate, expectancy, coverage, leakage-alarm status. Tune-until-green becomes visible by construction.
- Leakage alarm unchanged (already implemented); now also evaluated **per fold**, not only aggregate.
- Gate-3 runner (`gates.py`) executes **base + 2× stress back-to-back**; both must pass (H-7).
- Blackout check passes `symbol` (M-6): `is_blackout(day, sym)` per screened name in `build_dataset`.

### 8.5 Geometry study — `geometry_study.py` (NEW; PLAN_v3 §8, Gate 2)

```python
run_grid(train_days) -> pd.DataFrame
    # a ∈ {0.25,…,0.6} × b ∈ {1.5,…,4.0}; per cell: relabel train-fold candidates,
    # baseline win rate, required δ = c/W at realistic c (Section 7), projected coverage
    # OUTPUT: reports/v3/geometry_grid_<ts>.json + chosen geometry frozen into config PR
    # Runs on TRAINING FOLDS ONLY; the test windows are never touched (enforced by arg type:
    # the function accepts a TrainWindow object only obtainable from make_folds()[k][0])
```

Gate-2 evidence = the grid artifact + the config commit freezing the chosen geometry.

---

## 9. Training Pipeline

### 9.1 `trainer.py` — the production train path (B-1)

```python
train_production(end_day: date, train_months: int | None = None) -> TrainResult
```

```
 1. days   = business days [end_day − train_months, end_day]
 2. data   = build_dataset(days, PIT universe)        # all integrity checks live here
 3. flow_real_share check → decide flow_model_active (Section 6.2), record decision
 4. screen weights: fit_weights(train window) → models/v3/screen_weights.json
 5. fit (Section 8.1 logic, single "fold" = full window with inner calib tail)
 6. SAVE bundle:
      models/v3/price_model.joblib
      models/v3/flow_model.joblib        (when active)
      models/v3/blender.joblib
      models/v3/gate_state.json          (fresh τ₀ unless one exists — never clobber live state)
      models/v3/feature_matrix.parquet   (drift reference, Section 12.4)
      models/v3/model_card.md            (Section 9.3)
 7. MLflow: one run — params (geometry, gate, training, git SHA, config hash, data span,
      row count, flow_real_share, DVC rev), metrics (calib-tail Brier, OOF win-rate@τ₀),
      artifacts (bundle + feature matrix + model card)
 8. mlflow registry: register version of `intraday-v3-signal`, stage = Staging
 9. dvc_utils.add_and_push(["models/v3", "data/bars", "data/bhavcopy"])   # raises on failure
10. journal entry: training event with run_id + version
```

CLI: `python main.py --mode train-v3 [--end YYYY-MM-DD] [--months N]`.

### 9.2 Promotion workflow (approval gate)

- Paper runner loads **Staging**; serving API loads **Production**.
- `python main.py --mode promote --version N` transitions Staging→Production **only if**: Gate-3 report attached to the version, paper evidence (≥ `gates.paper_days` sessions, gap ≤ 3 pts) attached, and the operator passes `--confirm`. Demotion (`--mode rollback`) re-stages the previous Production version; both transitions journaled. Retraining can never silently go live.

### 9.3 Model card (per registered version, MLflow artifact + `models/v3/model_card.md`)

Intended use (2–3h NSE large-cap signals, entries 09:30–11:00 IST) · training data span + row count + flow_real_share · geometry + gate params · OOF metrics + Brier · per-symbol exclusions · known limitations (flow channel status, pre-open archive depth, anything open from this spec) · owner · version + git SHA + DVC rev.

### 9.4 Rolling recalibration job (H-4; consumes `calibration_window_days`)

`python main.py --mode recalibrate` (weekly post-market, cron/manual): refit isotonic on the trailing 60 days of **matured live/paper signals** (features+outcomes from the journal); log before/after Brier to MLflow; save updated `blender.joblib`; journal the event. Refit aborts (no-op, alert) if < 200 matured signals in window.

---

## 10. Live Runtime

### 10.1 Recorder — `recorder.py`

- **All clocks `now_ist()`** (M-7).
- **08:55 auth ping:** cheap authenticated request; failure raises *before* market open with the token-refresh runbook reference (G.5).
- Poll cadence: full universe at 1-min; **screened names at `live_poll_seconds` (10 s)** after 09:30 (H-3 intraday mitigation).
- **`bars_today(symbol) -> pd.DataFrame`** (NEW — the B-2 fix): builds 1-min OHLCV from the in-memory tick buffer on demand, same schema as `load_1min` output, marked provisional.
- **Incremental crash-safety flush:** completed minutes appended to `data/bars/.session/<date>/<SYMBOL>.parquet` every 15 min; on restart the buffer rehydrates from there (A.3). The monthly store is touched **once**, at session close — and a partial day is *never* merged (H-9): `flush_bars` refuses (`raises`) before `session_close` unless `force=True` (used only by tests).
- **Post-close reconcile:** for every recorded symbol, `data_feed.reconcile_day` replaces provisional bars with official candles + bhavcopy cross-check (Section 3.6). Poll-built bars therefore never enter training (`load_1min` default excludes provisional).
- Poll exceptions: still logged-and-continue (correct for a live loop) **but counted**; > 10% failed polls in a session ⇒ flush marks the session degraded and raises after persisting what it has.

### 10.2 Paper runner — `paper_runner.py` (Gate 4)

State machine per session:

```
START   : is_halted() check → load Staging bundle from registry (fallback: models/v3 files)
          → resume session_state_<date>.json if present (restart safety)
08:55   : recorder auth ping
09:00   : capture pre-open
09:30:05: screened = screen_day(today, live_bars=recorder)      # B-2: live frames injected
          journal: screen output line
09:30–11:00 (every 15-min close):
          for each screened symbol without a position:
            risk.can_open → features_for_day(sym, today, df_1min=history+bars_today(sym))
            → check_schema(row)                                  # H-10: serve-time schema guard
            → p_price, p_flow, p_cal → gate.fire
            → journal: gate-evaluation line (fired or not, with p's and τ)
            if fired: quote → fill at bid/ask → position_size → SHAP top-10 recorded (Section 14)
            → journal: fill line; persist session state
continuous (10 s): manage_positions — target/stop/time/squareoff exits, journaled
14:45   : hard square-off
close   : flush bars → reconcile_day per symbol → POST-MARKET:
            gate.update(matured labels) + save (state + history)
            check_kill_switches(all paper trades, backtest win rate from registry)   # B-7: first caller
            drift.run_check()                                    # Section 12.4
            signals log: reports/v3/signals/paper_<date>.jsonl   (append-only)
            quoted-vs-assumed fill comparison logged per trade (PLAN_v3 §14 skew detector)
```

Restart mid-session: positions, closed trades, risk DayState, and tick buffer all rehydrate; a `pytest` simulated-clock test kills and resumes a session and asserts identical end-of-day state.

---

## 11. Risk Layer

- `RiskManager` unchanged (audit-verified) **plus**: `DayState` persists inside the session state file, so a restart cannot reset a daily-loss halt.
- `check_kill_switches` (B-7/M-7) rewritten plainly and unit-tested:
  - (a) mean net expectancy over trades of the last 20 sessions < 0 ⇒ `rolling_20d_expectancy_negative`
  - (b) last 30 trades' win rate < backtest win rate − 0.10 ⇒ `live_win_rate_10pts_below_backtest`
  - Callers: paper-runner post-market (10.2) and `gates.py`. Breach ⇒ `_persist_halt` (existing) ⇒ next session refuses to start.
- `events.csv` **populated** before Gate 3: results dates for the full universe over the backtest span (NSE corporate announcements), RBI/Fed/budget days. `build_dataset` passes `symbol` to `is_blackout` (M-6).
- Stops live note: software-only in paper; the Gate-5 (capital) prerequisite is broker-side SL-M order placement via the broker API — specified in COMPLIANCE.md, implemented only after Gate 4 passes.

---

## 12. Governance

### 12.1 Journal — `journal.py` (NEW, H-11)

```python
journal_write(event_type: str, payload: dict) -> None
    # appends one JSON line to reports/v3/journal/<date>.jsonl
    # line = {ts: now_ist(), type, payload, prev_sha: sha256(previous line)}
    # file opened append-only; a daily-close record seals the chain with the line count
journal_verify(date) -> bool             # recomputes the chain; used by gates.py
```

Event types: `screen`, `gate_eval`, `fill`, `exit`, `aci_update`, `train`, `promote`, `rollback`, `halt`, `drift`, `reconcile`. The journal is the SEBI-audit substrate and the train/serve-skew evidence base.

### 12.2 MLflow — `tracking.py` (NEW, B-6)

Thin wrapper: `start_run(kind)`, `log_backtest(report)`, `log_training(result)`. Hard rules: tracking failures **raise** in `train-v3` and gate runs (no silent local fallback); every run logs git SHA (dirty tree ⇒ raise in gate runs), config hash, data span, DVC rev, seeds. Registry helpers: `register(bundle) -> version`, `transition(version, stage)`, `load_stage(stage) -> bundle`.

### 12.3 DVC — `dvc_utils.py` (NEW, B-6)

`add_and_push(paths)` via subprocess; non-zero exit **raises** `DVCError`. One-time migration (committed): remote moved from `/tmp/dvc-cache` to durable storage (`DVC_REMOTE_URL`, external disk or S3), v1 pointers re-synced once and tagged `v1-archive`, then `data/bars`, `data/bhavcopy`, `data/preopen`, `data/depth`, `models/v3` tracked.

### 12.4 Drift — `drift.py` (NEW, B-7 / PLAN_v3 §13 switch c)

```
reference = production model's feature_matrix.parquet (training artifact)
current   = last drift.current_window_sessions sessions of live feature rows (journal/signals)
report    = Evidently DataDriftPreset → reports/v3/drift/<date>.html + .json
breach    = share of drifted columns ≥ drift.share_drifted_columns_alert
on breach : (1) force ACI recalibration flag for next session (gate re-init from trailing labels)
            (2) append alert to reports/v3/alerts/<date>.json
            (3) consecutive breaches ≥ drift.consecutive_breaches_to_halt → _persist_halt(["drift"])
```

Runs post-market from the paper runner and standalone via `--mode drift`. `/drift` endpoint reads its outputs.

---

## 13. Serving API

`src/api/app.py` — full rewrite. FastAPI, Pydantic v2, **every endpoint requires `X-API-Key`** matching `os.environ[cfg.api.api_key_env]` (no key configured ⇒ app refuses to start; B-8). Models loaded once at startup from registry **Production** stage (fallback `models/v3/` with WARN); inference via `run_in_executor`; `check_schema` before every predict.

| Endpoint | Returns |
|---|---|
| `GET /health` | feed-token validity, model version + age, gate τ + age, halt status, journal chain OK |
| `GET /signals/today` | today's journal `fill`/`exit` events |
| `GET /signals/history?days=N` | signals joined with outcomes; rolling win rate, expectancy, drawdown |
| `GET /screen/today` | persisted 09:30 screen output + scores + excluded symbols |
| `GET /performance` | rolling metrics vs gate thresholds + per-symbol coverage report (6.4) |
| `GET /drift` | last drift report summary + last ACI recalibration ts |
| `POST /backtest` | starts background job, returns job-id; `GET /backtest/{job_id}` for status/report |

Response models defined per endpoint (Pydantic); 503 with reason when halted; no model-mutating endpoint exists (training/promotion are CLI + registry only — the unauthenticated `/retrain` pattern does not return).

---

## 14. Explainability

- **SHAP at decision time** (H-12): on every fired signal, `shap.TreeExplainer(price_model)` top-10 feature attributions (and CatBoost importances when flow active) recorded in the `fill` journal line. Explainer object built once at startup.
- **Granite narration** (`src/slm/explainer.py` rewritten): `explain_signal(journal_fill_line) -> str` renders the recorded SHAP + setup into a one-paragraph explanation via Ollama; called **asynchronously post-fire** (background task), never on the decision path; result appended as a journal `explain` event and surfaced in `/signals/today`.

---

## 15. CLI

`main.py` rewritten — v3 modes only, reads `config_v3.yaml` only:

```
--mode backfill | bhavcopy | record | screen | geometry-study | backtest [--stress]
       | train-v3 | promote --version N --confirm | rollback | recalibrate
       | paper --capital N | drift | gates | serve
```

Every v1 mode (`ingest/validate/features/train/compare/monitor/predict/full-pipeline`) is removed with v1.

---

## 16. v1 Decommission

Order matters (recoverability first):

```
1. dvc push + git tag v1-archive                  # freeze artifacts
2. delete: src/models/ (6 files), src/slm/news_scraper.py, src/slm/sentiment.py,
           src/training/, src/evaluation/, config/config.yaml
3. import sweep: grep for the deleted modules across src/ + main.py + docker → zero hits (CI test)
4. src/data/: delete after the sweep confirms no v3 imports (none exist today)
5. models/manual/ removed from the working tree (recoverable via v1-archive tag)
6. README rewritten (Section 19); STATUS_2026-06-03.md retained as history
```

---

## 17. Deployment

**`docker/Dockerfile.api`:** FinBERT pre-download **deleted**; `requirements-api.txt` = v3 inference set (fastapi, uvicorn, lightgbm, catboost, scikit-learn, pandas, pyarrow, mlflow-skinny, shap, evidently); `USER appuser` (non-root); healthcheck hits `/health` **with** the API key.

**compose:** API service env = `MLFLOW_TRACKING_URI`, `OLLAMA_BASE_URL`, `V3_API_KEY` **only** — the broker token never enters the serving container (it belongs to the recorder/paper process on the host). MLflow + Ollama services unchanged. mounts: `models/`, `reports/`, `data/reference/` read-only for API.

**Secrets:** `.env` stays gitignored; `RUNBOOK.md` documents the daily Upstox token refresh; recorder auth-ping (10.1) catches a stale token at 08:55, not at the first poll.

**`requirements.txt`:** mapie is **not** added — the gate is a custom ACI implementation (simpler than mapie's interfaces for this update rule); this is a recorded deviation from PLAN_v3 §11/§17, justified here, so the dependency list stays honest.

---

## 18. Test Plan

`tests/` — pytest, deterministic synthetic fixtures (no network, no live data), CI on every push. The suite is itself a deliverable: **Gate 0 cannot pass without it green.**

| File | Asserts |
|---|---|
| `conftest.py` | synthetic 1-min bar generator (configurable trend/vol/gaps/splits), bhavcopy/preopen/FII fixture writers, frozen IST clock |
| `test_config.py` | every config key consumed; no missing keys (loads all modules, introspects `load_config()` access via a recording wrapper) |
| `test_bars.py` | validation raises on each defect class; resample label/closed semantics; **ATR strictly trailing** (property: ATR at t unchanged by mutating data ≥ t); ATR session-boundary mask; provisional exclusion |
| `test_corporate_actions.py` | 1:2 split fixture: gap/ATR/barrier continuity across ex-date; storage untouched |
| `test_labeler.py` | dual-touch ⇒ stop; time-barrier sign labeling; squareoff cap; entry = next bar open; no entry when next bar missing |
| `test_features.py` | **`test_no_lookahead_property`** (Section 4.4); FII future-row immunity (B-4 regression); missingness flags + FlowDataError threshold; schema v3.1 check |
| `test_screener.py` | point-in-time inputs (mutating data > 09:30 doesn't change output); liquidity floor; PIT universe respected; `fit_weights` recovers planted weights on synthetic data |
| `test_costs.py` | statutory math vs hand-computed example; impact term scaling; quoted-entry single-spread; 2× stress |
| `test_blend.py` | weight grid optimum; isotonic monotone; unfitted raises; calibration-consistency rule (calibrator never fit on in-fold predictions — structural test on `fit_fold`) |
| `test_conformal_gate.py` | ACI direction (err>α ⇒ τ↑); clip bounds; persistence round trip; history append-only; coverage_report CI math |
| `test_backtester.py` | fold purge/embargo at day boundaries (incl. inner split); **planted-leak test** (inject future-knowledge feature ⇒ LeakageAlarm fires); corrected coverage denominator; subsidy alarm; per-symbol exclusion; gates dict on synthetic trades |
| `test_risk.py` | sizing; sector/concurrency caps; daily-loss halt persists across restart; both kill switches on synthetic logs; blackout w/ symbol |
| `test_trainer.py` | `train_production` round trip on synthetic data: artifacts exist, registry version created (local mlflow), bundle reloads and predicts; flow-activation decision |
| `test_journal.py` | chain verification; tamper detection; write-once semantics |
| `test_drift.py` | breach detection on shifted synthetic features; 2-breach halt; recalib flag |
| `test_paper_session.py` | **simulated-clock full session** on recorded tick fixture: screen runs at 09:30 from live bars (B-2 regression), schema checked at serve, fires + exits + square-off; **kill-and-resume** mid-session ⇒ identical EOD state; no partial-day monthly flush |
| `test_api.py` | every endpoint: 401 without key; happy path against fixture artifacts; 503 when halted |
| `test_dead_imports.py` | deleted v1 modules unreferenced (Section 16 step 3) |

Coverage target: ≥ 85% on `src/intraday/`, enforced in CI.

---

## 19. Documentation Deliverables

- **README.md** — rewritten for v3 only: stack, the 89% contract in one paragraph, CLI reference (Section 15 modes), API table, gate status badge sourced from `reports/v3/gates/`.
- **RUNBOOK.md** — daily ops: token refresh, session start/stop, restart-resume procedure, halt review + clearance, promotion/rollback procedure (with the rehearsal log), drift-alert response, holiday-calendar maintenance.
- **COMPLIANCE.md** — SEBI algo posture (H-11): broker algo approval requirement, exchange order-tagging spec, audit-log retention mapping to the journal, leverage cap statement, the explicit rule **no live order before broker approval exists**; Gate-5 checklist.
- **Model card** — per version (Section 9.3).

---

## 20. Acceptance Gates

PLAN_v3 §19 gates unchanged; **Gate 0 added** and prerequisites bound to this spec. All gates are functions in `gates.py` returning `GateResult(passed, evidence_paths)`, results written append-only to `reports/v3/gates/`.

| Gate | Test (executable) | Pass criterion |
|---|---|---|
| **0 — Engineering readiness** (NEW) | full pytest suite + dead-control sweep + `train-v3` round trip + simulated-clock paper session + journal chain verify | all green in CI; zero grep hits for dead controls; coverage ≥ 85% |
| **1 — Data** | backfill stats + 5 live recorded sessions + reconcile + cross-check | ≥3y bars for the PIT universe; bar-vs-bhavcopy mismatch < 0.1%; recorder uptime 5/5; reconcile skew < 5 bps |
| **2 — Label/geometry** | `geometry_study.run_grid` on training folds | a geometry with baseline ≥ 86% and required δ ≤ 4 pts at projected coverage; frozen by config commit |
| **3 — Backtest** | `gates.py` runs base + 2× stress | win rate ≥ 89% ∧ expectancy ≥ +0.05% ∧ ≥1 signal/day ∧ no leakage alarm (any fold) ∧ stress survives ∧ **no subsidy alarm ∧ calibration gap ≤ 2 pts ∧ per-symbol coverage clean** |
| **4 — Paper** | ≥ 22 sessions of journals | win-rate gap ≤ 3 pts; positive expectancy; slippage within model (reconcile + fill logs) |
| **5 — Capital** | COMPLIANCE.md checklist + 1 month smallest size | kill switches never breached; expectancy positive; broker SL-M live; rollback rehearsed |

Honesty clause restated: if Gate 3 cannot pass after geometry and feature iteration, the finding is "edge below costs at this horizon" — a legitimate result; the response is a wider data moat, never looser gates.

---

## 21. Execution Order & Milestones

Dependency order (data backfill is the wall-clock critical path, exactly as PLAN_v3 §18 said):

```
M1  (start immediately, parallel):
    §4.2 FII fix + §4.3 missingness   ← before ANY dataset build
    §3.3 raise-on-miss + holidays     ← then START bhavcopy + bar backfill (archive moat)
    §18 conftest + lookahead property test + labeler/bars tests
M2: §9.1 trainer + §10.1 recorder live-bars + §10.2 paper state machine
M3: §3.4 PIT universe + §3.5 corp actions + §3.6 reconcile + §5 screener
M4: §12 governance (journal, MLflow, DVC, drift) + §11 kill switches + §8 backtester rigor
M5: §8.5 geometry study + full Gate-0 run
M6: §13 API + §15 CLI + §16 v1 decommission + §17 Docker + §19 docs
M7: Gate 0 → Gate 1 evidence → PLAN_v3 §18 resumes from its Step 6 (geometry → Gates 2–5)
```

Zero-dependency items startable today: FII fix, missingness policy, raise-on-miss, holidays file, conftest + first tests, and the backfill the moment raise-on-miss lands.

---

## 22. Definition of Production-Ready

The system is production-ready when every box checks, each verified by an executable artifact (not a claim):

- [ ] Gate 0 green in CI (suite §18, coverage ≥ 85%, dead-control sweep clean)
- [ ] `train-v3` produces a registered Staging bundle with model card, reproducible from (git SHA, config hash, DVC rev, seed) — re-run produces identical predictions on a fixture
- [ ] Paper runner completes a real session end-to-end: live screen at 09:30, ≥1 gate evaluation journaled, square-off, post-market kill-switch + drift checks, reconciled bars
- [ ] Kill switches and drift halt demonstrated (forced-breach drill journaled)
- [ ] Promotion + rollback rehearsed; rehearsal in journal
- [ ] API serves all §13 endpoints with auth; 503 path verified under halt
- [ ] v1 fully decommissioned (`test_dead_imports.py` green; images rebuilt without FinBERT)
- [ ] RUNBOOK.md + COMPLIANCE.md complete; README matches reality
- [ ] Gates 1–3 evidence committed under `reports/v3/gates/` (then Gates 4–5 per PLAN_v3 timeline)

---

*Supplementary implementation specification — 2026-06-12. Companion to PLAN_v3.md. The parent plan engineered the 89%; this document specifies the production machine that proves it: every control with a call site, every claim with an artifact, every gate as code.*
