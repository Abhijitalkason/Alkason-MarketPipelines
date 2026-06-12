# PLAN v3 — Intraday Selective Signal System (NSE)

> **Project:** Intraday (2–3 hour window) selective trading-signal system for NSE large caps
> **Goal contract:** ≥ **89% win rate on emitted signals** (engineered via barrier geometry) **AND** ≥ **+0.05% post-cost expectancy per trade** (delivered by model edge). Both must hold — a win rate without expectancy is a system that wins daily and dies quarterly.
> **Trading window:** Entries 09:30–11:00 IST · time barrier entry+3h · hard square-off 14:45 · always flat overnight
> **Supersedes:** PLAN.md (v1, daily 5-class system). v1 post-mortem in Section 1.
> **Plan version:** 2026-06-11

---

## Table of Contents

1. [Why v1 Failed — Post-Mortem Register](#1-why-v1-failed--post-mortem-register)
2. [The 89% Contract — Geometry, Edge, and the Expectancy Identity](#2-the-89-contract)
3. [System Overview & Trading-Day Timeline](#3-system-overview--trading-day-timeline)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Project Structure — Keep / Modify / Delete](#5-project-structure)
6. [Phase 1 — Intraday Data Foundation](#6-phase-1--intraday-data-foundation)
7. [Phase 2 — Morning Momentum Screen](#7-phase-2--morning-momentum-screen)
8. [Phase 3 — Triple-Barrier Labeling & Geometry Tuning](#8-phase-3--triple-barrier-labeling--geometry-tuning)
9. [Phase 4 — Feature Engineering](#9-phase-4--feature-engineering)
10. [Phase 5 — Models (Price Model + Flow Model)](#10-phase-5--models)
11. [Phase 6 — Blend, Calibration & Adaptive Conformal Gate](#11-phase-6--blend-calibration--adaptive-conformal-gate)
12. [Phase 7 — Backtester & Experimentation Protocol](#12-phase-7--backtester--experimentation-protocol)
13. [Phase 8 — Risk Layer](#13-phase-8--risk-layer)
14. [Phase 9 — Paper-Trading Runner](#14-phase-9--paper-trading-runner)
15. [Phase 10 — Serving & Ops](#15-phase-10--serving--ops)
16. [Configuration (config_v3.yaml)](#16-configuration)
17. [Technology Stack Changes](#17-technology-stack-changes)
18. [File Creation Order](#18-file-creation-order)
19. [Acceptance Gates — Go/No-Go](#19-acceptance-gates--gono-go)
20. [Honest Economics & Expectations](#20-honest-economics--expectations)

---

## 1. Why v1 Failed — Post-Mortem Register

Findings from the full end-to-end review (June 2026). These are constraints on v3 design, not blame.

| # | v1 defect | v3 design response |
|---|---|---|
| 1 | 5-class next-day accuracy target (73–85%) had no basis in credible research; real result 23–41% ≈ majority baseline | Win rate engineered from barrier geometry (Section 2), not demanded from prediction |
| 2 | Chronos-2 integration was dead code (API mismatch, exception swallowed → uniform 0.2 forever) | Chronos removed entirely. No silent exception swallowing anywhere in v3 — every model failure raises |
| 3 | FinBERT sentiment: 100% of training rows neutral-imputed; `build_sentiment_df` had zero callers; news cache had lookahead | News/sentiment removed from v3 scope. (Optional far-future: re-add only from a forward-built, timestamped archive) |
| 4 | LSTM trained unscaled (standalone) and on SMOTE-shuffled sequences (ensemble) → degenerate output | LSTM removed. No SMOTE anywhere in v3 (class weights only) |
| 5 | 6-model stack: meta-learner trained on tuning-contaminated OOF; train/serve skew at API (single-row predict) | Two models, one weighted blend, identical feature pipeline train/serve, serving contract specified (Section 15) |
| 6 | yfinance daily data: documented NSE adjustment errors; adjustment seam on incremental merge; no intraday capability | Broker API 1-min data + official NSE bhavcopy as ground truth (Section 6) |
| 7 | Validation/monitoring effectively dead (processed checks never called; drift alerts compared wrong units) | Every gate in v3 has an executable acceptance test; metric contract in Section 12 |
| 8 | DVC silently stale; metadata overwritten | DVC failures raise loudly; append-only run records (Section 12) |

**Carried forward from v1 (the good parts):** config-driven design (nothing hardcoded), MLflow tracking, Docker 3-service deployment, FastAPI serving shell, Evidently drift monitoring (repurposed to trigger conformal recalibration), DVC versioning.

---

## 2. The 89% Contract

### 2.1 The geometry theorem

For a trade with profit target `a` and stop `b` (in ATR units), the probability of hitting the target before the stop on a driftless price path is:

```
P(win | no edge) = b / (a + b)
```

| Target a | Stop b | Baseline win rate | EV before costs |
|---|---|---|---|
| 1.0×ATR | 1.0×ATR | 50.0% | 0 |
| 0.5×ATR | 1.5×ATR | 75.0% | 0 |
| 0.3×ATR | 1.8×ATR | 85.7% | 0 |
| 0.3×ATR | 2.4×ATR | 88.9% | 0 |
| 0.25×ATR | 2.0×ATR | 88.9% | 0 |

The 89% win rate is **purchased by geometry at exactly fair price** — pre-cost expectancy is zero at any geometry with no model. The win-rate number is therefore guaranteed and auditable from day one.

### 2.2 The expectancy identity (the most important equation in this plan)

Let `δ` = the model's edge in probability points above the geometric baseline (achieved by firing only on the best setups), `W = a + b` = total barrier width, `c` = round-trip cost. Then:

```
EV per trade = δ × W − c
```

Profit requires: **δ > c / W**

With round-trip cost c ≈ 0.08–0.10% and ATR(2h) ≈ 0.7% on NSE large caps:

| Geometry (a / b) | Width W | Required edge δ | Fire threshold (calibrated P) |
|---|---|---|---|
| 0.25 / 2.0 ×ATR | 1.58% | ≥ 6.3 pts → realized ≥ 95.2% | impractical |
| 0.3 / 2.4 ×ATR | 1.89% | ≥ 5.3 pts → realized ≥ 94.2% | hard |
| 0.4 / 3.2 ×ATR | 2.52% | ≥ 4.0 pts → realized ≥ 92.9% | target zone |
| 0.5 / 4.0 ×ATR | 3.15% | ≥ 3.2 pts → realized ≥ 92.1% | target zone |

Three consequences that shape the whole system:

1. **Wider total barriers reduce the required edge.** The 3-hour time barrier truncates most stop-side paths before a 3–4×ATR stop is hit, so wide stops are rarely realized in full. Time-barrier exits are scored by P&L sign and (driftless) split roughly 50/50 — they *raise* the share of profitable trades above the pure-geometry baseline. Exact calibration of (a, b, fire-threshold) is an **empirical tuning problem on real 1-min paths** (Phase 3), with the twin constraints: win rate ≥ 89% AND post-cost EV ≥ +0.05%.
2. **Cost control is a first-class feature.** Every basis point of slippage raises the required edge. Liquidity floor on the universe, limit-order entries, and no-trade event days are not optional hygiene — they are profit components.
3. **The model's only job** is to find the 2–6 setups per day where its calibrated win probability clears the fire threshold. Everything else is silence. The published selective-classification evidence (abstention improved every tested configuration; high precision requires low coverage) is consistent with a 3–4 point edge at ≤5% coverage being attainable — but it must be *proven* through the gates in Section 19, never assumed.

### 2.3 The honest framing (written into the contract)

- 89% is the share of **emitted signals** that close profitably. The system is silent on most stock-bars by design.
- If any backtest shows win rate AND high coverage AND high expectancy simultaneously, treat it as a leakage alarm, not success.
- The kill-switch metrics are expectancy and rolling drawdown, not win rate — at this geometry, win rate stays high right up until the system dies. Section 13 exists because of this.

---

## 3. System Overview & Trading-Day Timeline

```
09:00–09:15  PRE-OPEN: capture NSE pre-open auction data (indicative price,
             order imbalance) for the full universe. No trading.
09:15–09:30  OBSERVE: record first 15-min bar. Never trade the opening chaos.
09:30        SCREEN: rank NSE-200 by morning-momentum score on the closed
             15-min bar → today's tradeable set (top 10–15 names).
09:30–11:00  ENTRY WINDOW: every closed 15-min bar, the model evaluates each
             screened name. Fires long/short ONLY when blended, calibrated,
             conformal-gated P(win) ≥ fire threshold. Expect 2–6 signals/day.
entry+3h     TIME BARRIER: exit any position that hasn't hit target or stop.
14:45        HARD SQUARE-OFF: always flat. No exceptions, no overnight risk.
15:45+       POST-MARKET: record day's bars to archive, update labels for
             matured trades, recalibrate conformal threshold (ACI), log run
             to MLflow, weekly retrain check, drift check.
```

Models train ONLY on rows from the 09:30–11:00 window (time-of-day conditioning — a model trained on all-day bars and served only in the morning is train/serve skew, v1's signature bug in a new costume).

---

## 4. Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER (Phase 1)                              │
│                                                                         │
│  Broker API (Upstox/Kite)          NSE official files                   │
│  ├─ 1-min historical bars          ├─ Equity bhavcopy + delivery %      │
│  ├─ live quotes (1-min poll)       ├─ F&O bhavcopy (OI, futures basis)  │
│  └─ L2 depth snapshots (recorded)  ├─ Pre-open auction data             │
│                                    └─ FII/DII flows, India VIX          │
│              │                                  │                       │
│              ▼                                  ▼                       │
│        bars store (parquet, 1-min + 5-min + 15-min)                     │
│        point-in-time: every file stamped with capture time              │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  09:30 MORNING SCREEN (Phase 2)                                         │
│  volume surge × range expansion × gap quality × pre-open imbalance      │
│  → top 10–15 of NSE-200                                                 │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MODEL LAYER (Phases 4–6)                                               │
│                                                                         │
│   ┌──────────────────────┐      ┌──────────────────────┐               │
│   │ PRICE MODEL          │      │ FLOW MODEL           │               │
│   │ LightGBM             │      │ CatBoost             │               │
│   │ bars/VWAP/ORB/       │      │ OI Δ, futures basis, │               │
│   │ momentum features    │      │ delivery %, pre-open  │               │
│   └──────────┬───────────┘      │ imbalance, FII/DII   │               │
│              │                  └──────────┬───────────┘               │
│              └──────────┬─────────────────┘                            │
│                         ▼                                              │
│        Weighted blend (OOF-fit weights, ~0.6/0.4)                       │
│                         ▼                                              │
│        Isotonic calibration (rolling window)                            │
│                         ▼                                              │
│        ADAPTIVE CONFORMAL GATE (ACI, recalibrated daily)                 │
│        fire ⟺ P(win) ≥ threshold  → else SILENCE                       │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  RISK LAYER (Phase 8) — sizing, sector cap, daily loss limit,           │
│  event blackout (expiry/results/RBI/Fed), kill switches                 │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                   ┌───────────────┼────────────────┐
                   ▼               ▼                ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
        │ BACKTESTER   │  │ PAPER RUNNER │  │ FastAPI /signals │
        │ purged walk- │  │ broker API,  │  │ MLflow tracking  │
        │ forward,     │  │ 1-month gate │  │ Evidently drift  │
        │ full costs   │  │              │  │ → ACI recalib    │
        └──────────────┘  └──────────────┘  └──────────────────┘
```

---

## 5. Project Structure

### 5.1 New modules

```
AI-MLOps-Solution/
├── config/
│   └── config_v3.yaml                # All v3 configuration (Section 16)
│
├── data/
│   ├── bars/                         # 1-min parquet per symbol per month
│   ├── bhavcopy/                     # NSE equity + F&O daily files
│   ├── preopen/                      # Pre-open auction snapshots (daily)
│   ├── depth/                        # L2 order-book snapshots (recorded live)
│   └── reference/                    # Universe list, corporate actions, event calendar
│
├── src/intraday/
│   ├── __init__.py
│   ├── data_feed.py                  # Broker API client: historical backfill + live polling
│   ├── recorder.py                   # Live session recorder: bars, pre-open, L2 snapshots
│   ├── bhavcopy.py                   # NSE bhavcopy/F&O/FII-DII downloader + parser
│   ├── bars.py                       # Bar store: 1-min → 5-min/15-min resampling, ATR(2h)
│   ├── screener.py                   # 09:30 morning momentum screen (Phase 2)
│   ├── labeler.py                    # Triple-barrier labeling on 1-min paths (Phase 3)
│   ├── features.py                   # Intraday feature matrix (Phase 4)
│   ├── price_model.py                # LightGBM price model (Phase 5)
│   ├── flow_model.py                 # CatBoost flow/positioning model (Phase 5)
│   ├── blend.py                      # Weighted blend + isotonic calibration (Phase 6)
│   ├── conformal_gate.py             # Adaptive Conformal Inference gate (Phase 6)
│   ├── costs.py                      # Full Indian intraday cost stack (single source of truth)
│   ├── backtester.py                 # Event-driven purged walk-forward backtest (Phase 7)
│   ├── risk.py                       # Sizing, limits, blackout calendar, kill switches (Phase 8)
│   └── paper_runner.py               # Live paper-trading loop via broker API (Phase 9)
│
├── PLAN_v3.md                        # This file
└── PLAN.md                           # v1 — retained as historical record
```

### 5.2 Disposition of every v1 file

| v1 file | Disposition | Reason |
|---|---|---|
| `src/data/ingestion.py` | **Modify** | Demote to daily-context fetcher; primary prices move to bhavcopy/broker API |
| `src/data/validation.py` | **Modify** | Rewrite schemas for 1-min bars + bhavcopy; every check must be *called* by the pipeline and fail loudly |
| `src/data/features.py` | **Keep (reduced)** | Daily context features only (prior-day levels, regime); fix sma_200 zero-fill + VWAP bugs if retained |
| `src/slm/news_scraper.py` | **Delete** | No point-in-time integrity; proven zero contribution |
| `src/slm/sentiment.py` | **Delete** | Dead pipeline in v1; out of v3 scope |
| `src/slm/explainer.py` | **Keep** | Granite explanation of *emitted signals* is genuinely useful UX; explain fired signals only |
| `src/models/lgbm_model.py` | **Modify** | Becomes basis of `price_model.py`; fix SHAP ndarray handling, final-refit early stopping |
| `src/models/xgboost_model.py` | **Delete** | Redundant with LightGBM; diversity comes from the flow model's different *data*, not a third tree library |
| `src/models/lstm_model.py` | **Delete** | Degenerate in v1; no evidence it beats GBDT on intraday tabular features |
| `src/models/chronos_model.py` | **Delete** | Dead code in v1; published negative R² on equity returns |
| `src/models/autogluon_model.py` | **Delete** | Majority-class collapse in v1; remove from serving deps |
| `src/models/ensemble_model.py` | **Delete** | Replaced by `blend.py` (2 models, OOF-fit weights, no meta-learner) |
| `src/training/manual_trainer.py` | **Modify** | Becomes thin orchestration over price/flow model training + blend fit |
| `src/training/mlflow_trainer.py` | **Modify** | Same training + the v3 metric contract (Section 12.3); fix registry to log the full blend+gate pipeline |
| `src/evaluation/evaluator.py` | **Rewrite** | Evaluates win rate / expectancy / coverage frontier — not accuracy |
| `src/evaluation/monitoring.py` | **Modify** | Fix Evidently v0.7 result parsing + units bug; drift alert triggers ACI recalibration |
| `src/api/app.py` | **Modify** | New endpoints (Section 15); fix blocking-async, serving contract |
| `main.py` | **Modify** | New modes: `record`, `screen`, `label`, `backtest`, `paper`, `signals` |
| `docker/*` | **Keep (update)** | Same 3-service stack; api image gets catboost + mapie, drops autogluon-related conflicts |
| DVC setup | **Keep (fix)** | Per-run tracking, failures raise, remote moved off /tmp |

---

## 6. Phase 1 — Intraday Data Foundation

**Files:** `src/intraday/data_feed.py`, `recorder.py`, `bhavcopy.py`, `bars.py`
**This phase gates everything.** No model work starts until Gate 1 (Section 19) passes.

### 6.1 Sources (ranked by information value)

| Tier | Source | Content | Access | Cost |
|---|---|---|---|---|
| 1 | Upstox API (or Kite Connect) | Historical 1-min candles (~2022→), live quotes | REST + websocket | Free (Upstox) / ₹500–2000/mo (Kite) |
| 1 | NSE pre-open auction | Indicative open, order imbalance 09:00–09:15 | nseindia.com JSON endpoints | Free |
| 1 | NSE equity bhavcopy | EOD OHLCV + **delivery %** (ground-truth prices) | Daily file download | Free |
| 1 | NSE F&O bhavcopy | Stock futures **OI, basis**; index futures/options OI | Daily file download | Free |
| 2 | NSE FII/DII flows | Daily institutional net flows | Daily | Free |
| 2 | India VIX + index 1-min bars | Regime + concurrent index moves | Broker API | Free |
| 2 | L2 order-book depth | 5-level bid/ask sizes | Broker websocket — **record live from day 1** | Free |

### 6.2 Implementation requirements

- **Backfill:** download 1-min bars for NSE-200, as far back as the provider allows (target ≥3 years). Store monthly parquet per symbol: `data/bars/<SYMBOL>/<YYYY-MM>.parquet`. Resample to 5-min/15-min on read (never store derived bars without the source).
- **Corporate-action adjustment:** built from NSE corporate-action files, applied as explicit factor columns (never destructive in-place adjustment — v1's adjustment-seam bug must be impossible by construction).
- **Live recorder:** runs every market day from 08:55; captures pre-open snapshot, 1-min bars (poll or websocket), L2 snapshots at 1-min cadence for screened names. Everything stamped with capture time → the morning screen is point-in-time backtestable forever.
- **Validation (Pandera, actually wired in):** monotonic timestamps, exchange-session alignment (09:15–15:30 IST), no zero/negative prices, volume ≥ 0, bar-count-per-day sanity, cross-check daily aggregate of 1-min bars vs bhavcopy close (±0.1%). Validation failure **stops the pipeline loudly** — no drop-and-continue.
- **Event calendar:** expiry days, stock results dates, RBI/Fed/budget days — maintained in `data/reference/events.csv`, consumed by risk layer.

---

## 7. Phase 2 — Morning Momentum Screen

**File:** `src/intraday/screener.py` — runs at 09:30:05 on the closed 09:15–09:30 bar.

```
score(symbol) =
    w1 · z(first-15min volume   vs 20-day same-bar average)
  + w2 · z(first-15min range    vs 20-day same-bar average)
  + w3 · gap_quality            (|gap| in 0.3%–2.5% band; beyond = exhaustion risk)
  + w4 · preopen_imbalance      (signed normalized buy−sell imbalance)
  + w5 · prior_day_delivery_z   (institutional conviction)
  + w6 · index_alignment        (stock vs sector vs NIFTY first-15min direction agreement)
```

- Universe: NSE-200 minus liquidity failures (median spread > 5 bps or median 1-min volume below floor) minus event-blackout names for the day.
- Output: top `screen.top_n` (default 12) with direction hint (long-bias / short-bias from gap + imbalance sign).
- Weights `w1..w6` are fit once on the training window by logistic regression against "did this name produce a triple-barrier win that day" — then **frozen** per walk-forward fold (the screen is a model too; it gets the same leakage discipline).
- **Point-in-time rule:** the screen may consume nothing time-stamped after 09:30:00. Enforced by the backtester replaying only data with capture-time ≤ decision time.

---

## 8. Phase 3 — Triple-Barrier Labeling & Geometry Tuning

**File:** `src/intraday/labeler.py`

For each (symbol, entry-bar) in the 09:30–11:00 window of screened names:

```
entry  = next 15-min bar open after signal bar close (realistic fill)
ATR    = ATR(2h) computed from trailing 5-min bars, as of entry
target = entry ± a·ATR     (direction-dependent)
stop   = entry ∓ b·ATR
t_bar  = entry_time + 3h   (and never past 14:45)

label  = 1 if target touched first (scan 1-min path)
         0 if stop touched first
         sign(PnL at exit) if time barrier hit first
```

- Touch detection on the **1-min path**, conservative tie-break (if a 1-min bar spans both barriers, count as stop — pessimistic by construction).
- **Geometry tuning protocol:** grid over `a ∈ {0.25..0.6}`, `b ∈ {1.5..4.0}` evaluated **on the training folds only**, selecting the geometry that satisfies (i) baseline+model win rate ≥ 89%, (ii) post-cost EV ≥ +0.05%/trade at achievable coverage, via the identity `EV = δ·(a+b)·ATR − c`. Starting point: `a=0.4, b=3.2`. The chosen geometry is **frozen before the test folds are ever touched.**
- One label row per (symbol, 15-min bar) in the entry window → ~6 bars × 12 screened names ≈ 70 candidate rows/day ≈ 50k+ rows over 3 years. Two orders of magnitude more training data than v1's 1,193 daily rows.

---

## 9. Phase 4 — Feature Engineering

**File:** `src/intraday/features.py` — every feature answers one question: *given this morning setup, does the path continue `a·ATR` before retracing `b·ATR`?*

| Group | Features (~40 total) |
|---|---|
| Opening range | position of price in 15-min opening range; breakout distance in ATR; retest-and-hold flag; minutes since breakout |
| VWAP | (price − VWAP)/ATR; VWAP slope; consecutive bars above/below |
| Gap | gap % ; fraction filled; gap direction × first-15min direction agreement |
| Volume | cumulative volume vs 20-day same-time-of-day norm (the U-curve normalization); volume slope last 3 bars; large-trade share (from L2 when available) |
| Momentum | 5-min return ladder (1,3,6,12 bars); acceleration; RSI(14) on 5-min; high-of-day proximity |
| Flow (CatBoost model) | futures OI Δ vs prior day; basis (futures − spot)/spot; prior-day delivery % z-score; pre-open imbalance; FII/DII 5-day net z |
| Context | NIFTY + sector concurrent 15-min move; India VIX level + 1-day change; day-of-week; minutes since open |
| Regime | 20-day realized vol percentile; ADX(daily) from v1 daily pipeline |

Rules: every rolling statistic uses data strictly before the decision bar; time-of-day normalization mandatory for volume/range features; no forward-fill across session boundaries; feature matrix schema is versioned and **checked at serve time against the training schema** (v1's silent-skew bug is structurally prevented).

---

## 10. Phase 5 — Models

### 10.1 Price model — `src/intraday/price_model.py`
- **LightGBM binary classifier**: P(label=1) on price/volume/momentum/context features.
- Class weights for imbalance (geometry makes wins the majority class — weight accordingly). **No SMOTE anywhere in v3.**
- Optuna ≤ 30 trials, tuned on an **inner split of the training fold only** (never the OOF/validation block — v1's contamination bug is structurally excluded), params frozen before OOF prediction.
- Monotonicity constraints where economically justified (e.g., spread → P(win) non-increasing).

### 10.2 Flow model — `src/intraday/flow_model.py`
- **CatBoost binary classifier** on flow/positioning features only (OI, basis, delivery, pre-open imbalance, FII/DII).
- Deliberately *blind* to the price-path features — its value is decorrelation via a different information channel, not architecture novelty.

### 10.3 What is deliberately absent
LSTM, Chronos, transformers, AutoGluon, stacking meta-learner, sentiment. Each was either degenerate in v1, contradicted by published evidence on this task, or adds failure surface without adding an information channel. Two strong models on two channels beat six weak ones on one.

---

## 11. Phase 6 — Blend, Calibration & Adaptive Conformal Gate

**Files:** `src/intraday/blend.py`, `src/intraday/conformal_gate.py`

```
P_blend = w·P_price + (1−w)·P_flow        # w fit on OOF predictions (expect ≈0.6)
P_cal   = isotonic(P_blend)               # fit on a rolling 60-day calibration window
FIRE    ⟺ P_cal ≥ τ_t                     # τ_t from Adaptive Conformal Inference
```

- **ACI (Gibbs–Candès):** target miscoverage α set so realized win rate ≥ 89%; threshold τ_t updated **daily** after labels mature: `τ_{t+1} = τ_t + γ·(α − err_t)`. Static conformal thresholds provably lose coverage in regime shifts; intraday bars violate exchangeability — online adaptation is mandatory, not optional.
- Agreement filter: fire only if both models individually exceed a floor (P_price, P_flow ≥ 0.75) — a blend hiding total disagreement is a low-quality signal.
- Expected coverage at the required δ: **2–6 signals/day** from 12 screened names × 6 bars. If tuned coverage collapses below ~1 signal/day, the edge is insufficient — that is Gate 3 failing honestly (Section 19), not a parameter to fudge.
- Library: `mapie` for conformal machinery + custom ACI update loop.

---

## 12. Phase 7 — Backtester & Experimentation Protocol

**File:** `src/intraday/backtester.py`

### 12.1 Backtest engine
- Event-driven replay on 1-min data: screen at 09:30 → features at each 15-min close → gate → fill at next bar open with slippage model → barrier resolution on the 1-min path → costs from `costs.py`.
- **Slippage model:** half-spread + impact term scaled by (order size / median 1-min volume); conservative defaults (0.03% large caps), stress-tested at 2× (Gate must survive 2× slippage).
- **Cost stack** (`costs.py`, single source of truth): brokerage (₹20/order flat), STT 0.025% sell-side, exchange txn 0.00297%, SEBI 0.0001%, stamp 0.003% buy-side, GST 18% on (brokerage+txn). Round trip ≈ 0.045% statutory + slippage ≈ **0.08–0.11% total**.

### 12.2 Walk-forward protocol
- Expanding window: train ≥ 18 months → calibrate 60 days → trade out-of-sample 3 months → roll. Minimum 6 out-of-sample folds (~2 years OOS).
- **Purging + embargo:** drop training rows whose barrier windows overlap the calibration/test boundary; 1-day embargo.
- The screen weights, geometry, hyperparameters, blend weight, and τ are all frozen per fold from training data only.
- **Leakage alarm rule (executable):** if any fold shows win rate > 92% at coverage > 10%, the run fails with a leakage investigation, not a celebration.

### 12.3 MLflow metric contract (replaces v1's accuracy tables)
Logged per fold and aggregate: win rate (primary, target ≥89%), **post-cost expectancy/trade (kill-switch metric, target ≥ +0.05%)**, coverage (signals/day), win-rate-vs-coverage frontier curve (artifact), realized δ vs geometric baseline, max drawdown, worst day, profit factor, Sharpe (daily P&L), per-fold stability table, calibration reliability diagram (artifact). "Accuracy" does not appear anywhere.

### 12.4 Versioning
Every run: append-only run record (never overwrite metadata — v1 bug), DVC add of model + gate state with **failures raising loudly**, DVC remote on durable storage (not /tmp).

---

## 13. Phase 8 — Risk Layer

**File:** `src/intraday/risk.py` — exists because at 0.4/3.2 geometry one full stop erases ~8 winners. High-win-rate systems die by tails; this layer is load-bearing.

| Control | Rule |
|---|---|
| Position sizing | Fixed ₹-risk per trade: size = (capital × 0.4%) / stop-distance — every full stop costs the same |
| Concurrency | Max 4 open positions; max 2 per sector |
| Daily loss limit | −1.5% of capital → flat everything, no new signals today |
| Event blackout | No signals on: monthly/weekly index expiry days, the stock's results day ± 1, RBI/Fed/budget days |
| Stop discipline | Stops are broker-side orders (SL-M), never mental/software-only |
| Kill switches | (a) rolling 20-day expectancy < 0 → halt, retrain, re-gate; (b) realized win rate 10 points below backtest over 30 trades → halt (calibration broken); (c) drift alert (Evidently) → force ACI recalibration before next session |
| Leverage | MIS intraday leverage capped at 3× notional (broker allows ~5×; we don't take it) |

---

## 14. Phase 9 — Paper-Trading Runner

**File:** `src/intraday/paper_runner.py`

- Runs the full live loop (recorder → screen → features → gate → simulated orders at real quoted prices) every market day for **at least 22 trading days**.
- Logs every signal with the full feature vector, gate state, and simulated fill vs actual quote — this is the train/serve-skew detector.
- **Pass criterion (Gate 4):** live win rate within 3 points of backtest, live expectancy positive, live slippage within model assumptions. A 10-point gap means screen lookahead or fill fantasy — system returns to Phase 7.

---

## 15. Phase 10 — Serving & Ops

**File:** `src/api/app.py` (modified)

| Endpoint | Description |
|---|---|
| `GET /health` | Service + data-feed + model freshness status |
| `GET /signals/today` | Signals fired today: symbol, direction, entry, target, stop, P(win), explanation |
| `GET /signals/history?days=N` | Past signals with outcomes — the live track record, win rate, expectancy |
| `GET /screen/today` | This morning's screened names + scores |
| `GET /performance` | Rolling win rate, expectancy, drawdown vs gate thresholds |
| `GET /drift` | Drift status + last ACI recalibration |
| `POST /backtest` | Trigger backtest job (job-id tracked, body = config overrides) |

Serving fixes mandated from v1 review: model + gate loaded at startup (not per request); blocking inference moved off the event loop (`run_in_executor` / worker); feature schema check before every predict; **the serving path consumes the identical feature pipeline as training** (same module, same functions — no reimplementation in app.py); explainer (Granite) explains fired signals only, asynchronously.

Docker: same 3 services. API image: + catboost, mapie; − autogluon/optuna conflicts (training stays host-side or in a separate training image). MLflow + Ollama unchanged.

---

## 16. Configuration

**File:** `config/config_v3.yaml`

```yaml
universe:
  source: nse200                  # data/reference/universe.csv
  min_median_1min_volume: 5000
  max_median_spread_bps: 5

data:
  provider: upstox                # upstox | kite
  bars_path: data/bars
  backfill_years: 3
  record_depth_levels: 5

screen:
  time: "09:30:05"
  top_n: 12
  gap_band: [0.003, 0.025]
  lookback_days: 20

geometry:                          # tuned in Phase 3, then frozen per fold
  target_atr: 0.40
  stop_atr: 3.20
  time_barrier_hours: 3
  squareoff: "14:45"
  atr_window_minutes: 120

gate:
  fire_threshold_init: 0.92        # τ_0; ACI adapts daily
  aci_gamma: 0.01
  target_win_rate: 0.89
  model_agreement_floor: 0.75
  calibration_window_days: 60

training:
  train_min_months: 18
  test_fold_months: 3
  embargo_days: 1
  optuna_trials: 30
  blend_weight_init: 0.6

costs:
  brokerage_per_order_inr: 20
  stt_sell_pct: 0.00025
  exchange_txn_pct: 0.0000297
  stamp_buy_pct: 0.00003
  gst_pct: 0.18
  slippage_pct: 0.0003
  slippage_stress_multiplier: 2.0

risk:
  risk_per_trade_pct: 0.004
  max_positions: 4
  max_per_sector: 2
  daily_loss_limit_pct: 0.015
  max_leverage: 3.0
  blackout: [expiry, results, rbi, fed, budget]

gates:                             # Section 19, machine-checkable
  min_win_rate: 0.89
  min_expectancy_pct: 0.0005
  min_signals_per_day: 1.0
  leakage_alarm_win_rate: 0.92
  leakage_alarm_coverage: 0.10
  paper_days: 22
  paper_max_winrate_gap: 0.03
```

---

## 17. Technology Stack Changes

| Component | v1 | v3 | Reason |
|---|---|---|---|
| Intraday data | — | Upstox/Kite API + NSE bhavcopy | The information substrate; yfinance cannot do this |
| Primary model | LightGBM (1 of 6) | LightGBM (primary of 2) | Survives every credible benchmark |
| Second model | XGB/LSTM/Chronos/AutoGluon | CatBoost on flow features | Decorrelation via data channel, not architecture |
| Imbalance | SMOTE | class weights | SMOTE on time series was a v1 root-cause bug |
| Selectivity | — | mapie + custom ACI | The 89% gate; online recalibration mandatory |
| Labels | 5-class next-day return | triple-barrier on 1-min paths | Where the 89% is engineered |
| Metric | accuracy | win rate + expectancy + coverage | Accuracy is banned from dashboards |
| Sentiment | FinBERT + ddgs | removed | Zero contribution + unfixable lookahead in v1 |
| Explanation | Granite 4.1 (all predictions) | Granite 4.1 (fired signals only) | Kept — genuinely useful |
| Tracking/Deploy | MLflow, Docker, DVC, Evidently | same, with v1 bugs fixed | Good infrastructure, keep |

---

## 18. File Creation Order

```
Step 1:  config/config_v3.yaml, data/reference/universe.csv, events.csv
Step 2:  src/intraday/data_feed.py + bhavcopy.py     # backfill starts NOW (longest lead time)
Step 3:  src/intraday/bars.py + validation rewiring
Step 4:  src/intraday/recorder.py                     # live recording starts NOW (archive moat)
Step 5:  src/intraday/costs.py
Step 6:  src/intraday/labeler.py                      # + geometry tuning study
Step 7:  src/intraday/screener.py
Step 8:  src/intraday/features.py
Step 9:  src/intraday/price_model.py, flow_model.py
Step 10: src/intraday/blend.py, conformal_gate.py
Step 11: src/intraday/backtester.py                   # → GATES 2–3
Step 12: src/intraday/risk.py
Step 13: src/intraday/paper_runner.py                 # → GATE 4 (22 sessions)
Step 14: src/api/app.py modifications, main.py modes, Docker updates
Step 15: v1 deletions (slm/news_scraper, sentiment, xgboost/lstm/chronos/autogluon/ensemble models)
```

Steps 2 and 4 start on day one — historical backfill and the live archive are the schedule's critical path.

---

## 19. Acceptance Gates — Go/No-Go

Each gate is machine-checkable; failing a gate sends the project back, never forward.

| Gate | Test | Pass criterion |
|---|---|---|
| **1 — Data** | Backfill + 5 recorded live sessions; bhavcopy cross-check | ≥3y 1-min bars for ≥180 names; bar-vs-bhavcopy mismatch <0.1%; recorder uptime 5/5 days |
| **2 — Label/geometry** | Geometry grid on training folds | A geometry exists with baseline ≥86% and required δ ≤ 4 pts at projected coverage |
| **3 — Backtest** | Full purged walk-forward, all costs, 2× slippage stress | Win rate ≥89%, expectancy ≥ +0.05%/trade, ≥1 signal/day, no leakage alarm, survives stress |
| **4 — Paper** | 22 live sessions | Win-rate gap vs backtest ≤3 pts; positive expectancy; slippage within model |
| **5 — Capital** | Smallest viable size, 1 month | Kill switches never breached; expectancy positive |

**Honesty clause:** if Gate 3 cannot be passed after geometry tuning and feature iteration, the finding is "the edge at this horizon with this data is below costs" — that is a legitimate, valuable result, and the response is to widen the data advantage (longer L2 archive, better flow features), not to loosen the gates.

---

## 20. Honest Economics & Expectations

Worked example at Gate-3 minimums (conservative):

```
4 signals/day × 0.05% expectancy        = +0.20%/day on traded notional
Risk-managed deployment ≈ 60% capital at 3× leverage on signal days
→ ~0.25–0.4% on capital per active day, ~18–20 active days/month
→ ~4–7%/month gross of the inevitable losing streaks; drawdowns of
  5–10% WILL occur (a 4-loss day at this geometry ≈ −1.5% limit hit)
```

| Quantity | Committed target | World-class |
|---|---|---|
| Win rate on emitted signals | ≥ 89% | 90–92% |
| Post-cost expectancy/trade | ≥ +0.05% | +0.10–0.15% |
| Signals/day | 2–6 | 4–8 |
| Edge δ over geometric baseline | 3–4 pts | 5 pts |
| Backtest→live win-rate gap | ≤ 3 pts | ≤ 1 pt |

What this system will never do — written here so it is never re-promised: predict every stock every day; sustain high win rate at high coverage; survive without the risk layer; keep working without daily recalibration and weekly retraining. The moat is the verified pipeline + the growing point-in-time archive (pre-open, L2, screen snapshots) — an asset competitors cannot backfill, which compounds every trading day the recorder runs.

---

*Plan v3 — 2026-06-11. Supersedes PLAN.md (v1). The 89% is engineered in Section 8, earned in Sections 9–11, proven in Section 19, and kept alive in Section 13.*
