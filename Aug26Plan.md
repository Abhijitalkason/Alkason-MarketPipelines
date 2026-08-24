# Aug26Plan — Automated Pre‑Market Stock Analysis & Buy‑Recommendation System

> **Document type:** Requirements specification (living document — Step 1 of N).
> **Created:** 2026‑07‑29 · **Owner:** user (nitink) · **Author of draft:** engineering.
> **Status:** DRAFT — requirement capture. The *per‑stock analysis method* is
> intentionally **deferred to the next step** ("what analysis and how" — to be
> discussed and appended as §6 detail).
> **Supersedes:** the abandoned v1/v3/v4/v6 plans (see [§13 Lineage & cleanup](#13-lineage--cleanup-what-this-replaces)).
> This plan is the "latest updated requirement" that [ACTION_ITEM.md](ACTION_ITEM.md)
> was waiting for; nothing is deleted until the keep/retire list in §13 is approved.

---

## 1. Purpose (one paragraph)

Build an **automated, observable, daily pre‑market workflow** that, every trading
morning *before the Indian market opens*, (1) fetches the latest stock data from
**Upstox** and other trustworthy APIs, (2) selects the **high‑volume / most‑liquid**
stocks, (3) runs an **end‑to‑end detailed analysis on each selected stock**, (4)
streams **step‑by‑step logs / step / status per stock to a UI page** so a human can
watch exactly what the system is doing in the background, and (5) at the final step
presents a ranked list of **BUY recommendations, each with its complete probability
details and a confidence value**. The system must **reuse the existing AI/ML
pipeline** already in this repository rather than rebuild it.

> **"Automatic" = the daily data→analysis→recommendation pipeline runs itself.**
> It does **not** mean automatic order placement. Placing live orders stays
> **out of scope** and gated (see [§10 Compliance](#10-compliance--guardrails-non-negotiable)).
> The product is a **decision‑support / recommendation** system: a human acts on
> the output.

---

## 2. Critical inherited context — read before anything else

This repository already contains a mature MLOps system for exactly this problem
domain. Two facts from that history are **load‑bearing** and shape every
requirement below — they are not optional caveats:

1. **The prior swing/geometry premise was measured and KILLED.** PLAN_v6's Phase 0
   pre‑registered gate fired a **KILL** (0 of 48 geometry cells passed; win rate
   tops out at ≈0.711, below the 0.72 floor) and the project was **STOPPED**. See
   [SOLUTION_FLOW.md](SOLUTION_FLOW.md) and [reports/v6/phase0_report.md](reports/v6/phase0_report.md).
2. **No model in this repo has demonstrated a reliable predictive edge** at daily
   / 1–5‑day horizons (walk‑forward AUC ≈ 0.50). The code says so honestly
   everywhere — e.g. [src/analysis/engine.py](src/analysis/engine.py) `model_view()`
   and the standing `DISCLAIMER`.

**What this means for THIS plan (a hard requirement, not a footnote):**

- Every "probability" and "confidence value" we show is a **calibrated model
  output plus a transparent evidence score** — **not** a proven edge. The UI and
  the final recommendation list **must** carry the honest measured‑edge note and a
  running scoreboard, never dress a number up as a promise.
- **Primary near‑term goal (user decision 2026‑07‑29): pursue a real, measurable
  predictive edge.** This is legitimate — but it is pursued the *only* honest way:
  through a **fresh, pre‑registered walk‑forward backtest gate** on the new analysis
  logic, **not** by tuning the thresholds of the old KILL. The pre‑existing gate
  scaffolding ([config/config_daily.yaml](config/config_daily.yaml) `gate.milestone1/2`,
  [src/daily/backtester.py](src/daily/backtester.py)) is exactly this instrument and is reused.
  **The edge must be *earned on out‑of‑sample data*, not asserted.** It may KILL
  again — that outcome is cheap and acceptable; shipping a *false* edge is not.
- In parallel we still deliver the honest substrate — **automation, transparency,
  liquidity‑filtered coverage, disciplined pre‑committed trade plans, and an
  auditable probability + confidence tracked for honesty over time** — so the
  product is useful even while the edge is being proven (or disproven).
- This is a real‑money system. Over‑stating confidence is a user‑harm, so the
  honesty scaffolding already in the code (`confidence`, `data_completeness`,
  `actionable` flag, `TOP_PICK` vs `STRONG_SIGNAL` tiers, provisional‑live
  labelling) is **kept and required**, not removed.

---

## 3. Scope

### 3.1 In scope
- Scheduled, automatic **pre‑market daily run** (before 09:15 IST market open).
- **Data acquisition** from Upstox + NSE + supporting trustworthy APIs.
- **High‑volume universe selection** (liquidity‑filtered, point‑in‑time).
- **Per‑stock end‑to‑end analysis** (framework here; method detail in the next step).
- **Live UI observability**: per‑step and per‑stock status, logs and streaming.
- **Final BUY‑recommendation output**: ranked, each with probability + confidence
  + a full, pre‑committed trade plan + honest labels.
- **Reuse** of the existing `src/daily`, `src/analysis`, `src/intraday`, `src/api`
  pipeline and its MLOps scaffolding (MLflow, DVC, calibration, gates, scoreboard).

### 3.2 Out of scope (this phase)
- **Automated order execution / live trading with real capital.** Explicitly
  gated behind backtest + paper + compliance + broker approval (see §10, §11).
- **Intraday (minute‑level) trading signals** as the primary product. The 1‑min
  bar pipeline exists and *may* feed context, but the primary horizon is the
  **1–5 trading‑day swing** entered at the next open (config already set:
  [config/config_daily.yaml](config/config_daily.yaml) `horizons.swing_1_5d`).
- **Short‑selling recommendations.** Long‑only for now (retail cannot hold
  overnight cash‑segment shorts on NSE). SELL output = exit/avoid guidance only.
- **Re‑litigating the KILL** by tuning thresholds. Any claim of predictive edge
  must come from a *fresh, pre‑registered* backtest gate, not from re‑labelling.

---

## 4. Actors & primary use case

| Actor | Interest |
|---|---|
| **Trader (user)** | Opens the UI in the morning, watches the run, reads the ranked BUY list with probability + confidence, decides what to trade. |
| **Scheduler / cron** | Fires the pipeline automatically pre‑market; no human needed to start it. |
| **Operator** | Handles token refresh, halts, drift, holidays ([RUNBOOK.md](RUNBOOK.md)). |

**Primary use case (the "happy path"):**
> *At ~08:30 IST the pipeline auto‑starts. The trader opens the UI, sees each step
> light up (data fetch → universe → per‑stock analysis → ranking), can drill into
> any individual stock's live log, and by market open has a ranked list of BUY
> recommendations — each showing its calibrated probability, confidence value,
> the evidence behind it, and a ready trade plan (entry/stop/target/quantity/cost).*

---

## 5. End‑to‑end functional flow (the six stages)

Each stage maps to code that **already exists** and is orchestrated today by
[`main.py --mode daily-pipeline`](main.py) (`cmd_daily_pipeline`). The new plan
**re‑scopes and hardens** this flow; it does not start from zero.

```
[0] Pre‑market trigger (scheduler, before 09:15 IST)
        │
        ▼
[1] Data acquisition ──────────► Upstox (live snapshot + 1‑min bars),
        │                        NSE bhavcopy (EOD full market), yfinance
        │                        (global/macro + fundamentals), GDELT (news),
        │                        FII/DII
        ▼
[2] High‑volume universe select ─► PIT top‑N by trailing turnover (liquidity floor)
        │
        ▼
[3] Per‑stock end‑to‑end analysis ─► technicals · flow · fundamentals · news ·
        │  (METHOD DETAIL = NEXT STEP)   regime · model P(win) · scorecard · verdict
        ▼
[4] Live observability UI ─────────► per‑step status + per‑stock trace + live stream
        │
        ▼
[5] BUY recommendations ───────────► ranked list; each = probability + confidence +
                                     why + pre‑committed trade plan + honesty labels
```

### Stage 0 — Pre‑market trigger *(new/expand)*
- The run must start **automatically before market open**, with no manual step.
- Options (decision in §14): OS cron / launchd, the existing `/pipeline/run`
  endpoint hit by a scheduler, or the repo's `schedule`/`loop` tooling.
- Must respect the **NSE trading calendar** ([data/reference/nse_holidays.csv](data/reference/nse_holidays.csv)) —
  no run on holidays/weekends.
- **Timing reality (must be documented in the UI):** NSE's *final* EOD bhavcopy for
  a session is published ~18:30 IST **that evening**. So a pre‑market run at, say,
  08:30 IST works from **yesterday's final close** + **fresh overnight global/macro**,
  and recommends entries for **today's 09:15 open**.
- **Data basis — user decision 2026‑07‑29: run on BOTH.** The reliable base is
  **yesterday's official EOD close** (final, complete, no intraday gaps). Alongside
  it, the run also produces a **live Upstox pre‑open snapshot preview**, clearly
  labelled `PROVISIONAL` (close not final; delivery %/OI/FII absent intraday). The
  UI presents them side by side so the user sees the settled signal and the live
  preview without confusing the two. Both paths already exist —
  [`live_snapshot.py`](src/daily/live_snapshot.py) + the pipeline's "fall back to
  last session on disk" logic — and just need to run and render together.

### Stage 1 — Data acquisition *(reuse)*
Fetch the latest data from Upstox and supporting trustworthy sources. See the
[data‑source table in §7](#7-data-sources). Each source **fails soft and is
traced** — a dead optional source degrades a feature to `_present=0`, it never
fabricates or silently drops. Reused modules:
- NSE EOD (full market ~2,400 symbols, incl. delivery %): [src/intraday/bhavcopy.py](src/intraday/bhavcopy.py)
- Upstox live pre‑open/intraday snapshot: [src/daily/live_snapshot.py](src/daily/live_snapshot.py)
- Upstox 1‑min bar history (context): [src/intraday/data_feed.py](src/intraday/data_feed.py)
- Global/macro (S&P, Nasdaq, crude, USD/INR, India VIX…): [src/daily/global_data.py](src/daily/global_data.py)
- FII/DII flow: [src/daily/fii_backfill.py](src/daily/fii_backfill.py)
- Macro/event calendar: [src/daily/macro.py](src/daily/macro.py)

### Stage 2 — High‑volume universe selection *(reuse)*
- Select the **most‑liquid / high‑volume** stocks: **PIT top‑N by trailing
  6‑month median daily turnover**, recomputed monthly from the full NSE market,
  survivorship‑free, with a liquidity floor and a circuit‑lock exclusion.
- Modules: [src/v6/universe.py](src/v6/universe.py) (`monthly_pit_universe`),
  built by [scripts/build_daily_universe.py](scripts/build_daily_universe.py),
  stored at [data/reference/universe_top100.csv](data/reference/universe_top100.csv).
  Knobs in [config/config_daily.yaml](config/config_daily.yaml) `universe:`
  (`min_median_daily_turnover_lacs: 100`, top‑100).
- **★ NEW hard eligibility filter — market cap ≤ ₹50,000 Cr (user directive 2026‑07‑30):**
  only stocks whose market capitalisation is **not greater than ₹50,000 crore** are
  eligible. Applied **before** the liquidity ranking, so the universe becomes
  *"the most‑liquid / high‑volume names **among stocks ≤ ₹50,000 Cr**"* → then top‑N of those.
  - **Effect (confirm this is the intent):** this **excludes the mega‑caps**
    (RELIANCE, TCS, HDFC Bank… all ≫ ₹50,000 Cr) and points the system at
    **mid‑caps and below** (₹50,000 Cr sits around the SEBI large/mid‑cap boundary).
    The liquidity floor (`min_median_daily_turnover_lacs`) still applies, so the
    mid‑caps that survive are the **tradeable** ones — not illiquid micro‑caps.
  - **Market‑cap data source + PIT handling:** market cap = shares outstanding ×
    price. Options: (a) yfinance `marketCap` (already used per‑stock in the analyzer,
    [src/analysis/engine.py](src/analysis/engine.py)); (b) a maintained NSE/Upstox
    **shares‑outstanding** reference × the as‑of close. For the **live** daily
    selection, current market cap is exactly right. For the **M‑Edge backtest**
    universe it must be **PIT‑honest** — trailing shares outstanding × the historical
    close, refreshed on the monthly rebuild — never today's cap applied to history
    (that would leak). New reference store `data/reference/market_cap.csv` (see §7A).
  - **Config‑driven (added 2026‑07‑30):** the ceiling is a single tunable knob in
    [config/config_daily.yaml](config/config_daily.yaml) →
    `universe.max_market_cap_cr: 50000` (with `universe.market_cap_file`). Change
    that one number to widen/narrow the band; set `0`/null to disable. It is
    consumed at universe‑build time ([scripts/build_daily_universe.py](scripts/build_daily_universe.py)
    → [src/v6/universe.py](src/v6/universe.py) `monthly_pit_universe`) — wiring the
    filter into that build is the implementation task (the knob exists now).
- **Partly‑resolved decision (§14):** the universe is now **liquidity/volume‑ranked
  AND capped at ≤ ₹50,000 Cr**. The one bit still open is the *exact* liquidity
  metric — ₹ **turnover** (current, the liquidity standard) vs. raw **share‑volume**
  vs. **volume‑surge** ("unusual volume today") — and the universe size (top‑N).

### Stage 3 — Per‑stock end‑to‑end analysis *(reuse framework; **method = next step**)*
For **each** selected stock, run a full background analysis. The existing engine
[src/analysis/engine.py](src/analysis/engine.py) `analyze(symbol)` already produces,
per stock:

| Section | Content (existing) |
|---|---|
| identity | name, sector, industry, history source, sessions on disk |
| technicals | returns 1d/1w/1m/3m/6m/1y, SMA 20/50/200, RSI(14), ATR%, 52w position, max drawdown, annualized vol, volume‑vs‑20d ratio |
| flow | delivery % (level + 60‑day z), futures OI change, basis % |
| fundamentals | yfinance: PE, market cap, P/B, dividend yield, beta, analyst rec/target |
| news | GDELT 30‑day sweep, bucketed today/week/month, scored by FinBERT (or disclosed lexicon fallback) |
| regime | India VIX + global tone |
| model_view | calibrated **P(win)** from the `swing_1_5d` LightGBM+isotonic bundle |
| scorecard | weighted components → composite score (each component: weight, score, present, reason) |
| verdict | **BUY / HOLD / SELL** + composite score + **confidence (HIGH/MED/LOW)** + **data completeness** |
| trade_plan | entry rule, ref close, stop, target 1 (1:1), target 2 (2:1), qty@₹1L, qty@1%‑risk, round‑trip cost %, hold ≤ N days |

> **➡️ NEXT STEP (explicitly deferred by the user):** the *definitive* list of
> analysis steps, their order, the exact indicators, weights, thresholds, any new
> data (e.g. options chain, sector breadth, peer‑relative strength, event
> proximity), and how the model's P(win) combines with the rule‑based scorecard —
> will be specified and appended here as **§6 (Analysis Method — detailed)**. The
> table above is the **baseline** we start the discussion from.

### Stage 4 — Live observability UI *(reuse + expand)*
The "see all step‑by‑step logs / step / status per stock" requirement is **already
substantially built** and must be reused and extended:
- **Step status** ([src/daily/status.py](src/daily/status.py)): per‑step state,
  live one‑liners, rolling log tail, and derived honesty states (`STALLED`,
  `STOPPED_UNEXPECTEDLY`) — atomically written to `reports/daily/pipeline_status.json`.
- **Per‑stock trace** ([src/daily/trace.py](src/daily/trace.py)): one structured
  JSONL event per item — every source hit (with URL/ticker), every symbol's panel
  build, **every stock's indicator values + calibrated score (or skip reason)**,
  the final ranking → `reports/daily/pipeline_trace.jsonl`.
- **Narrative** ([src/daily/story.py](src/daily/story.py)): the run as a numbered,
  human‑readable story ("why the top pick won", from the real numbers).
- **API/UI** ([src/api/daily_app.py](src/api/daily_app.py), port 8001): `/pipeline/status`,
  `/pipeline/trace`, `/pipeline/stream` (live NDJSON, event‑as‑it‑happens),
  `/pipeline/story`, `/ui/trace` (dashboard), `/ui` (analyzer), `/pipeline/run` +
  `/pipeline/stop` (job control), `/analyze/{symbol}`, `/daily/list`,
  `/daily/scoreboard`. UI pages: [src/api/ui/trace.html](src/api/ui/trace.html),
  [src/api/ui/analyze.html](src/api/ui/analyze.html).
- **Expansion required (§14):** a single **"morning run" landing page** that shows
  the pipeline steps, a per‑stock progress grid (queued → analysing → scored →
  ranked), and the final BUY list — consolidating the today‑separate trace and
  analyzer pages into one operator view.

### Stage 5 — BUY recommendations output *(reuse)*
Final step produces the ranked recommendations. See the [output spec in §9](#9-recommendation-output-specification).
Reused modules: [src/daily/screener.py](src/daily/screener.py) (score → rank →
top‑N with SHAP "why" + trade plan) and [src/daily/run_list.py](src/daily/run_list.py)
(tiering: daily **TOP_PICK** always; **STRONG_SIGNAL** only when calibrated
P(win) ≥ 0.80). Written to `reports/daily/lists/list_<date>.json` and served at
`/daily/list`.

---

## 6. Analysis Method — detailed  *(TO BE COMPLETED IN THE NEXT STEP)*

> Placeholder. This section will hold the agreed, definitive per‑stock analysis
> pipeline: exact stages, indicators, data inputs, scoring/weighting, the
> model‑vs‑rules combination, and the precise probability & confidence formula.
> Until then, the **baseline** is the existing `analyze()` scorecard in
> [src/analysis/engine.py](src/analysis/engine.py) (§5, Stage 3 table).

Open design threads to resolve here (seed list — expand in discussion):
- Which indicators are decisive for a 1–5 day BUY vs. context‑only?
- Does the final probability come from (a) the calibrated model P(win), (b) the
  rule‑based composite, or (c) a disclosed blend of both — and how is that blend
  validated?
- New data to add? (options chain / PCR, sector & market breadth, peer relative
  strength, earnings/event proximity, delivery‑spike detection.)
- Per‑stock pass/fail filters (e.g. skip if news‑blackout, illiquid today,
  circuit‑locked, upcoming earnings inside the hold window).

---

## 7. Data sources

All sources are **point‑in‑time (PIT)**: a decision made for today's open never
reads data it could not have had at that moment. Each is **traced** and **fails
soft**.

| Source | API / feed | Provides | Cadence | On failure |
|---|---|---|---|---|
| **Upstox** | market‑quote v2 + historical (**1‑yr read‑only Analytics Token** — no daily login) | Live pre‑open/intraday OHLC + LTP + volume; 1‑min bar history | intraday / on‑demand | Provisional bar skipped → fall back to last EOD; token check first |
| **NSE bhavcopy** | nsearchives.nseindia.com | EOD full‑market OHLCV, **delivery %**, F&O OI/basis, FII/DII | daily ~18:30 IST | Trailing‑week self‑heal; missing today ⇒ analyse last session |
| **Yahoo Finance** | yfinance | Global/overnight (S&P, Nasdaq, crude, gold, USD/INR, US10Y), **India VIX**, fundamentals | daily / on‑demand | Feature block → `_present=0`, degrade honestly |
| **GDELT** | DOC API v2 | 30‑day company news → sentiment (FinBERT or lexicon) | on‑demand (Analyzer) | Cached; throttled ⇒ show cached + label |
| **Macro/events** | curated CSVs | macro calendar, earnings dates, monsoon | daily | optional block |

> **Honesty contract for live Upstox (already coded):** an intraday snapshot's
> `close` is the *current* price, not the final 15:30 close; delivery %/OI/FII
> don't exist intraday. A pick from a live snapshot is a **PROVISIONAL preview**,
> flagged as such — never a final signal. It cannot create edge the model lacks.
> See [src/daily/live_snapshot.py](src/daily/live_snapshot.py) docstring.

> **Upstox auth — 1‑year Analytics Token (decision 2026‑08‑03):** the client uses a
> **read‑only Analytics Token** (Developer Apps → Analytics → Generate Token), which
> grants the market‑data GET APIs we actually use — **Market Quote** and **Historical
> Data** — **without a static IP and without daily login**, valid **1 year**. We never
> place orders, so read‑only is sufficient (and aligns with the no‑live‑orders posture
> in §10). Paste it into `.env` as `UPSTOX_ACCESS_TOKEN`; regenerate ~once a year — the
> daily token treadmill is gone. `auth_ping()` was moved off `/user/profile` (an
> Account API that would need a static IP) onto a Market Quote GET so the check passes
> under this token. See [src/intraday/data_feed.py](src/intraday/data_feed.py).

**Trustworthiness / redundancy requirement:** where feasible, cross‑check the
primary price source. NSE bhavcopy is the settlement‑grade truth for EOD; Upstox
is the live truth pre‑open; disagreements beyond a tolerance must be surfaced in
the trace, not hidden.

### 7A. Where all fetched & derived data is stored (the persistence layer)

**Answer to "where are we storing the data?":** everything lives on the **local
filesystem inside the repo** as **versioned files — there is no external database
today**. Formats: **Parquet** for time‑series/panels (columnar, fast), **CSV** for
reference tables, **JSON/JSONL** for run outputs/observability, **PNG** for charts.
Reproducibility is provided by **DVC** (data versioning) + **MLflow** (model &
experiment tracking). Secrets (Upstox token, API keys) live in **`.env` only** —
never in the data store or git.

| Layer | Path | Format | Written by |
|---|---|---|---|
| Raw NSE EOD (full market) + F&O | `data/bhavcopy/eq_<date>.parquet`, `fo_<date>.parquet` | Parquet | [src/intraday/bhavcopy.py](src/intraday/bhavcopy.py) |
| FII/DII flows | `data/bhavcopy/` · `data/reference/` | Parquet/CSV | [src/daily/fii_backfill.py](src/daily/fii_backfill.py) |
| **Per‑symbol daily panel (the main per‑stock store)** | `data/daily/<SYMBOL>.parquet` | Parquet | [src/daily/panel.py](src/daily/panel.py) |
| Full‑market panel cache | `data/processed/v6_panel.parquet` | Parquet | panel / v6 |
| Corporate actions | `data/reference/corporate_actions.csv` · `data/processed/` | CSV/Parquet | [src/intraday/corporate_actions.py](src/intraday/corporate_actions.py) |
| Global / overnight / commodity / FX | `data/global/` | Parquet | [src/daily/global_data.py](src/daily/global_data.py) |
| Regime day‑files (India VIX, global tone) | `data/regime/global/<date>.parquet` | Parquet | [src/intraday/regime_data.py](src/intraday/regime_data.py) |
| News (RSS + GDELT + FinBERT) | `data/news/` | Parquet/JSON | [src/intraday/news_regime.py](src/intraday/news_regime.py) |
| Analyzer news cache | `reports/analysis/news_cache/<sym>_<date>.json` | JSON | [src/analysis/engine.py](src/analysis/engine.py) |
| 1‑min Upstox bars | `data/bars/` | Parquet | [src/intraday/data_feed.py](src/intraday/data_feed.py) |
| Live Upstox snapshot | provisional row injected into `data/daily/<SYMBOL>.parquet` | Parquet | [src/daily/live_snapshot.py](src/daily/live_snapshot.py) |
| **Market cap / shares outstanding** *(NEW — for the ≤₹50,000 Cr filter)* | `data/reference/market_cap.csv` | CSV | *to build (§5 Stage 2)* |
| Reference (universe, calendars, holidays, events, monsoon) | `data/reference/*.csv` | CSV | [scripts/build_daily_universe.py](scripts/build_daily_universe.py) etc. |
| Sentiment forward archive | `data/sentiment/` | Parquet | `src/daily/sentiment/` |
| Model bundles (price + calibrator) | `models/daily/` | joblib/pickle | [src/daily/trainer.py](src/daily/trainer.py) |
| Experiment / registry | `mlruns/`, `mlartifacts/` | MLflow | trainer |
| Data versioning | `.dvc/` (+ cache) | DVC | dvc |
| **Daily watchlist / recommendations** | `reports/daily/lists/list_<date>.json` | JSON | [src/daily/run_list.py](src/daily/run_list.py) |
| Pipeline observability (status/trace/history) | `reports/daily/pipeline_status.json`, `pipeline_trace.jsonl`, `pipeline_runs.jsonl` | JSON/JSONL | [src/daily/status.py](src/daily/status.py), [src/daily/trace.py](src/daily/trace.py) |
| Honesty scoreboard / run registry | `reports/daily/run_registry.csv` | CSV | [src/daily/evaluate.py](src/daily/evaluate.py) |
| Charts | `reports/daily/charts/`, `reports/analysis/*.png` | PNG | [src/daily/charts.py](src/daily/charts.py) |

Paths are configured in [config/config_daily.yaml](config/config_daily.yaml)
(`data:` and `paths:` blocks). PIT discipline is structural: the daily panel is
rebuilt each run over a trailing window and a signal decided at close(D) never sees D+1.

> **Open decision (§14):** keep the **file‑based Parquet + DVC** store (simple,
> reproducible, right‑sized for ~2,400 symbols × daily bars) — **recommended** — or
> introduce a **database** (DuckDB as a local query layer over the *same* Parquet /
> SQLite / Postgres+TimescaleDB) if we later need ad‑hoc queries, a serving cache, or
> multi‑machine access. DuckDB can be added over the existing Parquet with **no
> migration**, so we are not locked in.

---

## 8. UI / Observability requirements

The UI is a first‑class deliverable ("so we can understand what is happening at
the background"). Requirements:

1. **Auto‑refreshing run view** — shows every pipeline step with state
   (pending / running / done / failed), elapsed seconds, and the live one‑liner.
2. **Per‑stock visibility** — for each selected stock the user can see: fetched?
   analysed? indicator values, block presence, the calibrated score (or the exact
   skip reason). Backed by the per‑item trace.
3. **Live streaming** — events appear *as they happen* (not just on refresh);
   `/pipeline/stream` (NDJSON) already does this.
4. **Plain‑English narrative** — "what happened / what it means / what's next",
   including *why* the top pick ranked #1 (from the real numbers).
5. **Honesty surfaces** — `STALLED` / `STOPPED_UNEXPECTEDLY` states; the
   measured‑edge note on every probability; the `actionable` flag; provisional‑live
   banner when applicable.
6. **Final BUY list panel** — the ranked recommendations (see §9), with drill‑down
   into the single‑stock Analyzer.
7. **Job control** — start/stop the run from the UI (already: `/pipeline/run`,
   `/pipeline/stop`), guarded so two runs can't overlap.
8. **Security** — every data endpoint requires `X‑API‑Key` (`DAILY_API_KEY`); the
   API refuses to start without it. No model‑mutating or order endpoint exists.

---

## 9. Recommendation output specification

The final BUY list. Each recommendation object carries **complete probability
details and a confidence value** (the user's explicit ask), plus a pre‑committed
trade plan and honesty metadata:

| Field | Meaning | Source |
|---|---|---|
| `rank` | 1 = the day's top pick | screener |
| `symbol`, `direction` | e.g. RELIANCE, LONG | screener |
| `prob` | **calibrated P(win)** ∈ [0,1] (isotonic‑calibrated LightGBM) | model bundle |
| `confidence` | **HIGH / MEDIUM / LOW** — from composite strength × evidence completeness | analyzer verdict |
| `data_completeness` | fraction of evidence sources present | analyzer verdict |
| `tier` | `TOP_PICK` (always, rank #1) or `STRONG_SIGNAL` (P ≥ 0.80, rare) | run_list |
| `why` | top‑K SHAP drivers, human‑phrased | screener |
| `actionable` | true only after the after‑cost backtest gate (Milestone 2) passes | backtest verdict |
| **trade plan** | ref close, entry rule (next open), stop, target 1 (1:1), target 2 (2:1), max hold days, qty@₹1L, qty@1%‑risk, round‑trip cost % | screener / engine |
| `confidence_note` | honest one‑liner (relative‑best vs. cleared‑the‑bar; measured‑edge caveat) | run_list |
| `disclaimer` | research aid, not advice; no proven edge; verify before acting | engine |

**Hard rules:**
- **Every** recommendation shows its `prob` **and** its `confidence` **and** the
  honesty note. No naked "BUY".
- If no pick qualifies, say so — **never** silently substitute a weak pick as if
  it were strong (already enforced in `run_list.top_pick_block`).
- Track the recommendations' real outcomes over time
  ([src/daily/evaluate.py](src/daily/evaluate.py) scoreboard) and expose it — the
  probability's credibility is earned by the running record, not asserted.

---

## 10. Compliance & guardrails (non‑negotiable)

- **No live order placement** in this phase. There is no order/execution endpoint
  and none will be added until the full gate chain + broker approval exists. See
  [COMPLIANCE.md](COMPLIANCE.md).
- **SEBI algo posture** preserved; the system is research/decision‑support.
- **Every output is disclaimed** as a research aid, not investment advice, with the
  measured no‑edge record explicit.
- **Provisional‑live labelling** is mandatory whenever a recommendation used an
  intraday snapshot instead of a final close.
- **Secrets**: Upstox token + API keys in `.env` only (never committed). The
  recommended Upstox credential is a **1‑year read‑only Analytics Token** (no daily
  login; regenerate yearly) — see §7. A standard OAuth token still expires daily
  (~03:30 IST); if that is used instead, the pre‑market run verifies it first
  ([RUNBOOK.md](RUNBOOK.md)).

---

## 11. Milestones — phased delivery

> **Sequencing per the 2026‑07‑29 decision:** the **predictive edge is the primary
> near‑term goal**, so the plan drives toward the M‑Edge gate early. M0/M1 are not
> "ship and stop" — they are the *substrate the backtest needs* (you cannot
> pre‑register a gate on an analysis pipeline that doesn't exist yet). They are
> built quickly, then the edge is put to the pre‑registered test.

| Milestone | Deliverable | Gate to pass |
|---|---|---|
| **M0 — Consolidate** | One pre‑market run (EOD base + labelled live preview) + one unified UI landing page over the *existing* pipeline; auto‑schedule pre‑market. | Runs green end‑to‑end pre‑market on a schedule; UI shows per‑stock live status; **green Playwright E2E smoke test (§12)**. |
| **M1 — Analysis method (§6)** | The agreed detailed per‑stock analysis pipeline implemented; probability + confidence formula finalised & documented. | Deterministic, unit‑tested; trace shows every stage per stock. |
| **★ M‑Edge (PRIMARY GOAL)** | **Pre‑register** the acceptance gate *before* looking at results, then run a **fresh walk‑forward backtest** of the §6 logic. | `gate.milestone1` ([config_daily.yaml](config/config_daily.yaml)): OOS AUC > 0.52, fold‑consistent, calibrated (≤0.05 gap), top‑N lift ≥ 1.05×. **PASS ⇒ edge exists; KILL ⇒ honest stop / re‑contract.** |
| **M‑Actionable** | After‑cost validation. | `gate.milestone2`: net‑of‑cost expectancy ≥ 0, also at 2× cost stress → flips `actionable=true`. |
| **M‑Paper / Capital** | Forward paper run, then (only after broker approval) smallest‑size live. | Paper win‑rate gap small + positive expectancy; COMPLIANCE checklist clean. |

> M‑Edge is the headline. Because the prior premise KILLed here before, the gate is
> **frozen before measurement** so the answer is trustworthy either way. A KILL is a
> cheap, valid result (₹0 capital); a *false* PASS shipped to real money is the only
> unacceptable outcome. While the edge is being proven, M0/M1 still deliver honest
> value (automation + transparency + disciplined plans).

---

## 12. Testing & quality assurance (incl. Playwright UI automation)

Testing is a **first‑class, gating** deliverable — every milestone in §11 exits only
when its tests are green. Two layers, both automated and re‑runnable by CI **and by
the assistant (Claude)**:

### 12.1 Layer A — Unit / integration (existing `pytest`, keep & extend)
The repo already has a deterministic pytest suite (synthetic fixtures, **no
network**): [tests/](tests/) — e.g. `test_analysis.py` (scorecard/verdict/trade‑plan),
`test_screener.py`, `test_features.py` (`test_no_lookahead_property`),
`test_pipeline_status.py`, `test_api.py`, plus the labeler/universe/cost tests.
This stays the backbone for **pure logic + PIT discipline** and extends to cover the
new §6 analysis method and the probability/confidence formula.
*Gate:* pytest green + coverage ≥ 85% (the existing "Gate 0 — Engineering").

### 12.2 Layer B — End‑to‑end UI automation (**NEW — Playwright**)
Playwright drives a **real browser** against the running daily UI/API
([src/api/daily_app.py](src/api/daily_app.py), port 8001;
[src/api/ui/analyze.html](src/api/ui/analyze.html), [src/api/ui/trace.html](src/api/ui/trace.html))
to verify the observable pipeline and the recommendations **the way the user sees
them**. This is the "automated testing using the plugin" the requirement asks for.

**E2E scenarios (the minimum suite):**
1. **UI loads** — `/ui` renders; symbol autocomplete populates from `/symbols`.
2. **Run lifecycle** — trigger `/pipeline/run` (test mode); assert every step goes
   `pending → running → done`; assert **no** `STALLED` / `STOPPED_UNEXPECTEDLY` on success.
3. **Per‑stock trace populates** — `/ui/trace` shows each stock's indicator values +
   calibrated score; skipped stocks render **with their reason** (no silent drop).
4. **Live stream** — connect to `/pipeline/stream`; assert step/trace/state events
   arrive **as they happen** during a run.
5. **Recommendation list** — the final BUY list shows each pick with **probability
   AND confidence AND the honesty note**; `TOP_PICK` always present; `STRONG_SIGNAL`
   only when P(win) ≥ 0.80; "no picks" is shown honestly (**no** weak‑pick substitution).
6. **Analyzer drill‑down** — `/analyze/{symbol}` renders scorecard + verdict + trade
   plan + chart.
7. **Honesty guardrails (regression lock)** — the disclaimer is present; the
   provisional‑live banner appears on a `--live` run; **no order/execution control
   exists** anywhere in the UI.
8. **Auth** — any data endpoint without `X‑API‑Key` returns **401**.

**Determinism (hard requirement — tests must be hermetic):**
- E2E tests must **not** depend on live market data, a valid Upstox token, or GDELT.
  Run against a **frozen fixture session** (the pipeline already "falls back to the
  latest session on disk"; tests seed a tiny fixture universe + panel).
- Network sources (yfinance / GDELT / Upstox) are **stubbed or disabled**
  (`_present=0`), so runs are fast and repeatable.
- The API is booted on an **ephemeral port** with a **test `DAILY_API_KEY`** by the
  test harness (Playwright `webServer` config, or a pytest fixture that launches uvicorn).

### 12.3 How the assistant runs these tests — two mechanisms (status is honest)
| Mechanism | What it is | Status in this environment |
|---|---|---|
| **Committed Playwright suite (primary)** | `tests/e2e/*.spec.ts` run via **`npx playwright test`** through Bash. Deterministic, CI‑able, re‑runnable by anyone. **No MCP plugin required.** | ✅ **Available now** — Playwright **v1.62.0 is already installed**. |
| **Playwright MCP ("plugin", optional)** | A Playwright **MCP server** giving Claude live browser tools (navigate/click/snapshot) to explore the UI, reproduce a bug, or author new specs interactively. | ⚠️ **Not connected in this session** (only `gemini`, `zen` MCPs are). Enable with `claude mcp add playwright npx @playwright/mcp@latest` then authorize via `/mcp`. |

> **Recommendation:** the **committed suite is the backbone** (hermetic, versioned,
> CI‑gated, and runnable by Claude via Bash today). The MCP plugin is a convenience
> layer for interactive exploration and spec authoring — nice to have, not required
> for the automated gate.

### 12.4 CI & layout
- **CI job** (e.g. GitHub Actions): install deps → boot API in test mode → run
  `pytest` **and** `npx playwright test` → on failure upload the Playwright **HTML
  report + trace viewer artifacts** (per‑step DOM snapshots/screenshots) for debugging.
- **Proposed layout:** [tests/](tests/) (existing Python) · `tests/e2e/`
  (new: `playwright.config.ts`, `*.spec.ts`, fixtures, the test‑mode `webServer`) ·
  a `package.json` for the JS toolchain (or reuse the installed global Playwright).
- The E2E specs double as **living documentation of the honesty contract** — a
  regression guard so the disclaimer, confidence labels, and "no silent substitution"
  behaviour can never be quietly stripped.

*Gate (extends Gate 0 to the UI):* **M0 exits only with a green Playwright E2E smoke
test** (run lifecycle + list renders + auth 401); the suite grows one scenario group
per milestone.

---

## 13. Lineage & cleanup (what this replaces)

Per [ACTION_ITEM.md](ACTION_ITEM.md), once this requirement is approved a
**keep / retire list** is produced and confirmed **before any deletion** (git
history preserved throughout).

**Proposed KEEP / REUSE (the working pipeline this plan is built on):**
- `src/daily/` (pipeline, screener, run_list, status, trace, story, live_snapshot,
  features, panel, trainer, model, evaluate, charts, global_data, macro, fii_backfill)
- `src/analysis/engine.py`, `src/api/daily_app.py` + `src/api/ui/*`
- `src/intraday/` modules still used by the daily flow (bhavcopy, data_feed,
  corporate_actions, regime_data, news_regime, costs, risk)
- `src/v6/universe.py` + `scripts/build_daily_universe.py`
- MLOps scaffolding: MLflow (`mlruns/`, `models/`), DVC (`.dvc/`), `config/config_daily.yaml`
- Ops docs: [RUNBOOK.md](RUNBOOK.md), [COMPLIANCE.md](COMPLIANCE.md)

**ARCHIVED (done 2026‑07‑29 — moved to [archive/](archive/), *not* deleted):**
- Old plan docs → `archive/`: `PLAN.md`, `PLAN_v3.md`, `v3_supplementry.md`,
  `PLAN_v4.md`, `PLAN_v4_STATUS.md`, `PLAN_v6.md`, `SYSTEM_EXPLAINED.md`,
  `STATUS_2026‑06‑03.md`. Each now carries an **EXPIRED** banner at the top
  pointing back to this plan. Moved with `git mv` (history preserved); inbound
  links in kept docs (README, SOLUTION_FLOW) were repointed to `archive/`.

**Still proposed for RETIRE (pending approval — not yet touched):**
- v6 Phase‑0 one‑shot probe scripts (`scripts/v6_*.py`) and their reports once no
  longer referenced.
- The intraday *trading* system's KILLed decision path, if not feeding the daily flow.

> **Deletions** are still finalised **with the user** — nothing is *removed*
> unilaterally. Archiving above is non‑destructive (files remain in `archive/` and
> in git history) and reversible.

---

## 14. Decisions & open questions

### 14.1 Decision log
| Date | Decision | Outcome |
|---|---|---|
| 2026‑07‑29 | **Product goal** | **Pursue a real predictive edge as the primary near‑term goal** — via a fresh, pre‑registered backtest gate (not threshold‑tuning the old KILL). See §2, §11 M‑Edge. |
| 2026‑07‑29 | **Data basis** | **Both** — yesterday's official EOD close as the reliable base *and* a labelled `PROVISIONAL` live Upstox pre‑open preview, shown side by side. See §5 Stage 0, §7. |
| 2026‑07‑29 | **"High volume" definition** | Partly resolved — see the ≤₹50,000 Cr cap below; exact liquidity metric still open (Q1). |
| 2026‑07‑29 | **Automated testing** | **Adopt Playwright for automated E2E/UI testing** alongside the existing `pytest` suite. The committed `npx playwright test` suite is the CI‑gated backbone (works today — v1.62.0 installed); the Playwright **MCP plugin** is an optional interactive layer (not yet connected). See §12. |
| **2026‑07‑30** | **Market‑cap ceiling** | **Hard eligibility filter: only stocks with market cap ≤ ₹50,000 Cr.** Applied before liquidity ranking ⇒ mid‑cap‑and‑below focus (excludes mega‑caps). **Config‑driven** — `universe.max_market_cap_cr` in [config/config_daily.yaml](config/config_daily.yaml) (added). Needs a market‑cap reference + PIT handling for the backtest. See §5 Stage 2. |
| **2026‑07‑30** | **Data storage** | **Confirmed: local‑filesystem, file‑based store** — Parquet (panels) / CSV (reference) / JSON‑JSONL (outputs) under `data/`, `models/`, `reports/`; versioned by **DVC + MLflow**; no external DB today. Full map documented in **§7A**. |
| **2026‑08‑03** | **Upstox auth** | **Use the 1‑year read‑only Analytics Token** (market‑data GET APIs, no static IP, no daily login) — removes the daily‑token dependency. Code: `auth_ping()` switched from `/user/profile` to a Market Quote GET so it works under this token ([src/intraday/data_feed.py](src/intraday/data_feed.py)). See §7. |

### 14.2 Still open (for the next step)
1. **Liquidity metric for the (≤₹50,000 Cr) universe** — ₹ *turnover* (current) vs.
   *unusual‑volume‑today* vs. raw *share‑volume*; and universe size (top‑N). *(The
   market‑cap cap itself is now decided; this is only the ranking metric within it.)*
2. **Analysis method (§6)** — the whole per‑stock analysis detail (deferred by you).
3. **Probability source** — model P(win) only, rule‑composite only, or a disclosed,
   validated blend (ties to §6 and directly to the M‑Edge backtest design).
4. **Run timing & scheduler** — exact pre‑market clock (e.g. 08:15 IST?) and the
   mechanism (cron/launchd vs. `/pipeline/run` trigger vs. repo `schedule`).
5. **Notifications** — desktop only (current `osascript`), or add email/Slack/Telegram.
6. **UI consolidation** — one new landing page, or extend the existing `/ui` + `/ui/trace`.

---

## 15. Acceptance criteria for THIS document (Step 1)

- [x] Captures the five functional requirements (fetch → high‑volume → per‑stock
      analysis → step‑by‑step UI → BUY list with probability + confidence).
- [x] Grounds every stage in the **existing** pipeline with a concrete reuse map.
- [x] Preserves the inherited honesty/compliance discipline (KILL, no‑edge, no
      live orders).
- [x] Defers the analysis method to §6 (next step), as requested.
- [x] Defines the automated testing approach (§12 — `pytest` unit + Playwright E2E,
      incl. how the assistant runs it).
- [x] Ties into the [ACTION_ITEM.md](ACTION_ITEM.md) cleanup sequencing.
- [ ] **User review & sign‑off** → then proceed to §6 (analysis method) + M0.

---

*References: [SOLUTION_FLOW.md](SOLUTION_FLOW.md) (end‑to‑end flow + KILL history) ·
[README.md](README.md) (stack & lineage) · [RUNBOOK.md](RUNBOOK.md) (ops) ·
[COMPLIANCE.md](COMPLIANCE.md) (SEBI posture) · [config/config_daily.yaml](config/config_daily.yaml)
(all tunable knobs). If this document and the code ever disagree, reconcile
explicitly — do not let the doc drift.*
