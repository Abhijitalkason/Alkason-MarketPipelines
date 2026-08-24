> ⚠️ **EXPIRED — this plan is no longer active and has been superseded. Please check the current plan: [Aug26Plan.md](../Aug26Plan.md).**

# PLAN v4 — 80% Fired-Signal Win Rate + Pre-Open Regime/News Channel (NSE Intraday)

> **Project:** Intraday (1–2 hour window) selective trading-signal system for NSE large caps
> **Goal contract:** ≥ **80% win rate on emitted signals** AND positive **post-cost expectancy per trade**. Win rate is engineered by barrier geometry; expectancy is delivered by model edge. Both must hold.
> **Trading window:** Entries 09:30–11:00 IST · time barrier entry+2h (grid-confirmed) · hard square-off 14:45 · always flat overnight
> **Supersedes:** PLAN_v3.md (targets and evaluation contract only — the v3 architecture and module set are retained). v3_supplementry.md remains the implementation-detail reference where not contradicted here.
> **Plan version:** 2026-07-02

---

## 1. Context — Why v4 Exists (the honest post-mortem of "0.50 accuracy")

The reported failure — *"we executed PLAN_v3 and achieved only 0.50 accuracy, a coin flip"* — is a **measurement error, not a system verdict**:

1. The 0.50 came from `scripts/prediction_accuracy.py`: raw directional accuracy at p ≥ 0.5 over **all** out-of-fold rows (100% coverage). PLAN_v3 §12.3 explicitly banned accuracy as a metric and §2.3 predicted the model would look like a coin flip at full coverage. A selective system's model is *supposed* to be near-random on the rows it would never trade.
2. The system's actual contract — win rate on the few **fired** signals (2–6/day past the conformal gate) plus post-cost expectancy — was **never measured**. As of 2026-07-02: no walk-forward backtest artifacts exist in `reports/`, the run registry is absent, the ACI gate state has never adapted (one init entry), and Gate 1 (data) is marked FAILED.
3. Gate 1 failed for mechanical reasons: it ran 2026-06-19 **before** the bar backfill landed (2026-06-22), and its `min_names=180` threshold is unreachable against a 46-symbol bar store / 51-name starter universe.
4. No news/sentiment code exists on `main`. The "intraday categories" feature work does not change any of the above.

**Conclusion:** the pipeline architecture is not wrong; the goal metric and the evaluation were. v4 keeps the entire v3 skeleton (bar store, triple-barrier labeler, morning screen, LightGBM price model, blend + isotonic calibration, ACI conformal gate, purged walk-forward backtester, risk layer, paper runner) and changes exactly four things: the geometry/gate targets, one new pre-open regime feature group, one fire-time regime veto, and the evaluation contract.

### 1.1 Decisions locked in (2026-07-02)

| Decision | Choice |
|---|---|
| Success metric | ≥ 80% win rate on **emitted signals** + positive post-cost expectancy. NOT per-stock daily direction accuracy — that is unachievable at a 1–2h horizon by any known method, with or without news. |
| News usage | **Pre-09:30 regime filter**: overnight global markets + macro/news sentiment → market-direction bias features and a trade/no-trade veto. NOT live intra-window news (priced in within seconds; retail feeds lag the move). |
| Path forward | Redesign now, assume current price/flow features carry no edge. Acceptance is a proper purged walk-forward backtest on fired signals. Full-coverage accuracy is retired as a decision metric. |

### 1.2 The geometry math, corrected

Required model edge is `δ = c / W`, where `W = (a+b)·ATR` is **total** barrier width and `c` is round-trip cost. Relaxing the win-rate floor from 89% → 80% helps **only if the freed slack is spent on width**:

| Geometry (a / b ×ATR) | Floor b/(a+b) | Width W (ATR≈0.7%) | Required δ (c≈0.09%) |
|---|---|---|---|
| 0.5 / 2.0 (naive pick) | 80.0% | 2.5 ATR ≈ 1.75% | ~5.1 pts — **worse than v3** |
| 0.4 / 3.2 (v3 frozen) | 88.9% | 3.6 ATR ≈ 2.52% | ~3.6 pts |
| **0.75 / 3.0** | **80.0%** | 3.75 ATR ≈ 2.6% | **~3.4 pts** |
| **1.0 / 4.0** | **80.0%** | 5.0 ATR ≈ 3.5% | **~2.6 pts** |

The grid (§3) therefore favors wide-width cells at a ~0.80 floor. Caveat measured on real paths: wide targets + short time barrier increase time-exits (~50/50 splits) which dilute realized win rate below the pure floor — the geometry study on 1-min paths decides, not the table.

---

## 2. Verified Repo Facts That Shape the Work

- **Gate 1 failure evidence:** `reports/v3/gates/gate1_20260619_110101.json` (`names_ok: 0`). Stale run + structurally unreachable `min_names=180` hardcoded in `src/intraday/gates.py`.
- **Latent labeler bug:** `src/intraday/labeler.py:81-82` reads `time_barrier_hours` and `squareoff` from global config, ignoring the per-call `geo` override dict (the `entry_window` override at lines 77-80 is the correct pattern). Any geometry grid that sweeps the time barrier silently uses the config value. **Must fix before the grid runs.**
- **No global-market data exists** — no `data/daily/`, no SPX/VIX/USDINR parquets anywhere.
- **Legacy `data/news/*.json`** (416 files, 16 symbols, 2026-06-04→06-22) lack per-headline published timestamps → not point-in-time replayable. Freeze as-is; do not migrate; format precedent only.
- **The regime veto must NOT use `is_blackout`/events.csv:** `backtester.build_dataset` (lines ~58, 68) drops blacked-out days from the dataset entirely. Vetoed days must remain labeled data for ablation and coverage accounting → the veto applies at **fire time**.
- **Cached dataset insufficient:** `data/processed/v3_dataset.parquet` (20,878 rows, 16 symbols, from 2024-12) cannot support 18m train + 6×3m OOS folds. Rebuild over the full bar span 2023-06-23 → present, all 46 symbols.
- **Config plumbing:** `CONFIG_PATH` is hardcoded to `config/config_v3.yaml` (`src/intraday/__init__.py:19`) and tests monkeypatch it. Keep the filename; edit keys in place (provenance preserved via config hash in the run registry); bump `paths.reports → reports/v4`, `paths.state → models/v4` so v3 artifacts stay untouched. `tests/test_config.py` fails on dead keys → every new key ships with its consumer in the same milestone.

---

## 3. Milestone M0 — Gate 1 Fix (½ day)

- `src/intraday/gates.py` `gate1()`: replace hardcoded `min_names=180` with new config key `gates.min_data_names_frac: 0.90`, evaluated relative to the point-in-time universe size.
- Re-run Gate 1 — the backfill landed 2026-06-22; expected to pass at ~46/51.
- Record the universe decision: v4 stays at the current ~50-name universe; NSE-200 expansion is out of scope.

## 4. Milestone M1 — Geometry Retarget (2–3 days)

**Files:** `src/intraday/labeler.py`, `src/intraday/geometry_study.py`, `config/config_v3.yaml`.

1. Fix the labeler geo-override bug (§2) so `geo["time_barrier_hours"]` / `geo["squareoff"]` are honored with config fallback.
2. `geometry_study.py`: replace the hardcoded 0.86-baseline / δ≤4pt thresholds (lines ~96-97, 130) with config keys:
   ```yaml
   geometry_study:
     min_baseline: 0.76          # floor must sit BELOW the 0.80 target so the model adds the edge
     max_baseline: 0.84          # too-high floor reproduces the v3 trap (required δ unreachable)
     max_required_delta_pts: 3.0
   ```
3. Extend the grid: `A_GRID = [0.4, 0.5, 0.6, 0.75, 1.0]`, `B_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]`, secondary axes `time_barrier_hours ∈ {2, 3}` and `atr_window_minutes ∈ {60, 120}` (thread an ATR-window parameter through `bars.atr_2h` if needed). Selection favors wide W at ~0.80 floor; **expected pick ≈ a=0.75, b=3.0, time barrier 2h** (matches the stated 1–2h trade duration) — but the grid on real 1-min paths decides.
4. Run `python main.py --mode geometry-study` on **fold-0 train days only** (existing `TrainWindow` discipline). Freeze the winner into config:
   ```yaml
   geometry: { target_atr: <grid>, stop_atr: <grid>, time_barrier_hours: <grid>, ... }
   gate:     { fire_threshold_init: 0.84, target_win_rate: 0.80 }        # α = 0.20
   gates:    { min_win_rate: 0.80, leakage_alarm_win_rate: 0.90,
               leakage_alarm_coverage: 0.10, calibration_max_gap: 0.03 }
   ```
   τ₀ = 0.84 sits a few points above the 0.80 target so ACI relaxes toward equilibrium rather than starting under-covered. Leakage alarm at 0.90/0.10 keeps the same ~+8–10pt "too good" margin over the floor that v3 used.
5. `paths.reports → reports/v4`, `paths.state → models/v4`.

**Tests:** `test_labeler.py` (override honored), `test_config.py` (new keys consumed).

## 5. Milestone M2 — Global/Macro Regime Data Layer (2–3 days, parallel with M1)

**New:** `src/intraday/regime_data.py`, `scripts/backfill_regime.py`, `--mode regime-capture` in `main.py`.

| Source | Instruments | Access | Backfillable? |
|---|---|---|---|
| US prior close | ^GSPC, ^IXIC | yfinance (add `yfinance>=0.2` to requirements) | Yes — US close lands ~02:00 IST, strictly pre-09:15 |
| Asia prior close | ^N225, ^HSI | yfinance | Yes (prior full session) |
| Asia same-morning snapshot | ^N225, ^HSI partial session | yfinance live | **Forward-only** |
| FX / commodities | USDINR=X, BZ=F (Brent) | yfinance | Yes |
| India | ^NSEI prior close, ^INDIAVIX | yfinance; NSE archive fallback for VIX (reuse the `bhavcopy.py` NSE-session pattern) | Yes |
| GIFT Nifty pre-open gap | NSE-IX | best-effort scrape, behind `_present` flag | **Forward-only**; if never obtained, feature stays `present=0` |

- **Storage:** `data/regime/global/YYYY-MM-DD.parquet`, long format `[date, instrument, field, value, source, capture_ts]`. Reads go through the existing `point_in_time()` helper (`src/intraday/__init__.py:101`): prior-day EOD content dates pass regardless of capture time; same-day snapshots require `capture_ts ≤ 09:30`.
- **Backfill:** `scripts/backfill_regime.py` fills 2023-06 → now for all prior-day-close instruments.
- **Capture:** `--mode regime-capture` runs 08:45–09:10 IST via cron and is also invoked from the paper runner's pre-market setup. **Start the daily capture immediately** so forward-only features (Asia snapshot, GIFT gap) accrue from day one.

**Tests:** `test_regime_data.py` — PIT filtering (mutating data captured after 09:30 does not change the row).

## 6. Milestone M3 — Features v4 + Decision Backtests (3–4 days) — THE DECISION POINT

**Files:** `src/intraday/features.py`, `src/intraday/backtester.py`, `src/intraday/trainer.py`.

### 6.1 Feature schema v4.0

New group mirroring the flow-channel pattern (NaN → 0 only alongside a `_present=0` flag; group-level real-data share logged):

```
REGIME_FEATURES = [
  spx_ret_1d, ndx_ret_1d, asia_ret_1d, usdinr_ret_1d, crude_ret_1d,
  india_vix_prev, india_vix_chg_5d, gift_gap_pct,
  news_sent_market, news_sent_sector, news_sent_stock, news_count_stock,
  regime_dir_agree,                # candidate direction × sign(composite risk score)
]
REGIME_PRESENT = [f + "_present" for f in REGIME_FEATURES]
FEATURE_ORDER  = PRICE + FLOW + FLOW_PRESENT + REGIME + REGIME_PRESENT + ["direction"]
SCHEMA_VERSION = "v4.0"
```

- New `_regime_row(symbol, sector, day)` mirroring `_flow_row` (features.py:98); market-level values cached per day (constant across the ~12 screened symbols — do not re-read parquets 12×).
- **Routing:** all regime features go into the existing LightGBM `PriceModel`. No third model — the two-channel blend/ACI machinery stays; `flow_model_active` staging remains the pattern if per-regime evaluation later demands separation.
- **Screener untouched:** the direction hint stays `sign(gap + f15_ret)`; regime direction enters as the `regime_dir_agree` feature and the M4 veto. Changing screener ranking would confound the A/B ablation.
- Schema bump invalidates `models/v3` — retraining is expected and required.

### 6.2 Overfit guard — day-level clustering

Regime features are **constant per day** → their effective sample is ~440 trading days, not 20k+ rows, and all fired trades on a day share the regime (clustered outcomes). Consequences, enforced in code:

- The fired-win-rate confidence interval vs the geometric floor is computed **day-clustered** (block bootstrap over days, or day-level mean win rate) — never trade-i.i.d.
- The regime group is capped at the 13 features above; LightGBM regularization unchanged.

### 6.3 Decision runs (each auto-registers in `reports/v4/run_registry.csv`)

| Run | Setup | Purpose |
|---|---|---|
| **A** | v4 geometry, all regime features `present=0` | Pure ablation baseline (price features alone) |
| **B** | v4 geometry + backfilled global/macro features (news `present=0`) | **The decision run** |
| **D** | Run B restricted to the user's 16-symbol pre-decided list | Quantifies the coverage/WR cost of the fixed list vs the 46-name screen (informational, not a gate) |

Backtester additions: `regime_bucket` (day risk-score tercile) in the fairness tables; the day-clustered CI-vs-floor check in `_finalize`.

### 6.4 Acceptance (Gate 3, v4) — Run B, base AND 2× slippage stress

- Fired win rate ≥ 0.80
- Post-cost expectancy ≥ +0.05%/trade
- ≥ 1 signal/day
- Calibration gap ≤ 0.03
- No fold with > 0.90 win rate at > 0.10 coverage (leakage alarm)
- No subsidy breach (per-symbol/sector fairness tables)
- Fired-WR 95% **day-clustered** CI lower bound above the geometric floor

### 6.5 Kill criterion (pre-registered — no tuning past this point)

If Run B's fired win rate is statistically indistinguishable from the geometric floor (floor inside the 95% day-clustered CI) **AND** expectancy ≤ 0 — after the regime features — the finding is: **"no exploitable edge at the 09:30–11:00 horizon with this information set."** Response: stop intraday model work, keep the data layers (they compound), and redirect the news/regime channel to a **daily/swing-horizon system** (a new plan, not a v4 patch). The run registry makes tune-until-green visible by construction.

## 7. Milestone M4 — Regime Veto (1–2 days, conditional on M3 passing)

**New:** `src/intraday/regime.py`.

```python
day_regime(day) -> RegimeVerdict     # {risk_score, bias: -1|0|+1, no_trade: bool, present: bool}
veto(day, direction) -> (bool, str)  # True = suppress signal
```

- `risk_score` = config-weighted composite of the global features. `no_trade` when `abs(risk_score) ≥ regime.no_trade_abs_score` and the regime is conflicted/chaotic (e.g. VIX spike + risk-off). Direction veto when `sign(direction) == -sign(bias)` and `abs(risk_score) ≥ regime.direction_veto_threshold`. Missing data → `present=False` → **no veto** (fail-open, logged) — consistent with the `_present` philosophy.
- Config: `regime: { active: false, no_trade_abs_score: 0.75, direction_veto_threshold: 0.50 }` (staged flag, same pattern as `gate.flow_model_active`).
- **Plug points (fire time, never `is_blackout`):**
  - `backtester.run_backtest`: after `mask = gate.fire(...)` (~line 282), `mask &= ~veto(day, direction)` when `regime.active`; a `no_trade` day zeroes the mask but stays in the coverage denominator and dataset.
  - `paper_runner`: day-level no-trade check next to the existing `is_blackout` call (~line 242), journaled as `regime_no_trade`; per-candidate direction veto immediately before `gate.fire` (~line 120), journaled as a `gate_eval` skip reason. Every vetoed signal records its counterfactual label once mature — the data that proves or disproves the veto.
  - `RiskManager.can_open` untouched (position plumbing, not signal selection).
- **Run C:** Run B + `regime.active=true`. The veto must not be *needed* to pass Gate 3 — only demonstrated not to hurt.

## 8. Milestone M5 — News Sentiment Channel (starts after M2, matures over ~5 weeks, forward-only)

**New:** `src/intraday/news_regime.py`, `data/reference/symbol_aliases.csv`.

- **Sources:** RSS (ET Markets, Moneycontrol, Business Standard, Reuters India, LiveMint) + GDELT DOC 2.0 API as a secondary net. Pulls at ~08:00 and ~09:10 IST; only headlines with `published_ts < 09:15` are usable for the day.
- **Storage:** `data/regime/news/YYYY-MM-DD.parquet`: `[published_ts, capture_ts, source, url, title, matched_symbols, matched_sectors, sent_score]` — fixes the legacy timestamp defect by construction.
- **Scorer:** FinBERT (`ProsusAI/finbert`, CPU is sufficient for ~100 headlines/day). Symbol/sector matching from universe names + the alias map. Daily aggregates: `news_sent_market`, `news_sent_sector[sector]`, `news_sent_stock[symbol]`, counts.
- **Honesty clause — no backfill:** historical Indian financial headlines with trustworthy pre-09:15 published timestamps are not freely recoverable at scale. News features enter training with `_present=0` historically and **accrue forward**. They are deliberately excluded from the M3 accept/reject backtest.
- **News-channel acceptance:** after ≥ `gates.paper_days` (22) paper sessions with news features live, compare fired WR/expectancy on news-present vs news-absent days from the journal; only then consider a retrain treating news rows as `present=1`.

## 9. Milestone M6 — Cleanup (½ day)

- Move `scripts/prediction_accuracy.py` → `scripts/diagnostics/prediction_accuracy.py` with a banner: *the contract metric is fired-signal win rate + post-cost expectancy from `backtester.py`; full-coverage accuracy is diagnostic only and near-0.5 is expected.*
- Update README/RUNBOOK references; promote `models/v4` via the existing trainer/promote flow once gates pass.

---

## 10. Explicitly Unchanged (the kept skeleton)

`bars.py` store + validation · `bhavcopy.py` · `screener.py` (ranking + direction hint) · `labeler.py` mechanics (only the geo-override fix) · `blend.py` isotonic calibration · `conformal_gate.py` ACI loop (config-only α/τ₀ change) · `backtester.py` fold/purge/embargo machinery · `risk.py` sizing/limits/kill-switches/blackout · `paper_runner.py` structure · `trainer.py` · `costs.py` · journal/tracking/drift/DVC.

**Retired:** `prediction_accuracy.py` as a decision metric · the `models/v3` bundle (superseded after retrain) · legacy `data/news/*.json` (frozen, unreferenced).

---

## 11. Sequencing & Dependencies

```
M0  Gate-1 fix (½d) ─────────────┐
M1  Geometry retarget (2–3d) ────┤──► M3 Features v4 + Runs A/B/D (3–4d) ──► DECISION POINT
M2  Global/macro data (2–3d) ────┘         │ pass                    │ kill criterion met
    (capture cron starts day 1)            ▼                         ▼
                                   M4 Regime veto + Run C     Stop intraday model work;
                                   (1–2d) ──► Gate 3          redirect news/regime to a
                                           │                  daily/swing system (new plan)
M5  News channel (forward-only, ~5 weeks accrual) ──► Gate 4 paper (≥22 sessions)
M6  Cleanup (½d)
```

M3's decision run needs only M1 + M2 (global features backfill validly). News sentiment is intentionally excluded from the accept/reject backtest because it cannot be honestly backfilled.

## 12. Verification

1. `pytest` — existing suite plus: labeler override test, `test_config.py` (new keys consumed), `test_regime_data.py` PIT filtering, the features-v4 lookahead property test extended to the regime group.
2. `python main.py --mode gates` → Gate 1 passes post-fix; Gate 2 evidence = geometry grid artifact + the config commit freezing the winner.
3. Runs A/B/C/D via `--mode backtest` (+ `--stress`); inspect `reports/v4/run_registry.csv` and the per-regime fairness tables; apply §6.4/§6.5.
4. Paper sessions journal `regime_no_trade` and veto skip reasons with counterfactual labels; news-present vs news-absent comparison after 22 sessions.

---

## 13. Honest Expectations

- **80% of fired signals winning is achievable by construction** — the geometry buys it at fair price. The real fight, and the kill-switch metric, is **post-cost expectancy**: the model must add ~2.5–3.5 points of edge over the floor at low coverage.
- **News will not rescue a 1–2h horizon by itself.** Its realistic intraday value is regime bias and staying out on hostile days; its proven alpha (post-event drift) lives at the daily/swing horizon — which is exactly where this plan redirects it if the kill criterion fires.
- If Gate 3 passes, the v3 economics table (PLAN_v3 §20) applies with the v4 geometry; if it fails, the finding is real information that saves capital, and the data layers built here carry forward unchanged.

---

*PLAN v4 — 2026-07-02. Retargets PLAN_v3's contract from 89%→80% with corrected width math, adds the pre-open regime/news channel with point-in-time discipline, and pre-registers the kill criterion so the next verdict — pass or fail — is finally measured on the system's actual contract.*
