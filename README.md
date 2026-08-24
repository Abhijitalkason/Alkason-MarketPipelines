# Intraday Selective Signal System (v4) — NSE

A 1–2 hour intraday **selective** trading-signal system for NSE large caps.
Entries 09:30–11:00 IST, time barrier entry+2h, hard square-off 14:45, always
flat overnight. Current spec: [PLAN_v4.md](archive/PLAN_v4.md) (targets, regime/news
channel, kill criterion) + [PLAN_v4_STATUS.md](archive/PLAN_v4_STATUS.md) (implementation
status, executed results, decision-phase runbook). [PLAN_v3.md](archive/PLAN_v3.md) /
[v3_supplementry.md](archive/v3_supplementry.md) remain the architecture reference.

**Stack:** Upstox/NSE data · pre-open global/macro regime channel (yfinance) +
forward-only news sentiment (RSS + GDELT, FinBERT) · LightGBM (price) + CatBoost
(flow) · weighted blend + isotonic calibration · custom Adaptive Conformal
Inference gate · triple-barrier labels · MLflow registry · Evidently drift · DVC
· FastAPI (Swagger at `/docs`) · IBM Granite (Ollama).

## The v4 contract (one paragraph)

The win rate is **engineered by barrier geometry**, not demanded from prediction:
at target `a`·ATR / stop `b`·ATR the theoretical no-edge win rate is `b/(a+b)`.
v4 targets **win rate ≥ 80% on emitted signals AND post-cost expectancy ≥
+0.05%/trade** via the expectancy identity `EV = δ·(a+b)·ATR − c`; the model's
only job is the few setups/day whose calibrated P(win) clears the conformal fire
threshold — everything else is silence. The kill-switch metric is expectancy, and
the fired-win-rate confidence interval is computed **day-clustered** (regime
features are constant per day). ⚠️ Measured caveat (PLAN_v4_STATUS §2c): on real
1-min paths the 2h time barrier truncates wide stops, so the *realized* baseline
sits far below `b/(a+b)` — Gate 2 (geometry study) must find a geometry whose
**measured** baseline reaches the band before Gate 3 means anything.

## CLI (config_v3.yaml only)

```
python main.py --mode backfill          # 1-min bar history (Upstox)
                  bhavcopy               # NSE EOD: equity/F&O/FII-DII + raise-on-miss
                  corp-actions           # corporate-action factor table
                  record                 # live session recorder (08:55→close)
                  screen [--date D]      # 09:30 morning momentum screen
                  geometry-study         # Phase-3 grid → Gate 2
                  backtest [--stress] [--tune] [--symbols S1 S2 ...]  # purged walk-forward → Gate 3
                                          # --symbols restricts the universe (PLAN_v4 Run D)
                  regime-backfill [--years N]      # global/macro prior-close history (yfinance)
                  regime-capture [--date D]        # pre-open snapshot (cron 08:45 IST)
                  news-capture [--date D]          # RSS+GDELT headlines → FinBERT (cron 08:00/09:10 IST)
                  train-v3 [--end D] [--months N]  # production train → Staging
                  promote --version N --confirm    # Staging → Production
                  rollback --confirm
                  recalibrate            # rolling 60-day isotonic refit
                  paper --capital N      # Gate-4 paper loop (≥22 sessions)
                  drift                  # Evidently drift → ACI recalib → halt
                  gates [--start S --end E]        # executable Gates 0/3/4
                  serve                  # FastAPI (requires V3_API_KEY)
```

## API (every endpoint needs `X-API-Key`)

| Endpoint | Returns |
|---|---|
| `GET /health` | model source/age, gate τ + age, halt status, journal chain OK |
| `GET /signals/today` | today's fills/exits/explanations (from the journal) |
| `GET /signals/history?days=N` | past signals + rolling win rate/expectancy/drawdown |
| `GET /screen/today` | the 09:30 screen + scores + excluded symbols |
| `GET /performance` | rolling metrics vs gate thresholds + per-symbol coverage |
| `GET /drift` | last drift report + last ACI recalibration |
| `POST /backtest` | background job (`GET /backtest/{job_id}` for status) |

The API loads the **Production** registry bundle at startup (fallback `models/v3/`),
runs inference off the event loop, and checks the feature schema before every
predict. No model-mutating endpoint exists — training/promotion are CLI + registry.

## Acceptance gates (`reports/v3/gates/`, all executable)

| Gate | Test | Pass (v4) |
|---|---|---|
| 0 Engineering | pytest suite + dead-control sweep + coverage ≥ 85% | all green |
| 1 Data | ≥3y bars for ≥`min_data_names_frac` of PIT universe + reconcile + cross-check | mismatch <0.1% |
| 2 Label/geometry | geometry grid on train folds | measured baseline in [0.76, 0.84], δ ≤ 3 pts |
| 3 Backtest | base + 2× stress, full metric contract | win ≥80% ∧ EV ≥+0.05% ∧ day-clustered CI above floor ∧ no leakage/subsidy ∧ calib ≤3pts |
| 4 Paper | ≥22 sessions | win-rate gap ≤3 pts, positive expectancy |
| 5 Capital | COMPLIANCE checklist + 1 month smallest size | kill switches clean |

## Tests

`pytest` — deterministic synthetic fixtures, no network. The load-bearing tests:
`tests/test_features.py::test_no_lookahead_property` (would catch B-4),
`tests/test_regime.py` (PIT lookahead exclusion for the regime channel + veto
logic + news degradation), and `tests/test_geometry_v4.py` (the labeler geometry
override — the latent bug that would have silently broken every grid sweep). The
dead-control + dead-import sweeps (`tests/test_dead_controls.py`) fail if any
control loses its live call site or any v1 module is still referenced.

## Diagnostics (not decision metrics)

`scripts/diagnostics/prediction_accuracy.py` — full-coverage OOF accuracy/AUC.
**Retired as a decision metric** (PLAN_v4 §9): a selective system is designed to
look near-random at 100% coverage, so ~0.50 there is expected, not failure. The
contract metrics are fired-signal win rate + expectancy from the backtester.

## Operations

[RUNBOOK.md](RUNBOOK.md) — token refresh, session start/stop, restart-resume,
halt clearance, promotion/rollback, drift response, holiday maintenance.
[COMPLIANCE.md](COMPLIANCE.md) — SEBI algo posture; **no live order before broker
approval exists**.

## History

[PLAN.md](archive/PLAN.md) (v1, daily 5-class system) and `STATUS_2026-06-03.md` are
retained as historical record only. v1 was fully decommissioned (its source
tree, models, sentiment/news pipeline, and config removed); see
[v3_supplementry.md](archive/v3_supplementry.md) §16.
