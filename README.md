# Intraday Selective Signal System (v3) — NSE

A 2–3 hour intraday **selective** trading-signal system for NSE large caps.
Entries 09:30–11:00 IST, time barrier entry+3h, hard square-off 14:45, always
flat overnight. See [PLAN_v3.md](PLAN_v3.md) and
[v3_supplementry.md](v3_supplementry.md) for the full specification.

**Stack:** Upstox/NSE data · LightGBM (price) + CatBoost (flow) · weighted blend
+ isotonic calibration · custom Adaptive Conformal Inference gate · triple-barrier
labels · MLflow registry · Evidently drift · DVC · FastAPI · IBM Granite (Ollama).

## The 89% contract (one paragraph)

The win rate is **engineered by barrier geometry**, not demanded from prediction:
at target `a`·ATR / stop `b`·ATR the no-edge win rate is `b/(a+b)`, so wide stops
buy a high win rate at exactly fair price (pre-cost EV = 0). Profit comes from the
**expectancy identity** `EV = δ·(a+b) − c`: the model's only job is to find the
2–6 setups/day where its calibrated P(win) clears the conformal fire threshold;
everything else is silence. Both gates must hold — **win rate ≥ 89% AND post-cost
expectancy ≥ +0.05%/trade** — and the kill-switch metric is expectancy, not win
rate, because at this geometry win rate stays high right up until the system dies.

## CLI (config_v3.yaml only)

```
python main.py --mode backfill          # 1-min bar history (Upstox)
                  bhavcopy               # NSE EOD: equity/F&O/FII-DII + raise-on-miss
                  corp-actions           # corporate-action factor table
                  record                 # live session recorder (08:55→close)
                  screen [--date D]      # 09:30 morning momentum screen
                  geometry-study         # Phase-3 grid → Gate 2
                  backtest [--stress] [--tune]   # purged walk-forward → Gate 3
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

| Gate | Test | Pass |
|---|---|---|
| 0 Engineering | pytest suite + dead-control sweep + coverage ≥ 85% | all green |
| 1 Data | ≥3y bars for PIT universe + reconcile + bhavcopy cross-check | mismatch <0.1% |
| 2 Label/geometry | geometry grid on train folds | baseline ≥86%, δ ≤ 4 pts |
| 3 Backtest | base + 2× stress, full metric contract | win ≥89% ∧ EV ≥+0.05% ∧ no leakage/subsidy ∧ calib ≤2pts |
| 4 Paper | ≥22 sessions | win-rate gap ≤3 pts, positive expectancy |
| 5 Capital | COMPLIANCE checklist + 1 month smallest size | kill switches clean |

## Tests

`pytest` — deterministic synthetic fixtures, no network. The load-bearing test is
`tests/test_features.py::test_no_lookahead_property` (would catch B-4). The
dead-control + dead-import sweeps (`tests/test_dead_controls.py`) fail if any
control loses its live call site or any v1 module is still referenced.

## Operations

[RUNBOOK.md](RUNBOOK.md) — token refresh, session start/stop, restart-resume,
halt clearance, promotion/rollback, drift response, holiday maintenance.
[COMPLIANCE.md](COMPLIANCE.md) — SEBI algo posture; **no live order before broker
approval exists**.

## History

[PLAN.md](PLAN.md) (v1, daily 5-class system) and `STATUS_2026-06-03.md` are
retained as historical record only. v1 was fully decommissioned (its source
tree, models, sentiment/news pipeline, and config removed); see
[v3_supplementry.md](v3_supplementry.md) §16.
