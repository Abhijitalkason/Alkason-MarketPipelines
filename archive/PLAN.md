# AI-MLOps Stock Prediction System — Project Plan (2026)

> **Project:** End-to-end production-grade stock prediction system for NSE/BSE Indian stocks  
> **Stack Year:** 2026 best-in-class libraries and models  
> **Goal:** 5-class trading signal (Strong Buy / Buy / Hold / Sell / Strong Sell) with maximum achievable accuracy  
> **Stocks:** Configurable via `config/config.yaml` — never hardcoded in source code

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [2026 Stack Audit & Decisions](#2-2026-stack-audit--decisions)
2b. [Sprint 1 vs Sprint 2 — Known Issues Register](#2b-sprint-1-vs-sprint-2--known-issues-register)
3. [Project Structure](#3-project-structure)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Phase 1 — Data Ingestion](#5-phase-1--data-ingestion)
6. [Phase 2 — Data Validation](#6-phase-2--data-validation)
7. [Phase 3 — Feature Engineering](#7-phase-3--feature-engineering)
8. [Phase 4 — SLM Layer (FinBERT + IBM Granite 4.1)](#8-phase-4--slm-layer)
9. [Phase 5 — ML Models](#9-phase-5--ml-models)
10. [Phase 6 — Training Pipelines](#10-phase-6--training-pipelines)
11. [Phase 7 — Evaluation & Comparison](#11-phase-7--evaluation--comparison)
12. [Phase 8 — Drift Monitoring](#12-phase-8--drift-monitoring)
13. [Phase 9 — FastAPI REST Service](#13-phase-9--fastapi-rest-service)
14. [Phase 10 — Docker Deployment](#14-phase-10--docker-deployment)
15. [Phase 11 — CLI Entry Point](#15-phase-11--cli-entry-point)
16. [Phase 12 — Configuration](#16-phase-12--configuration)
17. [Technology Stack Summary](#17-technology-stack-summary)
18. [File Creation Order](#18-file-creation-order)
19. [Verification & Testing](#19-verification--testing)
20. [Accuracy Expectations](#20-accuracy-expectations)

---

## 1. System Overview

### What This System Does
A production-grade end-to-end MLOps pipeline that:
- Downloads 5 years of historical OHLCV data for major NSE/BSE stocks
- Engineers 60+ technical indicator features
- Scrapes financial news and uses **FinBERT** to extract sentiment as additional ML features
- Trains 5 different models (including a 2025 foundation model — Amazon Chronos-2)
- Supports **two training modes**: Manual (save artifacts locally) vs MLflow-tracked (full experiment logging)
- Generates **5-class trading signals** with confidence scores
- Explains every prediction in plain English using **IBM Granite 4.1-3B** (April 2026)
- Serves predictions via **FastAPI REST API** and **CLI**
- Monitors for model/data drift using **Evidently AI**
- Deploys as **3 Docker containers** (MLflow server + FastAPI + Ollama/Granite)

### Target Stocks (Config-Driven)
Stocks are **never hardcoded** in source code. Every module reads from `config/config.yaml` at runtime:

```python
# Correct pattern used everywhere
cfg = yaml.safe_load(open("config/config.yaml"))
symbols = cfg["stocks"]["symbols"]   # e.g. ["RELIANCE.NS", "TCS.NS", ...]
```

Default symbols in `config.yaml` (changeable without touching any source file):
| Symbol | Company |
|---|---|
| RELIANCE.NS | Reliance Industries |
| TCS.NS | Tata Consultancy Services |
| INFY.NS | Infosys |
| HDFCBANK.NS | HDFC Bank |
| WIPRO.NS | Wipro |

### Prediction Output
```
Signal:      Strong Buy / Buy / Hold / Sell / Strong Sell
Confidence:  87%
Explanation: "RELIANCE.NS shows strong buy signals driven by oversold RSI (31.2),
              a bullish MACD crossover, volume 2.3x the 20-day average,
              and predominantly positive news sentiment from 8 of 10 recent headlines."
```

---

## 2. 2026 Stack Audit & Decisions

The following table documents every deliberate 2026 technology choice vs older approaches:

| Component | Old Approach | 2026 Decision | Reason |
|---|---|---|---|
| **Data ingestion** | yfinance | **OpenBB SDK** (yfinance provider) | Production-grade: caching, rate limiting, retry logic, async support |
| **Technical indicators** | pandas-ta | **pandas-ta-classic** | Original pandas-ta is unmaintained since 2022; this is the active May 2026 community fork |
| **Deep learning framework** | Keras / TensorFlow | **PyTorch** | PyTorch dominates in 2026 (55% research papers, 37.7% of AI jobs); TF in decline for new projects |
| **Time series model** | LSTM only | **PyTorch LSTM + Amazon Chronos-2** | Chronos-2 (Oct 2025) is zero-shot SOTA; adds a fundamentally different signal type |
| **AutoML baseline** | None | **AutoGluon** | Beats 99% of Kaggle competitors; strong benchmark to validate manual models against |
| **Data validation** | None | **Pandera + Pydantic v2** | Pandera for DataFrames; Pydantic v2 for API — catches bad data early in pipeline |
| **Drift monitoring** | None | **Evidently AI** | 2026 standard for ML monitoring; detects data, target, and prediction drift |
| **Experiment tracking** | MLflow | **MLflow 3.12+** | Still best open-source tracker; redesigned UI, new `search_logged_models()` API |
| **API framework** | FastAPI | **FastAPI** | Still #1 Python API framework in 2026 |
| **SLM Sentiment** | None | **FinBERT** (ProsusAI, 110M) | 97% accuracy on Financial PhraseBank — gold standard, unbeaten in 2026 |
| **SLM Explanation** | None | **IBM Granite 4.1-3B** (Apr 2026) | Latest IBM SLM; runs on 4GB RAM via Ollama; native tool calling |
| **Gradient boosting** | LightGBM + XGBoost | **LightGBM + XGBoost** | Still top performers on tabular financial data in 2026 |
| **HPO** | Optuna | **Optuna** | Still best open-source hyperparameter optimization |
| **Model & data versioning** | No versioning (files overwritten) | **DVC** (Data Version Control) | Git-integrated versioning for all `.pkl`, `.pt`, and processed data files — rollback any model to any previous version locally |

---

## 2b. Sprint 1 vs Sprint 2 — Known Issues Register

Decisions agreed with lead and management review. Issues are classified by whether they are **Sprint 1 must-fix** or **safe to defer to Sprint 2**.

| # | Issue | Sprint | Rationale |
|---|---|---|---|
| 0 | Stocks configurable via config — never hardcoded | **Sprint 1 — Fixed in plan** | Trivial to enforce during implementation; single pattern throughout |
| A | Scaler leakage for LightGBM / XGBoost / AutoGluon / Chronos-2 | **N/A — Not applicable** | Tree-based models and foundation models are scale-invariant; no scaling applied |
| A | Scaler leakage for LSTM only | **Sprint 2** | SMOTE+Scaler is now fitted per OOF fold (partially fixed); full per-fold scaler re-fit for LSTM sequences deferred. LSTM accuracy may be ~2–5% optimistic until fixed. Noted in evaluation reports. |
| — | Model versioning for manual .pkl files | **Sprint 1 — Fixed with DVC** | DVC added to plan. Every `.pkl`/`.pt` file tracked via `dvc add` after each training run. Git stores version pointer, DVC stores actual file. Rollback via `git checkout + dvc checkout`. |
| B | Async concurrent news scraping | **Sprint 2** | Pure performance optimisation — sequential works for 5 stocks. No correctness impact. |
| B | Historical news unavailability → neutral imputation | **Sprint 1 — Fixed in plan** | Must be in from day 1; without it the pipeline crashes on 90% of training rows. 3 lines of code. |
| C | OOF Stacking Ensemble (meta-learner overfitting) | **Sprint 1 — Fixed in plan** | Critical correctness issue. Deferring forces full rewrite in Sprint 2 and produces misleading metrics in Sprint 1 demo. Correct OOF flow documented in Sections 9.6 and 10. |

---

## 3. Project Structure

```
AI-MLOps-Solution/
│
├── config/
│   └── config.yaml                    # All configuration (stocks, thresholds, model params, SLM)
│
├── data/
│   ├── raw/                           # Raw OHLCV CSVs downloaded via OpenBB
│   ├── processed/                     # Feature-engineered Parquet files (60+ features + labels)
│   └── news/                          # Cached FinBERT-ready headlines per stock/date (JSON)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py               # OpenBB SDK downloader with yfinance fallback
│   │   ├── validation.py              # Pandera schema checks (raw + processed data)
│   │   └── features.py               # 60+ technical indicators + 4 sentiment features + labels
│   │
│   ├── slm/
│   │   ├── __init__.py
│   │   ├── news_scraper.py           # DuckDuckGo + NewsAPI headline fetcher with caching
│   │   ├── sentiment.py              # FinBERT → sentiment_score, confidence, count, trend
│   │   └── explainer.py              # IBM Granite 4.1-3B via Ollama → NL prediction explanation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lgbm_model.py             # LightGBM multi-class + Optuna tuning + SHAP
│   │   ├── xgboost_model.py          # XGBoost multi-class + early stopping
│   │   ├── lstm_model.py             # PyTorch 2-layer LSTM → Softmax(5)
│   │   ├── chronos_model.py          # Amazon Chronos-2 (Oct 2025) zero-shot forecasting
│   │   ├── autogluon_model.py        # AutoGluon tabular AutoML baseline
│   │   └── ensemble_model.py         # Stacking: LGBM+XGB+LSTM+Chronos-2 → LightGBM meta
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── manual_trainer.py         # Train all models → save .pkl / .pt artifacts locally
│   │   └── mlflow_trainer.py         # Same training + MLflow 3.10 logging + model registry
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Side-by-side metrics table + confusion matrix heatmaps
│   │   └── monitoring.py             # Evidently AI drift reports (data/target/prediction)
│   │
│   └── api/
│       ├── __init__.py
│       └── app.py                    # FastAPI service with Pydantic v2 models
│
├── docker/
│   ├── Dockerfile.api                 # FastAPI + FinBERT weights baked in
│   ├── Dockerfile.mlflow              # MLflow 3.10 tracking server
│   ├── Dockerfile.ollama              # Ollama + auto-pull Granite 4.1-3B on startup
│   └── docker-compose.yaml            # Orchestrates all 3 services + shared volumes
│
├── models/
│   ├── manual/                        # .pkl (LightGBM/XGBoost), .pt (LSTM), .json (AutoGluon)
│   └── mlflow/                        # MLflow model registry mirror for API access
│
├── reports/
│   ├── comparison_<date>.csv          # Model comparison output
│   └── drift/
│       └── <date>_drift_report.html   # Evidently AI drift HTML reports
│
├── mlruns/                            # MLflow artifact store (add to .gitignore)
│
├── notebooks/
│   └── exploration.ipynb             # EDA + SHAP plots + sentiment analysis + Chronos-2 demo
│
├── main.py                            # CLI entry point (argparse)
├── requirements.txt                   # All Python dependencies
├── .env.example                       # Environment variable template
├── .dvc/                              # DVC internal config (auto-created by dvc init)
│   └── config                         # DVC settings (local remote path, etc.)
├── .dvcignore                         # Files DVC should ignore (like .gitignore)
├── models/manual/*.dvc                # DVC pointer files — committed to Git
├── data/processed/*.dvc               # DVC pointer files for processed Parquet files
├── PLAN.md                            # This file
└── README.md                          # Setup and usage guide
```

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│                                                                      │
│  OpenBB SDK ──► OHLCV (5y daily)   DuckDuckGo/NewsAPI ──► Headlines │
│      │                                      │                        │
│      ▼                                      ▼                        │
│  Pandera Validation              FinBERT (110M, 97% acc.)            │
│      │                                      │                        │
│      ▼                                      ▼                        │
│  60+ Technical Indicators    4 Sentiment Features                    │
│  (pandas-ta-classic)         (score, confidence,                     │
│                               count_7d, trend_3d)                   │
│                    └──────────────┬──────────────┘                   │
│                                   ▼                                  │
│                        Feature Matrix (64+ features)                 │
│                        SMOTE for class balancing                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                                   │
│                                                                      │
│  ┌───────────┐  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ LightGBM  │  │  XGBoost  │  │ PyTorch LSTM │  │ Chronos-2   │  │
│  │ (Optuna)  │  │  (ES)     │  │  (2-layer)   │  │ (Amazon,    │  │
│  │   SHAP    │  │           │  │              │  │  Oct 2025)  │  │
│  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘  └──────┬──────┘  │
│        └──────────────┴────────────────┴──────────────────┘         │
│                                    │                                 │
│                                    ▼                                 │
│                    Stacking Ensemble (LightGBM meta-learner)         │
│                    5-class output: Strong Buy/Buy/Hold/Sell/S.Sell   │
│                                                                      │
│  ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌ ╌  │
│  AutoGluon  ← independent baseline only — NOT part of the stack     │
│  Trains on same features; used to benchmark ensemble quality         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                       ▼
┌─────────────────────┐  ┌──────────────────┐  ┌───────────────────┐
│  MANUAL TRAINER      │  │  MLFLOW TRAINER   │  │  EVIDENTLY AI     │
│                      │  │                  │  │  MONITORING       │
│  Saves .pkl/.pt to   │  │  MLflow 3.10     │  │                   │
│  models/manual/      │  │  - Experiments   │  │  - Data drift     │
│                      │  │  - Model Registry│  │  - Target drift   │
│  No experiment log   │  │  - SHAP plots    │  │  - Pred drift     │
│  Pure comparison     │  │  - Params/metrics│  │  - HTML reports   │
└─────────────────────┘  └──────────────────┘  └───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPLANATION LAYER                             │
│                                                                      │
│  Prediction + SHAP features + Sentiment score                        │
│                    │                                                 │
│                    ▼                                                 │
│         IBM Granite 4.1-3B (April 2026, via Ollama)                 │
│         "RELIANCE.NS is a Strong Buy because..."                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                           │
│                                                                      │
│  GET  /health                    POST /predict                       │
│  GET  /models                    POST /predict/batch                 │
│  GET  /recommendation/{symbol}   GET  /sentiment/{symbol}            │
│  POST /ask  (NL query)           GET  /comparison                    │
│  GET  /drift                     POST /retrain                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DOCKER INFRASTRUCTURE                            │
│                                                                      │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  mlflow:5000   │  │   api:8000        │  │  ollama:11434      │  │
│  │  MLflow 3.12   │  │  FastAPI+FinBERT  │  │  Granite 4.1-3B   │  │
│  │  Tracking UI   │  │  Pydantic v2      │  │  Explanation gen.  │  │
│  └────────────────┘  └──────────────────┘  └────────────────────┘  │
│         └─────────────────────┴──────────────────────┘              │
│                      Shared volumes: mlruns, models, data/news       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase 1 — Data Ingestion

**File:** `src/data/ingestion.py`

### Library
**OpenBB SDK** (2026 production-grade replacement for raw yfinance):
- Uses `yfinance` as the data provider internally — same data, better reliability
- Built-in: rate limiting, request caching, retry on failure, async support
- Simple fallback: if OpenBB fails, direct yfinance call as backup

### Data Fetched
- **Symbols:** Read at runtime from `config["stocks"]["symbols"]` — never hardcoded
- **Period:** Read from `config["stocks"]["period"]` (default: `5y`)
- **Fields:** Open, High, Low, Close, Volume, Adjusted Close
- **Output:** `data/raw/<SYMBOL>.csv` (one file per symbol)

### Key Behaviour
- Incremental refresh: detects last available date, fetches only missing rows
- Data quality pre-check: alerts if fewer than 200 rows returned
- Logs download status per symbol to console

---

## 6. Phase 2 — Data Validation

**File:** `src/data/validation.py`

### Library
**Pandera** (active Jan 2026) + **Pydantic v2** (API layer)

### Pandera Checks (Raw OHLCV)
- No null values in Open, High, Low, Close, Volume
- Close > 0 (sanity check)
- Volume >= 0
- Date index is monotonically increasing (no duplicates)
- Minimum 500 rows of data

### Pandera Checks (Processed Features)
- All 64 expected columns present
- No NaN values after feature calculation
- Labels are strictly in {0, 1, 2, 3, 4}
- Feature values within expected statistical bounds (z-score check)

### Pydantic v2 (API)
- All FastAPI request/response models defined as Pydantic v2 `BaseModel`
- Strict type validation on symbol format (must end in `.NS` or `.BO`)
- Confidence score clamped to [0.0, 1.0]

---

## 7. Phase 3 — Feature Engineering

**File:** `src/data/features.py`  
**Library:** `pandas-ta-classic` (active May 2026 fork, 192 indicators)

### Technical Indicators (60+)

| Category | Features | Count |
|---|---|---|
| **Returns** | Log returns: 1d, 3d, 5d, 10d, 20d; overnight gap | 6 |
| **Moving Averages** | SMA 5/10/20/50/200; EMA 5/10/20/50; price/MA ratios (8 ratios) | 17 |
| **Momentum** | RSI(7), RSI(14), Stochastic K, Stochastic D, Williams %R, CCI, ROC(10), ROC(20) | 8 |
| **Trend** | MACD, MACD Signal, MACD Histogram, ADX, Aroon Up, Aroon Down | 6 |
| **Volatility** | BB Upper, BB Lower, BB Width, BB %B, ATR(14), Historical Volatility(20) | 6 |
| **Volume** | OBV, VWAP, MFI(14), Volume SMA ratio (20d), Volume ROC | 5 |
| **Candlestick** | Body size, Upper shadow, Lower shadow, Doji flag, Hammer flag, Engulfing flag | 6 |
| **Calendar** | Day of week, Month, Quarter, Is month-end | 4 |
| **Sentiment (SLM)** | sentiment_score, sentiment_confidence, news_count_7d, sentiment_trend_3d | 4 |
| **TOTAL** | | **62–68 features** |

### Target Label
Next-day forward return thresholds:

| Class | Label | Condition |
|---|---|---|
| 4 | Strong Buy | next_day_return > +2.0% |
| 3 | Buy | next_day_return in (0.5%, 2.0%] |
| 2 | Hold | next_day_return in [-0.5%, 0.5%] |
| 1 | Sell | next_day_return in [-2.0%, -0.5%) |
| 0 | Strong Sell | next_day_return < -2.0% |

### Class Imbalance Handling
**SMOTE** (Synthetic Minority Oversampling Technique) applied **within each OOF fold's training portion only** — never on validation folds or globally — to prevent data leakage. See Section 9.6 for the per-fold procedure.

---

## 8. Phase 4 — SLM Layer

### Overview
The SLM layer adds two capabilities:
1. **FinBERT** → converts financial news into 4 ML features (improves model accuracy)
2. **IBM Granite 4.1-3B** → converts ML predictions into plain English (improves usability)

### 8.1 News Scraper (`src/slm/news_scraper.py`)
- **Source 1:** `ddgs` library (formerly `duckduckgo-search`, renamed in v8.x) — free, no API key, scrapes DuckDuckGo News
- **Source 2:** NewsAPI.org (optional, free tier: 100 requests/day)
- **Per stock:** fetches last 7 days of headlines
- **Cache key:** `(symbol, date)` composite → `data/news/<SYMBOL>_<YYYY-MM-DD>.json` — deterministic, avoids re-fetching
- **Filters:** English language, financial keywords only
- **Historical dates (no news available):** DuckDuckGo/NewsAPI cannot retrieve news from years ago. For all historical training rows where no cached news exists, all 4 sentiment features are **imputed as neutral** (`score=0.0, confidence=0.5, count=0, trend=0.0`). This is deterministic and logged — never a crash or skipped row. Only recent data (~last 90 days) will have real sentiment values.
- **Sprint 2 improvement:** Async `aiohttp`-based concurrent fetching for all symbols in parallel (current Sprint 1 implementation is sequential — acceptable for 5 stocks)

### 8.2 FinBERT Sentiment (`src/slm/sentiment.py`)

| Detail | Value |
|---|---|
| Model | `ProsusAI/finbert` |
| Parameters | 110M (BERT-based) |
| Accuracy | 97% on Financial PhraseBank |
| Speed | ~50–100 tokens/sec on CPU (instant per headline) |
| License | Apache 2.0 |
| Training data | 4.9B tokens: corporate reports, earnings transcripts, analyst reports |

**Output (4 ML features added to feature matrix):**

| Feature | Description |
|---|---|
| `sentiment_score` | Weighted average: positive=+1, neutral=0, negative=-1 |
| `sentiment_confidence` | Mean confidence across all headlines |
| `news_count_7d` | Total headlines fetched for the past 7 days |
| `sentiment_trend_3d` | Change in sentiment_score over last 3 days |

### 8.3 IBM Granite 4.1-3B Explainer (`src/slm/explainer.py`)

| Detail | Value |
|---|---|
| Model | IBM Granite 4.1-3B Instruct |
| Release | April 2026 (latest IBM SLM) |
| Deployment | Ollama locally (`ollama pull granite4.1:3b`) |
| RAM required | ~4 GB (CPU-only, no GPU needed) |
| Speed | 12–18 tokens/sec on CPU |
| License | Apache 2.0 |
| Optional cloud | IBM watsonx.ai SDK (`ibm-watsonx-ai`) |

**Input to Granite:**
```
Stock: RELIANCE.NS
Signal: Strong Buy (confidence: 87%)
Top SHAP features:
  - RSI_14 = 31.2 (oversold zone)
  - MACD = 12.4 (bullish crossover)
  - volume_sma_ratio = 2.3 (high volume)
  - sentiment_score = 0.72 (positive news)
  - price_sma200_ratio = 0.88 (12% below 200d average)

Task: Explain this recommendation in 2-3 sentences for a retail investor.
```

**Output:**  
_"RELIANCE.NS is showing strong buy signals. The RSI at 31.2 indicates the stock is oversold and may be due for a bounce, while a bullish MACD crossover confirms upward momentum. High trading volume (2.3x the 20-day average) and positive news sentiment further support this recommendation."_

---

## 9. Phase 5 — ML Models

### 9.1 LightGBM (`src/models/lgbm_model.py`)
- **Objective:** `multiclass` (5 classes)
- **Hyperparameter tuning:** Optuna (50 trials, time-series CV)
- **Key params tuned:** `num_leaves`, `max_depth`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `min_child_samples`
- **Explainability:** SHAP values logged for top-20 features
- **Why LightGBM:** Fastest training on tabular data, handles mixed feature types, leaf-wise growth is ideal for financial data patterns

### 9.2 XGBoost (`src/models/xgboost_model.py`)
- **Objective:** `multi:softprob` (returns probabilities)
- **Early stopping:** 50 rounds on validation F1
- **Key params:** `max_depth`, `eta`, `subsample`, `colsample_bytree`
- **Why XGBoost:** Strong comparison model; different regularisation approach to LightGBM reveals complementary strengths

### 9.3 PyTorch LSTM (`src/models/lstm_model.py`)
- **Framework:** PyTorch (2026 dominant framework — 55% research papers)
- **Architecture:**
  ```
  Input (sequence_len=20, n_features=64)
      → LSTM(128, num_layers=2, dropout=0.3, batch_first=True)
      → Linear(128 → 64)
      → ReLU
      → Dropout(0.3)
      → Linear(64 → 5)
      → Softmax
  ```
- **Training:** Adam optimizer, CrossEntropyLoss, ReduceLROnPlateau scheduler
- **Input:** Sliding window of 20 trading days of features per stock
- **Why PyTorch LSTM:** Captures temporal sequential patterns missed by tree models; PyTorch preferred over Keras/TF in 2026

### 9.4 Amazon Chronos-2 (`src/models/chronos_model.py`)
- **Model:** `amazon/chronos-bolt-small` (Chronos-2 family, October 2025)
- **Type:** Time series foundation model — zero-shot, no training required
- **Pre-training:** 700B+ data points from diverse time series datasets
- **Usage in our system:**
  1. Feed raw Close price sequence to Chronos-2
  2. Get next-day price forecast (probabilistic)
  3. Convert forecast to signal: if forecast > current + threshold → Buy, etc.
  4. Use Chronos-2 signal as an additional feature in the stacking ensemble
- **Why Chronos-2:** Adds fundamentally different information — a foundation model sees patterns invisible to feature-engineered models; zero-shot, no overfitting risk

### 9.5 AutoGluon (`src/models/autogluon_model.py`)
- **Type:** AutoML — trains 10+ models internally (RF, ExtraTree, LightGBM, XGB, NeuralNet, CatBoost, etc.) and auto-stacks
- **Purpose:** Strong automated baseline to benchmark our hand-crafted ensemble against
- **Expected result:** AutoGluon often beats manual single models; if our ensemble beats AutoGluon, that validates our architecture

### 9.6 Stacking Ensemble (`src/models/ensemble_model.py`)

- **Level 0 (base learners):** LightGBM + XGBoost + PyTorch LSTM + Chronos-2
- **Level 1 (meta-learner):** LightGBM trained **only on Out-Of-Fold (OOF) predictions** — never on in-sample predictions
- **Fallback:** Soft probability averaging (weighted by individual model F1 scores)
- **Why stacking:** Each model captures different signal types — tree models (features), LSTM (sequences), Chronos-2 (foundation patterns); meta-learner learns optimal combination

#### OOF Procedure (Critical — prevents meta-learner overfitting)

The naive approach of training Level-0 models on the full training set and predicting back on that same data causes **severe overfitting** of the meta-learner (it learns to trust whichever model memorised the training data, not which generalises best). The correct procedure:

```
Training Data (chronological, ~1000 trading days per stock)

┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  Fold 1  │  Fold 2  │  Fold 3  │  Fold 4  │  Fold 5  │
│ ~200 days│ ~200 days│ ~200 days│ ~200 days│ ~200 days│
└──────────┴──────────┴──────────┴──────────┴──────────┘

OOF Generation (TimeSeriesSplit — no shuffle, strictly chronological):

  Iter 1: TRAIN[F1]          → PREDICT[F2] → store OOF[F2]
  Iter 2: TRAIN[F1+F2]       → PREDICT[F3] → store OOF[F3]
  Iter 3: TRAIN[F1+F2+F3]    → PREDICT[F4] → store OOF[F4]
  Iter 4: TRAIN[F1+F2+F3+F4] → PREDICT[F5] → store OOF[F5]

  Each iteration: SMOTE applied to train portion only
                  Scaler fitted on train portion only, transforms val portion

OOF Pool → rows F2+F3+F4+F5 (~800 rows, fully independent predictions)
Meta-learner input:  [lgbm_proba(5), xgb_proba(5), lstm_proba(5), chronos_proba(5)] = 20 features
Meta-learner target: true labels for OOF rows

Train LightGBM meta-learner on OOF Pool only
       ↓
Retrain all Level-0 models on FULL training set (F1+F2+F3+F4+F5)
       ↓
Test set: Level-0 (full retrain) → 20-feature meta input → Meta-Learner → Final prediction
```

**Key rules enforced inside each OOF fold iteration:**
- SMOTE applied **within the fold's training portion only**
- Scaler fitted **on fold's training portion only**, applied to fold's validation portion (also addresses Issue A for LSTM)
- Chronos-2 receives raw price sequence — no scaling required
- Level-0 models trained from scratch each iteration (no warm-starting across folds)

**What gets logged to MLflow:**
- OOF F1 score per fold per model (shows temporal consistency across time periods)
- Full OOF prediction array saved as artifact (reproducible)
- Meta-learner feature importances (which Level-0 model the meta-learner trusts most)

---

## 10. Phase 6 — Training Pipelines

### 10.1 Manual Trainer (`src/training/manual_trainer.py`)

```bash
python main.py --mode train --trainer manual
python main.py --mode train --trainer manual --symbol RELIANCE.NS  # single stock
```

**What it does:**
1. Loads processed feature Parquet for each stock
2. Time-series aware train/test split (80/20, no shuffle)
3. Trains all models via OOF loop (per-fold SMOTE + Scaler)
4. Evaluates on held-out test set
5. Saves artifacts — **one file per stock per model type:**
   - LightGBM/XGBoost/Ensemble: `models/manual/<SYMBOL>_<model>.pkl`
   - PyTorch LSTM: `models/manual/<SYMBOL>_lstm.pt`
   - AutoGluon: `models/manual/<SYMBOL>_autogluon/`
   - Version metadata: `models/manual/metadata.json`
6. **DVC tracks every saved file:**
   ```bash
   dvc add models/manual/RELIANCE.NS_lgbm.pkl   # auto-run after each save
   git add models/manual/RELIANCE.NS_lgbm.pkl.dvc
   git commit -m "manual train v2 - RELIANCE.NS lgbm"
   ```
7. Prints full metrics table to terminal

**No MLflow logging — purely for comparison with MLflow results.**

#### DVC Rollback (Manual Mode)
To restore any previous version of a `.pkl` file:
```bash
# See all past versions
git log --oneline

# Roll back ALL models to a previous commit
git checkout <commit-hash>
dvc checkout
# → all .pkl and .pt files restored to that exact version

# Roll back a SINGLE stock's model only
git checkout <commit-hash> -- models/manual/HDFCBANK.NS_lgbm.pkl.dvc
dvc checkout models/manual/HDFCBANK.NS_lgbm.pkl
# → only HDFCBANK restored; other stocks unchanged
```

### 10.2 MLflow Trainer (`src/training/mlflow_trainer.py`)

```bash
python main.py --mode train --trainer mlflow
```

**What it does (same training + MLflow 3.10 logging):**
1. All steps from Manual Trainer above
2. Wraps each model in `mlflow.start_run(run_name="<model>_<symbol>")`
3. Logs:
   - **Parameters:** all hyperparameters
   - **Metrics:** accuracy, macro F1, weighted F1, per-class precision/recall, ROC-AUC
   - **Artifacts:** confusion matrix heatmap, SHAP feature importance plot, feature list
   - **Model:** saved model artifact via `mlflow.lightgbm.log_model()` etc.
4. Registers best ensemble to **MLflow 3.12 Model Registry** as `<SYMBOL>-stock-predictor`
5. Supports staging → production promotion via MLflow UI (http://localhost:5000)

### Training Pipeline Flow

```
Load Parquet features
        │
        ▼
Chronological train/test split (80% train / 20% test — no shuffle)
        │
        ├─────────────────────────────────────────────────────────────────┐
        │           PHASE A: Individual Model Training                    │
        │           (TimeSeriesSplit, 5 folds, per-fold SMOTE + Scaler)  │
        │                                                                 │
        │  ┌─────────────────────────────────────────────┐               │
        │  │ FOR each fold k in [1..4] (TimeSeriesSplit): │               │
        │  │   train_fold = rows up to fold k             │               │
        │  │   val_fold   = rows in fold k+1              │               │
        │  │                                              │               │
        │  │   Apply SMOTE  → train_fold only             │               │
        │  │   Fit Scaler   → train_fold only             │               │
        │  │   Transform    → val_fold (no fit)           │               │
        │  │                                              │               │
        │  │   Train: LightGBM (Optuna 50 trials)        │               │
        │  │   Train: XGBoost  (early stopping)           │               │
        │  │   Train: PyTorch LSTM (20-day window)        │               │
        │  │   Run:   Chronos-2 (zero-shot, no training)  │               │
        │  │                                              │               │
        │  │   Predict val_fold → store OOF[k+1]         │               │
        │  └─────────────────────────────────────────────┘               │
        │                    │                                            │
        │                    ▼                                            │
        │         OOF Pool (Folds 2–5, ~800 rows)                        │
        │         [lgbm_proba, xgb_proba, lstm_proba, chronos_proba]     │
        │                    │                                            │
        │                    ▼                                            │
        │         Train LightGBM Meta-Learner on OOF Pool                │
        │                    │                                            │
        │                    ▼                                            │
        │         Retrain all Level-0 on FULL train set                  │
        └─────────────────────────────────────────────────────────────────┘
        │
        │           PHASE B: AutoGluon Baseline (separate, no stacking)
        │           Train AutoGluon on full train set → evaluate on test
        │
        ▼
Evaluate ALL models on held-out TEST SET (last 20%, never seen during training)
        │
        ▼
Save artifacts / Log to MLflow
```

**Why this flow is correct:**
- The meta-learner trains on OOF predictions (independent) — not in-sample predictions
- Each OOF fold's scaler is independent — no future data in normalization
- SMOTE only touches training rows — validation rows are never oversampled
- AutoGluon runs independently as a comparison baseline (not part of the stack)

---

## 11. Phase 7 — Evaluation & Comparison

**File:** `src/evaluation/evaluator.py`

```bash
python main.py --mode compare
```

### Metrics Table (per model × per stock)
| Model | Accuracy | Macro F1 | Weighted F1 | Buy Precision | Sell Precision | Buy Recall |
|---|---|---|---|---|---|---|
| LightGBM (manual) | — | — | — | — | — | — |
| XGBoost (manual) | — | — | — | — | — | — |
| PyTorch LSTM (manual) | — | — | — | — | — | — |
| Chronos-2 (manual) | — | — | — | — | — | — |
| AutoGluon (manual) | — | — | — | — | — | — |
| **Stacking Ensemble (manual)** | — | — | — | — | — | — |
| **Stacking Ensemble (MLflow)** | — | — | — | — | — | — |

### Outputs
- Full metrics table printed to terminal
- Confusion matrix heatmaps saved as PNG
- `reports/comparison_<date>.csv` — machine-readable output

---

## 12. Phase 8 — Drift Monitoring

**File:** `src/evaluation/monitoring.py`  
**Library:** Evidently AI

```bash
python main.py --mode monitor
```

### What is Monitored

| Drift Type | Description | Alert Threshold |
|---|---|---|
| **Data drift** | Distribution of technical indicator features vs training baseline | PSI > 0.2 |
| **Target drift** | Distribution of actual next-day returns vs training period | KS test p < 0.05 |
| **Prediction drift** | Distribution of model output classes vs baseline | PSI > 0.2 |

### Output
- `reports/drift/<date>_drift_report.html` — interactive Evidently AI HTML report
- Console alert if any drift threshold exceeded
- Drift status accessible via `GET /drift` API endpoint

### Recommended Schedule
Run nightly at 2am after market close (configurable in `config.yaml`).  
If drift detected → trigger `POST /retrain` to kick off background retraining.

---

## 13. Phase 9 — FastAPI REST Service

**File:** `src/api/app.py`  
**Libraries:** `fastapi`, `uvicorn`, `pydantic v2`

### Endpoints

| Method | Endpoint | Description | Response |
|---|---|---|---|
| GET | `/health` | Service health + loaded model versions | `{status, models, uptime}` |
| GET | `/models` | All registered model versions + metrics | List of model info |
| POST | `/predict` | Single stock prediction | Full prediction response |
| POST | `/predict/batch` | Multiple symbols | List of predictions |
| GET | `/recommendation/{symbol}` | Latest signal + explanation | Full prediction + NL explanation |
| GET | `/sentiment/{symbol}` | Live FinBERT sentiment | `{score, label, confidence, news_count}` |
| POST | `/ask` | Natural language query | `{answer, sources}` |
| GET | `/comparison` | All models performance | Metrics table (JSON) |
| GET | `/drift` | Latest drift report status | `{status, drifted_features, report_url}` |
| POST | `/retrain` | Trigger background retraining | `{job_id, status}` |

### Full Prediction Response Schema
```json
{
  "symbol": "RELIANCE.NS",
  "signal": "Strong Buy",
  "signal_class": 4,
  "confidence": 0.87,
  "probabilities": {
    "strong_sell": 0.02,
    "sell": 0.03,
    "hold": 0.05,
    "buy": 0.16,
    "strong_buy": 0.74
  },
  "model_used": "stacking_ensemble_mlflow_v3",
  "sentiment": {
    "score": 0.72,
    "label": "positive",
    "confidence": 0.89,
    "news_count": 8
  },
  "top_features": {
    "RSI_14": 31.2,
    "MACD": 12.4,
    "volume_sma_ratio": 2.3,
    "sentiment_score": 0.72,
    "price_sma200_ratio": 0.88
  },
  "explanation": "RELIANCE.NS is showing strong buy signals...",
  "timestamp": "2026-05-25T10:00:00Z",
  "data_as_of": "2026-05-24"
}
```

---

## 14. Phase 10 — Docker Deployment

**Files:** `docker/Dockerfile.api`, `docker/Dockerfile.mlflow`, `docker/Dockerfile.ollama`, `docker/docker-compose.yaml`

### Services

| Service | Image | Port | Purpose |
|---|---|---|---|
| `mlflow` | Custom (Python + MLflow 3.12) | 5000 | Experiment tracking UI + artifact server |
| `api` | Custom (Python + FastAPI + FinBERT) | 8000 | REST API + FinBERT loaded at startup |
| `ollama` | `ollama/ollama` + Granite 4.1-3B | 11434 | IBM Granite 4.1-3B for NL explanations |

### docker-compose.yaml (simplified)
```yaml
services:
  mlflow:
    build:
      context: .
      dockerfile: docker/Dockerfile.mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./mlruns:/app/mlruns
      - ./data/news:/app/data/news
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - mlflow
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    entrypoint: ["/bin/sh", "-c", "ollama serve & sleep 5 && ollama pull granite4.1:3b && wait"]

volumes:
  ollama_models:
```

### Shared Volumes
- `./mlruns` → both `mlflow` and `api` containers (MLflow artifact access)
- `./models` → `api` container (load trained model artifacts)
- `./data/news` → `api` container (sentiment cache)
- `ollama_models` → `ollama` container (Granite model weights, persisted)

---

## 15. Phase 11 — CLI Entry Point

**File:** `main.py`  
**Library:** `argparse`

### Available Commands

```bash
# Data pipeline
python main.py --mode ingest                                          # Download/refresh data via OpenBB (all config stocks)
python main.py --mode ingest --symbols RELIANCE.NS TCS.NS             # Override: specific stocks only
python main.py --mode validate                                        # Run Pandera data quality checks
python main.py --mode features                                        # Engineer 64+ features + sentiment labels
python main.py --mode features --symbol RELIANCE.NS                   # Single stock feature engineering

# Training
python main.py --mode train --trainer manual                          # Train all 5 models, save locally
python main.py --mode train --trainer mlflow                          # Train + log everything to MLflow
python main.py --mode train --trainer manual --symbol TCS.NS          # Single stock
python main.py --mode train --trainer manual --symbols TCS.NS INFY.NS # Subset of stocks

# Evaluation
python main.py --mode compare                         # Full model comparison report
python main.py --mode monitor                         # Run Evidently AI drift check

# Prediction
python main.py --mode predict --symbol RELIANCE.NS    # Single stock prediction (CLI)

# Service
python main.py --mode serve                           # Start FastAPI server (uvicorn)

# Full pipelines (end-to-end)
python main.py --mode full-pipeline --trainer manual  # Ingest→Validate→Features→Train→Compare
python main.py --mode full-pipeline --trainer mlflow  # Same + MLflow tracking
```

---

## 16. Phase 12 — Configuration

**File:** `config/config.yaml`

```yaml
# --- Stocks ---
stocks:
  symbols:
    - RELIANCE.NS
    - TCS.NS
    - INFY.NS
    - HDFCBANK.NS
    - WIPRO.NS
  period: 5y
  interval: 1d
  provider: yfinance        # OpenBB provider backend

# --- Signal Thresholds ---
labels:
  strong_buy_threshold: 0.02      # +2%
  buy_threshold: 0.005            # +0.5%
  sell_threshold: -0.005          # -0.5%
  strong_sell_threshold: -0.02    # -2%

# --- Training ---
training:
  test_size: 0.2
  cv_folds: 5
  random_state: 42
  use_smote: true
  optuna_trials: 50
  sequence_length: 20             # LSTM lookback window (trading days)

# --- MLflow ---
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: stock-prediction-2026
  registry_model_name: "{symbol}-stock-predictor"

# --- SLM ---
slm:
  sentiment_model: ProsusAI/finbert
  explainer_model: granite4.1:3b
  ollama_base_url: http://localhost:11434
  watsonx_url: https://us-south.ml.cloud.ibm.com    # optional IBM cloud
  news_lookback_days: 7
  max_headlines_per_stock: 20
  cache_news: true

# --- Monitoring ---
monitoring:
  drift_threshold_psi: 0.2
  drift_check_schedule: "0 2 * * *"    # 2am daily (cron syntax)

# --- API ---
api:
  host: 0.0.0.0
  port: 8000
  reload: false

# --- Paths ---
paths:
  raw_data: data/raw
  processed_data: data/processed
  news_cache: data/news
  manual_models: models/manual
  reports: reports
```

### `.env.example` Contents

```bash
# ── MLflow ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5000

# ── Ollama / IBM Granite ──────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434

# ── IBM watsonx.ai (optional — cloud alternative to local Ollama) ─────
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_API_KEY=your-ibm-cloud-api-key-here
WATSONX_PROJECT_ID=your-project-id-here

# ── NewsAPI (optional — free tier 100 requests/day) ───────────────────
NEWSAPI_KEY=your-newsapi-key-here

# ── DVC remote (optional — for team model sharing) ───────────────────
DVC_REMOTE_URL=/tmp/dvc-cache
```

---

## 17. Technology Stack Summary

| Component | Library / Tool | Pinned Version | Notes |
|---|---|---|---|
| **Data ingestion** | `openbb` | 4.7.2 | yfinance as provider backend |
| **Technical indicators** | `pandas-ta-classic` | ≥ 0.3.14b (May 2026) | Active fork of pandas-ta; 192 indicators |
| **Gradient boosting** | `lightgbm` | 4.6.0 | Primary model |
| **Gradient boosting** | `xgboost` | 3.2.0 | Comparison model |
| **Deep learning** | `torch` (PyTorch) | 2.12.0 | Dominant 2026 DL framework (55% research papers) |
| **Time series foundation** | `chronos-forecasting` | 2.2.2 | Amazon Chronos-2 (`chronos-bolt-small`), Oct 2025 |
| **AutoML** | `autogluon` | 1.5.0 | AutoGluon tabular baseline |
| **HPO** | `optuna` | 4.8.0 | Hyperparameter optimization |
| **Class balancing** | `imbalanced-learn` | 0.14.1 | SMOTE (per-fold only) |
| **Experiment tracking** | `mlflow` | 3.12.0 | Redesigned UI, new `search_logged_models()` API |
| **SLM Sentiment** | `transformers` + FinBERT | 5.9.0 | ProsusAI/finbert; 97% accuracy |
| **SLM Explanation** | IBM Granite 4.1-3B via `ollama` | 0.6.2 | Local, 4GB RAM, Apache 2.0 |
| **IBM cloud (optional)** | `ibm-watsonx-ai` | 1.5.12 | Optional cloud deployment of Granite |
| **News scraping** | `ddgs` (formerly `duckduckgo-search`) | 8.1.1 | Package renamed in v8.x; free, no API key |
| **News scraping** | `newsapi-python` | optional | Free tier 100/day |
| **Data validation** | `pandera` | 0.31.1 | DataFrame schema validation |
| **API validation** | `pydantic` | 2.13.4 | Request/response models (v2) |
| **API framework** | `fastapi` | 0.136.3 | REST service |
| **API server** | `uvicorn` | 0.48.0 | ASGI server |
| **Drift monitoring** | `evidently` | 0.7.21 | Evidently AI — data/target/prediction drift |
| **Model & data versioning** | `dvc` | 3.67.1 | Git-integrated versioning for .pkl/.pt/parquet files |
| **Data processing** | `pandas` | 3.0.3 | Core data manipulation |
| **Data processing** | `numpy` | 2.4.6 | Numerical operations |
| **ML utilities** | `scikit-learn` | 1.8.0 | Preprocessing, metrics, TimeSeriesSplit |
| **Explainability** | `shap` | 0.51.0 | SHAP feature importance |
| **Visualization** | `matplotlib` | 3.10.9 | Plots and heatmaps |
| **Visualization** | `seaborn` | 0.13.2 | Statistical plots |
| **Containerization** | Docker + `docker-compose` | v5.1.4 | 3-service stack (no `version:` field — Compose v2+) |
| **Config** | `pyyaml` | 6.0.3 | YAML config parsing |
| **Config** | `python-dotenv` | 1.2.2 | .env file loading |

---

## 18. File Creation Order

Implementation should follow this order to respect dependencies:

```
Step 1:  requirements.txt
         config/config.yaml
         .env.example
         .gitignore                  (include: mlruns/, data/raw/, data/processed/, models/manual/*.pkl,
                                               models/manual/*.pt, __pycache__/, *.pyc, .env)
         src/__init__.py             (empty — marks src as package)
         src/data/__init__.py
         src/slm/__init__.py
         src/models/__init__.py
         src/training/__init__.py
         src/evaluation/__init__.py
         src/api/__init__.py
         dvc init                    (run once — creates .dvc/ folder)
         dvc remote add -d local_remote /tmp/dvc-cache   (local DVC cache)

Step 2:  src/data/ingestion.py          # OpenBB data download
Step 3:  src/data/validation.py         # Pandera schemas
Step 4:  src/data/features.py           # 60+ features + labels

Step 5:  src/slm/news_scraper.py        # Headline fetching
Step 6:  src/slm/sentiment.py           # FinBERT sentiment
Step 7:  src/slm/explainer.py           # Granite 4.1-3B explainer

Step 8:  src/models/lgbm_model.py
Step 9:  src/models/xgboost_model.py
Step 10: src/models/lstm_model.py       # PyTorch
Step 11: src/models/chronos_model.py    # Amazon Chronos-2
Step 12: src/models/autogluon_model.py
Step 13: src/models/ensemble_model.py

Step 14: src/training/manual_trainer.py
Step 15: src/training/mlflow_trainer.py

Step 16: src/evaluation/evaluator.py
Step 17: src/evaluation/monitoring.py   # Evidently AI

Step 18: src/api/app.py                 # FastAPI

Step 19: main.py                        # CLI entry point

Step 20: docker/Dockerfile.api
         docker/Dockerfile.mlflow
         docker/Dockerfile.ollama
         docker/docker-compose.yaml

Step 21: notebooks/exploration.ipynb
Step 22: README.md
```

---

## 19. Verification & Testing

### Step-by-Step Verification

```bash
# ── SETUP ────────────────────────────────────────────────────────────
pip install -r requirements.txt
ollama pull granite4.1:3b                      # One-time ~2GB download

# ── DVC SETUP (one-time) ─────────────────────────────────────────────
dvc init                                       # initialise DVC in project
dvc remote add -d local_remote /tmp/dvc-cache  # set local storage for model files
git add .dvc .dvcignore && git commit -m "init DVC"

# ── MANUAL PIPELINE ──────────────────────────────────────────────────
python main.py --mode ingest                   # Download 5y OHLCV data
python main.py --mode validate                 # Pandera schema checks
python main.py --mode features                 # Engineer 64 features
python main.py --mode train --trainer manual   # Train all 5 models
python main.py --mode compare                  # View comparison table

# ── MLFLOW PIPELINE ──────────────────────────────────────────────────
docker-compose -f docker/docker-compose.yaml up mlflow -d
open http://localhost:5000                     # MLflow UI
python main.py --mode train --trainer mlflow   # Train + log to MLflow

# ── MODEL COMPARISON ─────────────────────────────────────────────────
python main.py --mode compare
# Expected output: table with all 7 model variants × 6 metrics

# ── DRIFT MONITORING ─────────────────────────────────────────────────
python main.py --mode monitor
# Expected output: reports/drift/<date>_drift_report.html

# ── FULL DOCKER STACK ────────────────────────────────────────────────
docker-compose -f docker/docker-compose.yaml up
# 3 services: mlflow:5000, api:8000, ollama:11434

# ── API TESTING ──────────────────────────────────────────────────────
curl http://localhost:8000/health

curl http://localhost:8000/recommendation/RELIANCE.NS

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TCS.NS"}'

curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]}'

curl http://localhost:8000/sentiment/INFY.NS

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Which stocks should I buy today?"}'

curl http://localhost:8000/comparison
curl http://localhost:8000/drift

# ── MLFLOW UI ────────────────────────────────────────────────────────
open http://localhost:5000
# Explore: experiments, runs, parameters, metrics, confusion matrices, SHAP plots

# ── DVC VERSION MANAGEMENT ───────────────────────────────────────────
# View all tracked model versions
git log --oneline

# Check which .pkl files DVC is tracking
dvc status

# Roll back ALL models to a previous version
git checkout <commit-hash>
dvc checkout

# Roll back a SINGLE stock model only (e.g. HDFCBANK underperforming)
git checkout <commit-hash> -- models/manual/HDFCBANK.NS_lgbm.pkl.dvc
dvc checkout models/manual/HDFCBANK.NS_lgbm.pkl

# Push model files to remote (optional — for team sharing or backup)
dvc push
```

---

## 20. Accuracy Expectations

| Model | Expected Directional Accuracy | Notes |
|---|---|---|
| LightGBM alone | 65–72% | Strong baseline |
| XGBoost alone | 63–70% | Slightly below LightGBM |
| PyTorch LSTM | 60–68% | Captures sequence, misses tabular signals |
| Chronos-2 | 58–66% | Zero-shot, no domain tuning |
| AutoGluon | 66–74% | Often surprises; auto-stack |
| **Stacking Ensemble** | **70–82%** | Best of all models combined |
| + FinBERT features | **+3–8%** | News sentiment as extra signal |
| **Final target** | **73–85%** | With 5y data + 64 features + sentiment |

> **Honest note on 90%+ accuracy:** Achieving 90%+ multi-class accuracy on stock prediction is extremely difficult in practice — financial markets are partially efficient and contain irreducible randomness. The system is engineered to maximise accuracy using 2026 best-in-class methods. We prioritise **F1 score on Buy and Strong Buy signals**, which is the actionable metric for investment decisions.

---

*Plan version: 2026-05-27 | Stack validated with pinned versions (May 2026) | Ready for implementation*
