# Stock Prediction System — Plain English Guide

> **Who is this for?** Anyone who wants to understand what this system does and how it works — no coding knowledge needed.

---

## The Big Picture

Think of this system like a **smart stock analyst** who:
1. Reads 5 years of stock market history every day
2. Reads today's financial news
3. Runs the data through multiple prediction models
4. Gives you a clear recommendation: **Strong Buy / Buy / Hold / Sell / Strong Sell**
5. Explains in plain English *why* it made that recommendation
6. Keeps learning and alerts you if market conditions change dramatically

The system works for 5 major Indian stocks: **Reliance, TCS, Infosys, HDFC Bank, and Wipro** — but you can add or remove any stock just by editing a single settings file.

---

## The 7 Layers — What Each One Does

---

### Layer 1 — Data Layer
*"Collecting and cleaning the raw ingredients"*

This is where the system gathers all the information it needs before making any predictions.

---

#### OpenBB SDK
**What it is:** The data collector.

**Simple explanation:** Imagine sending someone to the stock exchange every morning to bring back today's prices — Open price, High price, Low price, Close price, and how many shares were traded (Volume). OpenBB does this automatically for all 5 stocks, going back 5 years.

**Why not just use free tools?** The older free tool (yfinance) works but sometimes breaks or gets rate-limited. OpenBB is more reliable — it automatically retries if something fails, caches data so it doesn't re-download what it already has, and handles failures gracefully.

---

#### Pandera (Data Validator)
**What it is:** The quality checker.

**Simple explanation:** Imagine a quality control inspector at a factory. Before raw materials go into production, the inspector checks: *Are any items missing? Are the numbers sensible? Is the date order correct?* Pandera does the same for our stock data — if a stock price comes in as a negative number, or if an entire day of data is missing, Pandera catches it and raises an alarm before it corrupts our predictions.

---

#### pandas-ta-classic (Technical Indicators)
**What it is:** The pattern calculator.

**Simple explanation:** Stock traders have invented hundreds of mathematical formulas over decades to spot patterns in price charts — things like "Is this stock oversold?" or "Is momentum building?". This library calculates all those formulas automatically. Instead of one data point (today's price), we end up with **60+ calculated signals** like RSI, MACD, Bollinger Bands, etc. These signals are the real fuel for our prediction models.

Think of it like a doctor who doesn't just look at your temperature — they also check blood pressure, heart rate, oxygen levels, and dozens of other indicators to form a complete picture.

---

### Layer 2 — SLM (Small Language Model) Layer
*"Reading the news and making sense of it"*

This layer adds intelligence that pure number-crunching cannot — it understands language and news.

---

#### News Scraper
**What it is:** The headline collector.

**Simple explanation:** Every day, this tool searches the internet for recent news about each stock — from financial news sites, Google News, and DuckDuckGo. It collects headlines from the past 7 days and saves them locally so it doesn't have to re-fetch the same news again.

**Important note:** For old historical data (e.g., what was the news about Reliance in 2021?), we simply cannot get that from the internet today. So for those old dates, we tell the system "no news available" and it treats the sentiment as neutral. Only recent data gets real news.

---

#### FinBERT (Sentiment Analyser)
**What it is:** The news mood detector.

**Simple explanation:** FinBERT is a small AI model trained specifically on millions of financial documents — annual reports, earnings calls, analyst notes. It reads each news headline and answers one question: *Is this headline positive, negative, or neutral for the stock?*

For example:
- *"Reliance reports record profits"* → **Positive** (score: +1)
- *"TCS faces regulatory probe"* → **Negative** (score: -1)
- *"Infosys maintains guidance"* → **Neutral** (score: 0)

It processes all headlines for a stock and produces 4 numbers that get added to the prediction model:
| Number | What it means |
|---|---|
| Sentiment Score | Overall mood of today's news (-1 to +1) |
| Sentiment Confidence | How sure the model is |
| News Count | How many headlines found in last 7 days |
| Sentiment Trend | Is sentiment improving or getting worse over last 3 days? |

**Why FinBERT specifically?** It achieves 97% accuracy on financial text — higher than general-purpose AI models, because it was trained only on financial language.

---

#### IBM Granite 4.1-3B (Explainer)
**What it is:** The plain-English writer.

**Simple explanation:** After the prediction models say "Strong Buy", most people want to know *why*. IBM Granite is a small but powerful language model (released April 2026) that reads the key signals behind the prediction and writes a 2–3 sentence human-readable explanation.

Input it receives:
```
Stock: RELIANCE.NS | Signal: Strong Buy | Confidence: 87%
Key signals: RSI=31 (oversold), MACD bullish crossover,
volume 2.3× average, positive news sentiment
```

Output it writes:
> *"RELIANCE.NS is showing strong buy signals. The RSI at 31 indicates the stock is oversold and due for a bounce, while a bullish MACD crossover confirms building momentum. High trading volume and positive news sentiment further support this recommendation."*

**Why run it locally?** IBM Granite runs on your own computer via a tool called Ollama — no internet API needed, no cost per query, and your stock data never leaves your machine.

---

### Layer 3 — Model Layer
*"The five analysts who each have their own opinion"*

This is the brain of the system. Five different prediction models each look at the same data and give their own prediction. A final "judge" model combines all their views.

---

#### LightGBM
**What it is:** The fast, pattern-matching expert.

**Simple explanation:** Imagine an experienced trader who has studied thousands of charts and memorised which combinations of indicators lead to price rises or falls. LightGBM does exactly this — it builds a massive decision tree that says things like "If RSI < 35 AND MACD is positive AND volume is high THEN likely Strong Buy". It tries 50 different versions of itself (using a tool called Optuna) to find the best combination of rules.

It also tells you *which signals mattered most* using a technique called SHAP — so you know it's not a black box.

---

#### XGBoost
**What it is:** The second opinion.

**Simple explanation:** Similar to LightGBM but uses a slightly different mathematical approach. Having two similar-but-different models is valuable — when they agree, confidence is high. When they disagree, it signals the prediction might be uncertain. Think of it as getting a second doctor's opinion.

---

#### PyTorch LSTM
**What it is:** The memory specialist.

**Simple explanation:** LSTM stands for Long Short-Term Memory. Unlike the other models that look at each day's signals independently, LSTM looks at the *sequence* of the last 20 trading days together. It asks: "Considering the pattern of everything that's happened over the past 20 days — not just today — what is likely to happen tomorrow?"

Think of it like reading a story chapter by chapter rather than just looking at the last page. It's built in PyTorch, the most popular AI framework in 2026.

---

#### Amazon Chronos-2
**What it is:** The foundation model forecaster.

**Simple explanation:** Chronos-2 (released October 2025) is a special type of model that was pre-trained by Amazon on over 700 billion data points from time series across many industries. It has never seen our stock data before, but it already understands the *language of time series* deeply.

We give it the raw price history and it predicts tomorrow's price range. We then convert that forecast into a signal: if it predicts price will rise more than 0.5%, that's a Buy signal.

Think of it like hiring a world-class economist who studied global markets for 10 years — they can give you useful insights about your local stock market even though they've never specifically studied it before.

---

#### AutoGluon
**What it is:** The automated benchmark.

**Simple explanation:** AutoGluon is an AutoML tool — it automatically tries dozens of different model types (Random Forest, Neural Networks, Gradient Boosting, etc.), figures out which combination works best, and stacks them together. We don't use it as part of our main prediction — instead, we use it as a **standard to beat**. If our hand-crafted ensemble of LightGBM + XGBoost + LSTM + Chronos-2 can outperform AutoGluon, we know our system is genuinely well-built.

Think of it as the par score in golf — you need to beat it to prove your approach is worth it.

---

#### Stacking Ensemble (The Judge)
**What it is:** The final decision-maker that combines all opinions.

**Simple explanation:** Each of the 4 models above (LightGBM, XGBoost, LSTM, Chronos-2) gives its own probability for each signal. The Stacking Ensemble is a fifth model — a "judge" — that has learned from experience which of the four to trust more in which situation.

**The critical rule — Out-of-Fold (OOF) training:**
Here's the important part. We cannot let the judge watch a model perform on the *same* data the model was trained on — because every model looks brilliant on data it has already memorised. Instead:
- We split 5 years of training data into 5 time blocks (Block 1 = oldest, Block 5 = most recent)
- We train each model on blocks 1–4, then ask it to predict block 5 (data it has never seen)
- We do this 4 times, rolling forward, so every block gets predicted by models that never trained on it
- Only these "honest" predictions are used to teach the judge

This way, the judge learns which model is genuinely better at predicting the future — not which model best memorises the past.

---

### Layer 4 — Training Layer
*"Teaching the models — two ways"*

---

#### Manual Trainer
**What it is:** The quick, local training mode.

**Simple explanation:** Run one command and the system trains all models on your computer, saves the results as files, and prints a comparison table showing how well each model performed. No tracking, no logs — just train, evaluate, and see the numbers. Good for fast experimentation.

---

#### MLflow Trainer
**What it is:** The full, recorded training mode.

**Simple explanation:** Does everything the Manual Trainer does, but also records every single detail — what settings were used, what accuracy was achieved, which features mattered most, and saves the trained model to a registry. You can go back weeks later and see exactly how a model was trained, compare it to a newer version, and promote the best one to "production" status.

Think of it like the difference between cooking a meal without writing anything down vs. using a proper recipe card that documents every ingredient and step so you can recreate it perfectly later.

The MLflow web interface (at `http://localhost:5000`) gives you a dashboard where you can visually compare all your experiments.

---

#### DVC — Model & Data Version Control
**What it is:** The version history keeper for model files.

**Simple explanation:** Git is great for tracking code changes but it was never designed to store large binary files like trained model files (`.pkl`). DVC (Data Version Control) fills this gap — think of it as Git specifically for model files and datasets.

Every time you finish a training run and save a new `.pkl` file, DVC takes a snapshot of it. If a future model performs worse, you can go back to any previous snapshot with two simple commands — like Ctrl+Z but for trained models.

**What DVC tracks in this system:**
- All `.pkl` files — one per stock per model (e.g. `RELIANCE.NS_lgbm.pkl`)
- All `.pt` files — PyTorch LSTM models (e.g. `RELIANCE.NS_lstm.pt`)
- Processed data files — the feature-engineered Parquet files

**How rollback works:**
```
Training Run 1 (May 1)  → saved → DVC snapshot → Git commit "v1 baseline"
Training Run 2 (May 15) → saved → DVC snapshot → Git commit "v2 + sentiment"
Training Run 3 (May 27) → saved → DVC snapshot → Git commit "v3 + Chronos-2"

Something goes wrong with v3 → roll back:
  git checkout v2-commit
  dvc checkout
  → all .pkl files instantly restored to May 15 versions
```

**Does it need cloud?** No — works entirely on your local machine. Cloud storage is optional if you want to share model files with a team.

**Per-stock rollback:** Because each stock has its own `.pkl` file, you can roll back just one stock's model without affecting the others. For example, if HDFCBANK's new model performs poorly but RELIANCE's is fine — restore only HDFCBANK's `.pkl` to the previous version.

---

### Layer 5 — Evaluation & Monitoring Layer
*"Checking if predictions are still good over time"*

---

#### Evaluator
**What it is:** The report card.

**Simple explanation:** After training, the evaluator compares all models side-by-side — like a teacher grading multiple students on the same exam. It shows a table of how well each model performed (accuracy, how often Buy signals were correct, how often Sell signals were correct, etc.) and saves the comparison to a CSV file. It also draws confusion matrices — charts that show exactly where the model got confused (e.g., predicted Hold when it should have predicted Buy).

---

#### Evidently AI (Drift Monitor)
**What it is:** The early warning system.

**Simple explanation:** A model trained on 2021–2026 data might stop being accurate if market conditions change significantly — for example, if a new regulatory regime changes how Indian stocks behave. Evidently AI runs every night and compares today's incoming data against what the model was trained on. If it detects that the data has "drifted" significantly, it raises an alert and can trigger automatic retraining.

Think of it like a smoke detector — silent when everything is fine, but it alerts you the moment something is wrong.

---

### Layer 6 — API Layer
*"The window through which you talk to the system"*

---

#### FastAPI REST Service
**What it is:** The communication interface.

**Simple explanation:** FastAPI is a web service that gives you a set of web addresses (called endpoints) you can call to get predictions. You don't need to run any Python code yourself — you just send a request to a web address and get an answer back in a format any programming language can understand (called JSON).

**Key endpoints — what you can ask the system:**

| What you want | How you ask |
|---|---|
| Is the system running? | `GET /health` |
| What signal for Reliance today? | `GET /recommendation/RELIANCE.NS` |
| Predict a stock | `POST /predict` with `{"symbol": "TCS.NS"}` |
| Predict multiple stocks at once | `POST /predict/batch` |
| What's the news sentiment for Infosys? | `GET /sentiment/INFY.NS` |
| Ask in plain English | `POST /ask` with `{"query": "Which stocks should I buy today?"}` |
| How are all models performing? | `GET /comparison` |
| Is the model still accurate? | `GET /drift` |
| Retrain the models now | `POST /retrain` |

---

### Layer 7 — Infrastructure Layer
*"The building that houses everything"*

---

#### Docker Containers
**What it is:** Isolated, portable boxes that run each part of the system.

**Simple explanation:** Docker is like putting each part of your system into its own sealed lunchbox — so they don't interfere with each other and can run on any computer without worrying about installation problems.

This system runs 3 containers:

| Container | What's inside | Port |
|---|---|---|
| **mlflow** | The experiment tracking dashboard | 5000 |
| **api** | The FastAPI service + FinBERT model loaded | 8000 |
| **ollama** | IBM Granite 4.1-3B for writing explanations | 11434 |

All three share some folders:
- The `mlruns/` folder (MLflow's storage) is shared between the MLflow container and the API container so both can read experiment results
- The `models/` folder is shared so the API can load trained model files
- The `data/news/` folder is shared so cached news headlines are available to the API

**Starting everything:** One command starts all three containers together:
```bash
docker-compose up
```

---

## How All Layers Connect — The Full Journey

Here is the complete journey from raw data to a stock recommendation:

```
Step 1:  OpenBB downloads 5 years of OHLCV price data
              ↓
Step 2:  Pandera checks the data quality — no bad values
              ↓
Step 3:  pandas-ta-classic calculates 60+ technical indicators
              ↓
Step 4:  News Scraper fetches today's headlines for each stock
              ↓
Step 5:  FinBERT reads headlines → produces 4 sentiment features
              ↓
Step 6:  All 64 features combined into one dataset per stock
              ↓
Step 7:  LightGBM, XGBoost, LSTM, Chronos-2 each make a prediction
              ↓
Step 8:  Stacking Ensemble combines all 4 predictions → final signal
              ↓
Step 9:  IBM Granite writes a plain-English explanation of the signal
              ↓
Step 10: FastAPI serves the result to whoever asked (app, curl, browser)
              ↓
Step 11: Evidently AI runs nightly to make sure accuracy hasn't degraded
```

---

## What the Two Training Modes Mean

| | Manual Training | MLflow Training |
|---|---|---|
| **What it does** | Trains models and saves files locally | Trains models AND records everything |
| **Speed** | Faster (no logging overhead) | Slightly slower |
| **Best for** | Quick experiments, testing new ideas | Production runs, comparing versions |
| **Can you compare runs?** | No — you'd have to remember results yourself | Yes — full history in the MLflow dashboard |
| **Where models are saved** | `models/manual/` folder | MLflow Model Registry |

---

## Accuracy — What to Realistically Expect

Stock markets are inherently unpredictable. No system can predict them with 100% accuracy — if it could, everyone would use it and the market would adjust. Here is what this system realistically aims for:

| Model | Expected Accuracy |
|---|---|
| Any single model alone | 60–74% |
| Our full ensemble | 70–82% |
| With news sentiment added | 73–85% |

We focus on **getting Buy and Sell signals right** rather than raw accuracy — because a correct Buy signal that leads to profit is more valuable than being right about Hold signals.

---

## Quick Glossary

| Term | Plain English meaning |
|---|---|
| **OHLCV** | Open, High, Low, Close prices and Volume traded for a day |
| **Technical Indicator** | A mathematical formula applied to price history (e.g. RSI, MACD) |
| **Sentiment** | The emotional tone of news — positive, negative, or neutral |
| **Model** | A mathematical pattern-matcher trained on historical data |
| **Ensemble** | Combining multiple models so their combined view is more accurate than any single one |
| **OOF (Out-of-Fold)** | Testing a model only on data it was never trained on — ensures honest accuracy numbers |
| **Drift** | When real-world data starts looking significantly different from the training data |
| **MLflow** | A tool that records all training experiments so you can compare and reproduce them |
| **Docker** | A tool that packages software into portable containers that run anywhere |
| **DVC** | Data Version Control — like Git but for large model files (.pkl). Lets you roll back to any previous trained model |
| **API / Endpoint** | A web address you call to get data or trigger an action |
| **SHAP** | A technique that explains which factors most influenced a model's prediction |
| **SMOTE** | A technique that creates extra training examples for rare cases (e.g., Strong Sell days) to help the model learn them better |

---

## Technology Versions (May 2026)

All libraries are pinned to specific versions to ensure reproducible results. Key versions:

| What it does | Tool | Version |
|---|---|---|
| Stock data download | OpenBB SDK | 4.7.2 |
| Technical indicators | pandas-ta-classic | ≥ 0.3.14b |
| Primary prediction model | LightGBM | 4.6.0 |
| Secondary prediction model | XGBoost | 3.2.0 |
| Deep learning framework | PyTorch | 2.12.0 |
| Time series foundation model | Amazon Chronos-2 | 2.2.2 |
| AutoML baseline | AutoGluon | 1.5.0 |
| Hyperparameter tuning | Optuna | 4.8.0 |
| Experiment tracking | MLflow | 3.12.0 |
| News sentiment AI | FinBERT (via Transformers) | 5.9.0 |
| Plain-English explanations | IBM Granite 4.1-3B (via Ollama) | 0.6.2 |
| News scraper | ddgs (formerly duckduckgo-search) | 8.1.1 |
| Data quality checks | Pandera | 0.31.1 |
| API framework | FastAPI | 0.136.3 |
| Drift monitoring | Evidently AI | 0.7.21 |
| Model versioning | DVC | 3.67.1 |
| Containerisation | Docker Compose | v5.1.4 |

---

*Document version: 2026-05-27 | Stack versions verified May 2026 | Intended audience: Non-technical stakeholders, new team members, management*
