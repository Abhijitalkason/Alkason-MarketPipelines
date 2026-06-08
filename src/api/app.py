"""
FastAPI REST service — 10 endpoints for stock predictions, sentiment, and drift.
All request/response models use Pydantic v2.
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("API started — pre-warming model cache for configured symbols ...")
    for symbol in CFG["stocks"]["symbols"]:
        try:
            _load_ensemble(symbol)
            logger.info("  Loaded model for %s", symbol)
        except HTTPException:
            logger.warning("  No model for %s — run training first", symbol)
    yield


app = FastAPI(
    title="AI-MLOps Stock Prediction API",
    description="Production-grade NSE/BSE stock prediction with LightGBM, XGBoost, LSTM, Chronos-2 + FinBERT sentiment",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Config ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

CFG = _load_config()
SIGNAL_MAP = {4: "Strong Buy", 3: "Buy", 2: "Hold", 1: "Sell", 0: "Strong Sell"}

# ── Model Cache (lazy loaded) ─────────────────────────────────────────

_model_cache: Dict[str, object] = {}

def _load_ensemble(symbol: str):
    key = f"ensemble_{symbol}"
    if key not in _model_cache:
        path = Path(CFG["paths"]["manual_models"]) / f"{symbol}_ensemble.pkl"
        if not path.exists():
            raise HTTPException(404, detail=f"No trained model found for {symbol}. Run: python main.py --mode train --trainer manual")
        with open(path, "rb") as f:
            _model_cache[key] = pickle.load(f)
    return _model_cache[key]


def _get_latest_features(symbol: str) -> tuple:
    """Load the latest processed row for live prediction."""
    from src.data.features import load_processed
    df = load_processed(symbol)
    feature_cols = [c for c in df.columns if c != "label"]
    return df[feature_cols].values.astype(np.float32), feature_cols


# ── Pydantic v2 Models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    symbol: str = Field(..., description="NSE/BSE symbol, e.g. RELIANCE.NS")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        if not (v.endswith(".NS") or v.endswith(".BO")):
            v = v + ".NS"  # auto-append NSE suffix when omitted
        return v


class BatchPredictRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=20)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)


class SentimentResponse(BaseModel):
    symbol: str
    score: float
    label: str
    confidence: float
    news_count: int
    timestamp: str


class PredictionResponse(BaseModel):
    symbol: str
    signal: str
    signal_class: int
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    sentiment: Optional[SentimentResponse] = None
    top_features: Dict[str, float] = {}
    explanation: str = ""
    timestamp: str
    data_as_of: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    uptime_seconds: float
    timestamp: str


_start_time = datetime.now()

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        loaded_models=list(_model_cache.keys()),
        uptime_seconds=(datetime.now() - _start_time).total_seconds(),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/models", tags=["System"])
async def list_models():
    """List all trained model artifacts for all configured symbols."""
    model_dir = Path(CFG["paths"]["manual_models"])
    models = []
    for symbol in CFG["stocks"]["symbols"]:
        for kind in ["lgbm", "xgb", "lstm", "autogluon", "ensemble"]:
            path = model_dir / f"{symbol}_{kind}.pkl"
            pt_path = model_dir / f"{symbol}_{kind}.pt"
            p = path if path.exists() else (pt_path if pt_path.exists() else None)
            if p:
                models.append({"symbol": symbol, "model": kind, "path": str(p),
                                "size_mb": round(p.stat().st_size / 1e6, 2)})
    return {"models": models, "total": len(models)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    symbol = request.symbol
    model = _load_ensemble(symbol)

    X, _ = _get_latest_features(symbol)
    latest_X = X[-1:].reshape(1, -1)

    proba = model.predict_proba(latest_X)[0]
    signal_class = int(np.argmax(proba))
    confidence = float(proba[signal_class])

    # Top features
    try:
        top_feat = model.top_features(X[-50:], n=5)
    except Exception:
        top_feat = {}

    # Sentiment
    from src.slm.sentiment import get_live_sentiment
    sent = get_live_sentiment(symbol)

    # Explanation
    from src.slm.explainer import explain_prediction
    explanation = explain_prediction(
        symbol=symbol,
        signal_class=signal_class,
        confidence=confidence,
        top_features=top_feat,
        sentiment_score=sent.get("sentiment_score", 0.0),
        sentiment_label="positive" if sent.get("sentiment_score", 0) > 0.1 else
                         "negative" if sent.get("sentiment_score", 0) < -0.1 else "neutral",
    )

    from src.data.features import load_processed
    df = load_processed(symbol)

    return PredictionResponse(
        symbol=symbol,
        signal=SIGNAL_MAP[signal_class],
        signal_class=signal_class,
        confidence=round(confidence, 4),
        probabilities={
            "strong_sell": round(float(proba[0]), 4),
            "sell":        round(float(proba[1]), 4),
            "hold":        round(float(proba[2]), 4),
            "buy":         round(float(proba[3]), 4),
            "strong_buy":  round(float(proba[4]), 4),
        },
        model_used="stacking_ensemble_manual",
        sentiment=SentimentResponse(
            symbol=symbol,
            score=round(sent.get("sentiment_score", 0.0), 4),
            label="positive" if sent.get("sentiment_score", 0) > 0.1 else
                  "negative" if sent.get("sentiment_score", 0) < -0.1 else "neutral",
            confidence=round(sent.get("sentiment_confidence", 0.5), 4),
            news_count=int(sent.get("news_count_7d", 0)),
            timestamp=datetime.now().isoformat(),
        ),
        top_features=top_feat,
        explanation=explanation,
        timestamp=datetime.now().isoformat(),
        data_as_of=str(df.index[-1].date()),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    results = []
    errors = []
    for symbol in request.symbols:
        try:
            resp = await predict(PredictRequest(symbol=symbol))
            results.append(resp)
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
    return {"predictions": results, "errors": errors, "total": len(results)}


@app.get("/recommendation/{symbol}", response_model=PredictionResponse, tags=["Prediction"])
async def recommendation(symbol: str):
    """Full recommendation with explanation — main endpoint for users."""
    return await predict(PredictRequest(symbol=symbol))


@app.get("/sentiment/{symbol}", response_model=SentimentResponse, tags=["SLM"])
async def sentiment(symbol: str):
    symbol = symbol.upper().strip()
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        symbol = symbol + ".NS"
    from src.slm.sentiment import get_live_sentiment
    sent = get_live_sentiment(symbol)
    return SentimentResponse(
        symbol=symbol,
        score=round(sent.get("sentiment_score", 0.0), 4),
        label="positive" if sent.get("sentiment_score", 0) > 0.1 else
              "negative" if sent.get("sentiment_score", 0) < -0.1 else "neutral",
        confidence=round(sent.get("sentiment_confidence", 0.5), 4),
        news_count=int(sent.get("news_count_7d", 0)),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/ask", tags=["SLM"])
async def ask(request: AskRequest):
    """Natural language query about stocks."""
    from src.slm.explainer import _call_ollama
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Get current predictions for all configured stocks
    predictions_summary = []
    for symbol in CFG["stocks"]["symbols"]:
        try:
            pred = await predict(PredictRequest(symbol=symbol))
            predictions_summary.append(
                f"{symbol}: {pred.signal} (conf={pred.confidence:.0%})"
            )
        except Exception:
            pass

    context = "\n".join(predictions_summary) if predictions_summary else "No predictions available."
    prompt = (
        f"Current stock predictions:\n{context}\n\n"
        f"User question: {request.query}\n\n"
        f"Answer concisely based on the data above:"
    )

    answer = _call_ollama(prompt, "granite4.1:3b", base_url)
    if not answer:
        answer = f"Based on current signals: {context}"

    return {"query": request.query, "answer": answer, "timestamp": datetime.now().isoformat()}


@app.get("/comparison", tags=["Evaluation"])
async def comparison():
    """Side-by-side comparison table of all trained models."""
    from src.evaluation.evaluator import compare_all
    try:
        df = compare_all(include_mlflow=False)
        return {"comparison": df.to_dict(orient="records"), "generated_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, detail=f"Comparison failed: {e}")


@app.get("/drift", tags=["Monitoring"])
async def drift():
    """Latest drift report status for all configured symbols."""
    from src.evaluation.monitoring import get_latest_drift_status
    return {
        "symbols": {s: get_latest_drift_status(s) for s in CFG["stocks"]["symbols"]},
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/retrain", tags=["Training"])
async def retrain(background_tasks: BackgroundTasks, trainer: str = "manual"):
    """Trigger background retraining for all symbols."""
    if trainer == "mlflow":
        from src.training.mlflow_trainer import train_all_mlflow
        background_tasks.add_task(train_all_mlflow)
    else:
        from src.training.manual_trainer import train_all
        background_tasks.add_task(train_all)

    return {
        "status": "retraining_started",
        "trainer": trainer,
        "message": "Retraining running in background. Check /models when complete.",
        "timestamp": datetime.now().isoformat(),
    }
