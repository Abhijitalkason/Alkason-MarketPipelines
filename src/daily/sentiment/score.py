"""Score raw headlines → sentiment (FinBERT or an LLM). Scoring runs async to
collection; each scored row keeps the headline's original release_ts (the scorer's
timestamp is NOT the PIT key — when the news became public is).

FinBERT is the default; absence raises with install guidance (scoring is a Phase-5
step, not needed for the factor v1).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.daily import now_ist
from src.daily.sentiment.archive import append_scored, read_raw, read_scored

logger = logging.getLogger(__name__)


def _finbert_scorer():
    try:
        from transformers import pipeline
    except ImportError:
        raise RuntimeError(
            "transformers not installed — `pip install transformers torch` for FinBERT scoring "
            "(Phase 5; the factor v1 does not need it)"
        )
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def score_pending(model: str = "finbert") -> dict:
    """Score raw headlines not yet in the scored archive. Returns counts."""
    raw = read_raw()
    if raw.empty:
        return {"status": "no_raw", "scored": 0}
    scored = read_scored()
    seen = set(zip(scored.get("symbol", []), scored.get("headline", [])))
    pending = raw[~raw.apply(lambda r: (r["symbol"], r["headline"]) in seen, axis=1)]
    if pending.empty:
        return {"status": "up_to_date", "scored": 0}

    pipe = _finbert_scorer()
    labels = pipe(list(pending["headline"].astype(str)))
    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    out = pending.copy()
    out["sentiment"] = [sign.get(x["label"].lower(), 0.0) * float(x["score"]) for x in labels]
    out["score_model"] = model
    out["scored_ts"] = now_ist()
    n = append_scored(out)
    logger.info("sentiment score: %d newly scored (%d total)", len(pending), n)
    return {"status": "ok", "scored": len(pending), "total": n}
