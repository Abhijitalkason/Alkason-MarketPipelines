"""Granite narration of FIRED intraday signals (PLAN_v3 §15; v3_supplementry §14).

explain_signal renders a recorded SHAP attribution + setup into a one-paragraph
plain-English explanation via Ollama (IBM Granite). It runs ASYNCHRONOUSLY
post-fire (background task), never on the decision path, and only for signals
that actually fired. Graceful template fallback when Ollama is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _call_ollama(prompt: str, model: str, base_url: str) -> Optional[str]:
    try:
        import ollama
        client = ollama.Client(host=base_url)
        resp = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.3, "num_predict": 160})
        return resp.message.content.strip()
    except Exception as e:  # noqa: BLE001 — explanation is UX, never gating
        logger.warning("Ollama call failed (%s): %s", model, e)
        return None


def _template(fill: dict) -> str:
    sym = fill.get("symbol", "?")
    direction = "long" if fill.get("direction", 1) == 1 else "short"
    p = fill.get("p_cal", 0.0)
    shap = fill.get("shap_top", {})
    drivers = ", ".join(f"{k.replace('_', ' ')}={v:+.3f}" for k, v in list(shap.items())[:3])
    return (f"{sym}: fired {direction} at calibrated P(win)={p:.1%}. "
            f"Top drivers: {drivers}. Entry {fill.get('entry')}, qty {fill.get('qty')}. "
            f"[Install Ollama + granite4.1:3b for richer narration.]")


def explain_signal(fill: dict) -> str:
    """fill: a journal `fill` payload (symbol, direction, entry, qty, p_cal,
    shap_top). Returns a one-paragraph explanation."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    shap = fill.get("shap_top", {})
    drivers = "\n".join(f"  - {k.replace('_', ' ')} = {v:+.4f}" for k, v in list(shap.items())[:6])
    direction = "LONG" if fill.get("direction", 1) == 1 else "SHORT"
    prompt = (
        "You are an intraday trading analyst. In 2-3 sentences, explain why this "
        "selective signal fired, for a professional reader. Be specific about the drivers.\n\n"
        f"Symbol: {fill.get('symbol')}\n"
        f"Direction: {direction}\n"
        f"Calibrated P(win): {fill.get('p_cal', 0):.1%}\n"
        f"Top SHAP drivers (signed contribution to P(win)):\n{drivers}\n\n"
        "Explanation:"
    )
    out = _call_ollama(prompt, "granite4.1:3b", base_url)
    return out or _template(fill)
