"""Pre-open regime veto (PLAN_v4 §7).

Turns the market-level regime features (regime_data) into two fire-time controls:
  - a market-direction *bias* (+1 risk-on / −1 risk-off / 0 neutral), and
  - a *veto* that suppresses signals fighting a strong, coherent regime, plus a
    day-level *no-trade* flag for hostile mornings (a VIX spike into a risk-off
    tape — the days a discretionary trader stays out).

Deliberately NOT wired through risk.is_blackout / events.csv: blackout days are
dropped from the labeled dataset (backtester.build_dataset), which would remove
the very rows we need to measure the veto's effect. The veto lives at fire time,
so vetoed candidates stay in the coverage denominator and every vetoed signal
can be scored against its counterfactual label (PLAN_v4 §2 verified fact).

Fail-open by construction: if the regime data is absent (present=False) there is
no veto and no bias — same philosophy as the _present flags in the feature
channels. `regime.active` gates the whole thing off until it has earned its place
(staged like gate.flow_model_active).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.intraday import load_config
from src.intraday.regime_data import regime_features

logger = logging.getLogger(__name__)


@dataclass
class RegimeVerdict:
    risk_score: float      # composite in ~[-1, 1]; >0 risk-on, <0 risk-off
    bias: int              # +1 / 0 / -1 (deadzoned sign of risk_score)
    no_trade: bool         # hostile/chaotic morning → stand down for the day
    present: bool          # enough regime data to judge; False ⇒ fail-open


def _defaults() -> dict:
    cfg = load_config()
    return cfg.get("regime", {}) or {}


def day_regime(day: date, decision_ts: pd.Timestamp | None = None) -> RegimeVerdict:
    """Compute the day's regime verdict from the pre-open global features."""
    r = _defaults()
    feats = regime_features(day, decision_ts)

    # directional components (present-weighted): overnight equities lead the tone;
    # INR depreciation and a VIX spike lean risk-off.
    def g(name: str) -> float | None:
        return feats[name] if feats.get(name + "_present", 0.0) >= 1.0 else None

    parts: list[tuple[float, float]] = []   # (signed_component, weight)
    w = r.get("weights", {}) or {}
    for name, weight, scale, sign in [
        ("spx_ret_1d", w.get("spx", 0.30), 0.01, +1),
        ("ndx_ret_1d", w.get("ndx", 0.25), 0.01, +1),
        ("asia_ret_1d", w.get("asia", 0.20), 0.01, +1),
        ("usdinr_ret_1d", w.get("usdinr", 0.15), 0.005, -1),  # INR up = risk-off
        ("india_vix_chg_5d", w.get("vix", 0.10), 0.10, -1),   # VIX up = risk-off
    ]:
        v = g(name)
        if v is not None:
            parts.append((sign * float(np.tanh(v / scale)), weight))

    present = len(parts) >= 2   # need at least a couple of real signals to judge
    if not present:
        return RegimeVerdict(risk_score=0.0, bias=0, no_trade=False, present=False)

    wsum = sum(wt for _, wt in parts) or 1.0
    risk_score = float(sum(c * wt for c, wt in parts) / wsum)

    dead = float(r.get("bias_deadzone", 0.10))
    bias = int(np.sign(risk_score)) if abs(risk_score) >= dead else 0

    # no-trade: a coherent risk-off tape with a rising-vol regime is the classic
    # "sit on your hands" morning.
    vix_chg = g("india_vix_chg_5d")
    chaotic = (vix_chg is not None and vix_chg >= float(r.get("vix_spike_5d", 0.20)))
    no_trade = (abs(risk_score) >= float(r.get("no_trade_abs_score", 0.75))
                and risk_score < 0 and chaotic)
    return RegimeVerdict(risk_score=risk_score, bias=bias, no_trade=no_trade, present=True)


def veto(day: date, direction: int, decision_ts: pd.Timestamp | None = None) -> tuple[bool, str]:
    """Should this (day, direction) signal be suppressed? Returns (veto?, reason).

    Fail-open when regime is inactive or data is absent. Suppresses when the
    candidate direction fights a strong regime bias, or on a no-trade morning."""
    r = _defaults()
    if not bool(r.get("active", False)):
        return False, "regime_inactive"
    v = day_regime(day, decision_ts)
    if not v.present:
        return False, "regime_absent"
    if v.no_trade:
        return True, "regime_no_trade"
    thr = float(r.get("direction_veto_threshold", 0.50))
    if v.bias != 0 and int(np.sign(direction)) == -v.bias and abs(v.risk_score) >= thr:
        return True, f"regime_direction_conflict(bias={v.bias:+d},score={v.risk_score:+.2f})"
    return False, "ok"
