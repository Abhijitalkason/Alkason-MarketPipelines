"""Adaptive Conformal Inference gate (PLAN_v3 Section 11).

fire ⟺ P_cal ≥ τ_t  AND  both models individually ≥ agreement floor.

τ adapts daily after labels mature:
    τ_{t+1} = τ_t + γ·(err_t − α)       err_t = 1 − win_rate among fired signals
(if realized error exceeds the α budget, the threshold RISES → fewer, better
signals; if under budget, it relaxes. Static thresholds provably lose coverage
under regime shift — adaptation is mandatory, not optional.)

State is persisted to models/v3/gate_state.json so live days resume correctly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)


@dataclass
class GateState:
    tau: float
    alpha: float          # error budget = 1 − target win rate
    gamma: float
    days_updated: int = 0


class ACIGate:
    def __init__(self, state: GateState | None = None):
        cfg = load_config()["gate"]
        self.floor = cfg["model_agreement_floor"]
        self.state = state or GateState(
            tau=cfg["fire_threshold_init"],
            alpha=1.0 - cfg["target_win_rate"],
            gamma=cfg["aci_gamma"],
        )

    # ── decision ──────────────────────────────────────────────────────

    def fire(self, p_cal: np.ndarray, p_price: np.ndarray, p_flow: np.ndarray) -> np.ndarray:
        """Boolean mask of signals to emit."""
        return (p_cal >= self.state.tau) & (p_price >= self.floor) & (p_flow >= self.floor)

    # ── daily adaptation ──────────────────────────────────────────────

    def update(self, fired_labels: list[int]) -> float:
        """Called after a day's fired signals mature. No signals → no update."""
        if fired_labels:
            err = 1.0 - float(np.mean(fired_labels))
            old = self.state.tau
            self.state.tau = float(np.clip(self.state.tau + self.state.gamma * (err - self.state.alpha),
                                           0.5, 0.995))
            self.state.days_updated += 1
            logger.info("ACI update: err=%.3f α=%.3f τ %.4f→%.4f", err, self.state.alpha, old, self.state.tau)
        return self.state.tau

    # ── persistence ───────────────────────────────────────────────────

    def _path(self) -> Path:
        return ROOT / load_config()["paths"]["state"] / "gate_state.json"

    def save(self) -> None:
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self.state), indent=2))

    @classmethod
    def load(cls) -> "ACIGate":
        gate = cls()
        p = gate._path()
        if p.exists():
            gate.state = GateState(**json.loads(p.read_text()))
        return gate
