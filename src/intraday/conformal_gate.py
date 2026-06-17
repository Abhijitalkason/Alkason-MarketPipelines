"""Adaptive Conformal Inference gate (PLAN_v3 §11; v3_supplementry §6.4).

fire ⟺ P_cal ≥ τ_t  AND  each ACTIVE model individually ≥ agreement floor.

τ adapts daily after labels mature:
    τ_{t+1} = τ_t + γ·(err_t − α)       err_t = 1 − win_rate among fired signals
(if realized error exceeds the α budget, the threshold RISES → fewer, better
signals; if under budget, it relaxes. This is the threshold-space form of the
Gibbs–Candès update — PLAN_v3 §11 writes the level-space sign; for a fire
threshold this direction is the correct one. Static thresholds provably lose
coverage under regime shift — adaptation is mandatory.)

State persists to models/v3/gate_state.json (resume point); every save also
APPENDS to gate_state_history.jsonl (G.1) — τ's trajectory is evidence.

Deviation from PLAN_v3 §11/§17, recorded per v3_supplementry §17: mapie is not
used — this custom ACI loop is the whole requirement and keeps the dependency
list honest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.intraday import ROOT, load_config, now_ist

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
        self.flow_active = bool(cfg["flow_model_active"])
        self.state = state or GateState(
            tau=cfg["fire_threshold_init"],
            alpha=1.0 - cfg["target_win_rate"],
            gamma=cfg["aci_gamma"],
        )

    # ── decision ──────────────────────────────────────────────────────

    def fire(self, p_cal: np.ndarray, p_price: np.ndarray,
             p_flow: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask of signals to emit. The agreement floor applies to
        every ACTIVE model; with the flow channel off, p_price only (§6.3)."""
        mask = (np.asarray(p_cal) >= self.state.tau) & (np.asarray(p_price) >= self.floor)
        if self.flow_active and p_flow is not None:
            mask &= np.asarray(p_flow) >= self.floor
        return mask

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
        hist = p.parent / "gate_state_history.jsonl"
        with open(hist, "a") as f:  # append-only τ trajectory (G.1)
            f.write(json.dumps({"ts": now_ist().isoformat(timespec="seconds"),
                                **asdict(self.state)}) + "\n")

    @classmethod
    def load(cls) -> "ACIGate":
        gate = cls()
        p = gate._path()
        if p.exists():
            gate.state = GateState(**json.loads(p.read_text()))
        return gate


def coverage_report(trades: pd.DataFrame, min_n: int = 30) -> dict:
    """Per-symbol realized win rate with exact (Clopper–Pearson) binomial 95% CI
    for symbols with ≥ min_n fired signals. Consumed by the backtester's
    fairness rules (§8.3) and /performance."""
    cfg = load_config()
    target = cfg["gate"]["target_win_rate"]
    out: dict[str, dict] = {}
    if trades.empty:
        return out
    for sym, g in trades.groupby("symbol"):
        n = len(g)
        if n < min_n:
            continue
        k = int(g["label"].sum())
        lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        out[str(sym)] = {
            "n": n, "win_rate": k / n,
            "ci_low": lo, "ci_high": hi,
            "below_target": hi < target,   # CI upper bound under target ⇒ exclude
        }
    return out
