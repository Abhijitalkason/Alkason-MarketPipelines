"""Blend + isotonic calibration (PLAN_v3 §11; v3_supplementry §6.3).

P_blend = w·P_price + (1−w)·P_flow   — w fit on OOF predictions by log-loss grid
P_cal   = isotonic(P_blend)          — fit on the day-boundary calibration tail

Staged flow activation (B-5): when flow_active is False the weight is PINNED
to 1.0, the calibrator is fit on price-model OOF alone, and p_flow is ignored
everywhere. The flow channel turns on when its data is real, not before.

Calibration-consistency rule (H-4): fit_calibration must only ever see
predictions from models that did NOT train on those rows — enforced by the
fit_fold protocol in backtester/trainer and unit-tested.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.intraday import load_config

logger = logging.getLogger(__name__)


class Blender:
    def __init__(self, weight: float | None = None, flow_active: bool | None = None):
        cfg = load_config()
        self.flow_active = (bool(cfg["gate"]["flow_model_active"])
                            if flow_active is None else bool(flow_active))
        if not self.flow_active:
            self.weight = 1.0
        else:
            self.weight = weight if weight is not None else cfg["training"]["blend_weight_init"]
        self.calibrator: IsotonicRegression | None = None

    def fit_weight(self, p_price: np.ndarray, p_flow: np.ndarray | None, y: np.ndarray) -> float:
        """Grid-search blend weight on OOF predictions (min log-loss).
        Flow inactive → pinned at 1.0, nothing to fit."""
        if not self.flow_active:
            self.weight = 1.0
            return self.weight
        if p_flow is None:
            raise ValueError("flow_active=True but p_flow is None")
        grid = np.arange(0.0, 1.01, 0.05)
        losses = [log_loss(y, np.clip(w * p_price + (1 - w) * p_flow, 1e-6, 1 - 1e-6)) for w in grid]
        self.weight = float(grid[int(np.argmin(losses))])
        logger.info("blend weight fit on OOF: w=%.2f (logloss %.4f)", self.weight, min(losses))
        return self.weight

    def blend(self, p_price: np.ndarray, p_flow: np.ndarray | None = None) -> np.ndarray:
        if not self.flow_active or p_flow is None:
            return np.asarray(p_price)
        return self.weight * p_price + (1 - self.weight) * p_flow

    def fit_calibration(self, p_blend: np.ndarray, y: np.ndarray) -> None:
        """Isotonic on the OOF calibration tail — never on in-fold predictions (H-4)."""
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(p_blend, y)

    def calibrated(self, p_price: np.ndarray, p_flow: np.ndarray | None = None) -> np.ndarray:
        p = self.blend(p_price, p_flow)
        if self.calibrator is None:
            raise RuntimeError("Blender calibration not fitted")
        return self.calibrator.predict(p)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weight": self.weight, "flow_active": self.flow_active,
                     "calibrator": self.calibrator}, path)

    @classmethod
    def load(cls, path: Path) -> "Blender":
        blob = joblib.load(path)
        b = cls(blob["weight"], flow_active=blob.get("flow_active", True))
        b.calibrator = blob["calibrator"]
        return b
