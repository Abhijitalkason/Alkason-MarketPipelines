"""Blend + isotonic calibration (PLAN_v3 Section 11).

P_blend = w·P_price + (1−w)·P_flow   — w fit on OOF predictions by log-loss grid
P_cal   = isotonic(P_blend)          — fit on the rolling calibration window
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
    def __init__(self, weight: float | None = None):
        self.weight = weight if weight is not None else load_config()["training"]["blend_weight_init"]
        self.calibrator: IsotonicRegression | None = None

    def fit_weight(self, p_price: np.ndarray, p_flow: np.ndarray, y: np.ndarray) -> float:
        """Grid-search blend weight on OOF predictions (min log-loss)."""
        grid = np.arange(0.0, 1.01, 0.05)
        losses = [log_loss(y, np.clip(w * p_price + (1 - w) * p_flow, 1e-6, 1 - 1e-6)) for w in grid]
        self.weight = float(grid[int(np.argmin(losses))])
        logger.info("blend weight fit on OOF: w=%.2f (logloss %.4f)", self.weight, min(losses))
        return self.weight

    def blend(self, p_price: np.ndarray, p_flow: np.ndarray) -> np.ndarray:
        return self.weight * p_price + (1 - self.weight) * p_flow

    def fit_calibration(self, p_blend: np.ndarray, y: np.ndarray) -> None:
        """Isotonic on the calibration window (OOF / trailing 60 days)."""
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(p_blend, y)

    def calibrated(self, p_price: np.ndarray, p_flow: np.ndarray) -> np.ndarray:
        p = self.blend(p_price, p_flow)
        if self.calibrator is None:
            raise RuntimeError("Blender calibration not fitted")
        return self.calibrator.predict(p)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weight": self.weight, "calibrator": self.calibrator}, path)

    @classmethod
    def load(cls, path: Path) -> "Blender":
        blob = joblib.load(path)
        b = cls(blob["weight"])
        b.calibrator = blob["calibrator"]
        return b
