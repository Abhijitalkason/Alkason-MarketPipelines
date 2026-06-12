"""Flow model — CatBoost on positioning features only (PLAN_v3 Section 10.2).

Deliberately blind to price-path features: its value is decorrelation via a
different information channel (OI, basis, delivery %, pre-open imbalance,
FII/DII), not architecture novelty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.intraday import load_config
from src.intraday.features import FLOW_FEATURES

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    iterations=400, learning_rate=0.05, depth=4,
    loss_function="Logloss", auto_class_weights="Balanced",
    verbose=0, allow_writing_files=False,
)


class FlowModel:
    FEATURES = FLOW_FEATURES + ["direction"]

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {}),
                       "random_seed": load_config()["training"]["random_state"]}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FlowModel":
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X[self.FEATURES], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FlowModel not fitted")
        return self.model.predict_proba(X[self.FEATURES])[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "FlowModel":
        blob = joblib.load(path)
        m = cls(blob["params"])
        m.model = blob["model"]
        return m
