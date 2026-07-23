"""Daily direction model — LightGBM on the daily feature surface (mirrors
src/intraday/price_model.py). Predicts P(up) directly; there is no pre-chosen
`direction` feature (the label IS the direction at the daily horizon).

Hyperparameters tune on an INNER split of the training fold only (never the OOF
block — the v1 contamination bug stays structurally excluded). Class weights for
imbalance; no SMOTE. shap_top powers the listing "why".
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import log_loss

from src.daily import load_daily_config
from src.daily.features import FEATURE_ORDER

logger = logging.getLogger(__name__)

# No economically-justified monotone constraints at the daily direction horizon
# (absent justification = no constraint, same discipline as the intraday model).
MONOTONE: dict[str, int] = {}


def _default_params() -> dict:
    m = load_daily_config()["model"]
    return dict(
        objective="binary", n_estimators=m["n_estimators"], learning_rate=m["learning_rate"],
        num_leaves=m["num_leaves"], max_depth=m["max_depth"], min_child_samples=m["min_child_samples"],
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        class_weight="balanced", verbosity=-1,
    )


def _day_cut(X: pd.DataFrame, frac: float) -> int:
    """Chronological cut at a DAY boundary (a daily row is one day, so this also
    keeps the split honest if a symbol appears multiple times per day)."""
    if "date" in X.columns:
        days = sorted(pd.to_datetime(X["date"]).dt.date.unique())
        cut_day = days[max(1, int(len(days) * frac)) - 1]
        return int((pd.to_datetime(X["date"]).dt.date <= cut_day).sum())
    return int(len(X) * frac)


class DailyModel:
    FEATURES = FEATURE_ORDER

    def __init__(self, params: dict | None = None):
        self.params = {**_default_params(), **(params or {}),
                       "random_state": load_daily_config()["training"]["random_state"]}
        self.params["monotone_constraints"] = [MONOTONE.get(f, 0) for f in self.FEATURES]
        self.model: LGBMClassifier | None = None

    def tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int | None = None) -> dict:
        import optuna

        n_trials = n_trials or load_daily_config()["training"]["optuna_trials"]
        cut = _day_cut(X, 0.8)
        X_tr, X_in = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_in = y.iloc[:cut], y.iloc[cut:]
        if y_tr.nunique() < 2 or y_in.nunique() < 2:
            logger.warning("daily tune skipped: single-class inner split")
            return {}

        def objective(trial: "optuna.Trial") -> float:
            p = {
                **self.params,
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            }
            m = LGBMClassifier(**p)
            m.fit(X_tr[self.FEATURES], y_tr, eval_set=[(X_in[self.FEATURES], y_in)],
                  callbacks=[early_stopping(50), log_evaluation(0)])
            return log_loss(y_in, m.predict_proba(X_in[self.FEATURES])[:, 1], labels=[0, 1])

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=load_daily_config()["training"]["random_state"]),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.params.update(study.best_params)
        logger.info("daily model tuned: %s (logloss %.4f)", study.best_params, study.best_value)
        return study.best_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DailyModel":
        cut = _day_cut(X, 0.9)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(
            X.iloc[:cut][self.FEATURES], y.iloc[:cut],
            eval_set=[(X.iloc[cut:][self.FEATURES], y.iloc[cut:])],
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DailyModel not fitted")
        return self.model.predict_proba(X[self.FEATURES])[:, 1]

    def shap_top(self, row: pd.DataFrame, n: int = 3) -> dict[str, float]:
        """Top-n SHAP attributions for one row — the listing 'why'."""
        import shap

        if self.model is None:
            raise RuntimeError("DailyModel not fitted")
        explainer = shap.TreeExplainer(self.model)
        vals = explainer.shap_values(row[self.FEATURES])
        vals = vals[1] if isinstance(vals, list) else vals
        flat = np.asarray(vals).reshape(-1)[: len(self.FEATURES)]
        order = np.argsort(-np.abs(flat))[:n]
        return {self.FEATURES[i]: float(flat[i]) for i in order}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "DailyModel":
        blob = joblib.load(path)
        m = cls(blob["params"])
        m.model = blob["model"]
        return m
