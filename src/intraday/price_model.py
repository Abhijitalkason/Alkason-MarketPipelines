"""Price model — LightGBM on price/momentum features (PLAN_v3 §10.1).

Hyperparameters tuned on an INNER split of the training fold only (never the
OOF/validation block — v1's contamination bug is structurally excluded here);
tune() is CALLED by trainer.py and backtester.fit_fold when training.tune is
on (M-5 — no dead control). Class weights for imbalance; no SMOTE anywhere.

Monotonicity (PLAN_v3 §10.1): constraints are an explicit dict below, one
comment per constraint; a feature absent from the dict is unconstrained —
absent justification = no constraint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import log_loss

from src.intraday import load_config
from src.intraday.features import PRICE_FEATURES

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    objective="binary", n_estimators=600, learning_rate=0.03,
    num_leaves=31, max_depth=6, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    class_weight="balanced", verbosity=-1,
)

# Economically-justified monotone constraints (+1 non-decreasing, −1 non-increasing).
MONOTONE = {
    # Participation above the time-of-day norm supports continuation; thin
    # follow-through does not become MORE likely as volume dries up.
    "cumvol_vs_norm": +1,
}


def _day_cut(X: pd.DataFrame, frac: float) -> int:
    """Chronological cut index at a DAY boundary when a date column exists
    (M-3 — a row-index cut can split a trading day across head/tail)."""
    if "date" in X.columns:
        days = sorted(pd.to_datetime(X["date"]).dt.date.unique())
        cut_day = days[max(1, int(len(days) * frac)) - 1]
        return int((pd.to_datetime(X["date"]).dt.date <= cut_day).sum())
    return int(len(X) * frac)


class PriceModel:
    FEATURES = PRICE_FEATURES + ["direction"]

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {}),
                       "random_state": load_config()["training"]["random_state"]}
        self.params["monotone_constraints"] = [MONOTONE.get(f, 0) for f in self.FEATURES]
        self.model: LGBMClassifier | None = None

    # ── tuning (inner split of the training fold ONLY) ────────────────

    def tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int | None = None) -> dict:
        import optuna

        n_trials = n_trials or load_config()["training"]["optuna_trials"]
        cut = _day_cut(X, 0.8)
        X_tr, X_in = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_in = y.iloc[:cut], y.iloc[cut:]

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
            return log_loss(y_in, m.predict_proba(X_in[self.FEATURES])[:, 1])

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=load_config()["training"]["random_state"]),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.params.update(study.best_params)
        logger.info("price model tuned: %s (logloss %.4f)", study.best_params, study.best_value)
        return study.best_params

    # ── fit / predict ─────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriceModel":
        cut = _day_cut(X, 0.9)  # small inner tail for early stopping (inside train fold)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(
            X.iloc[:cut][self.FEATURES], y.iloc[:cut],
            eval_set=[(X.iloc[cut:][self.FEATURES], y.iloc[cut:])],
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PriceModel not fitted")
        return self.model.predict_proba(X[self.FEATURES])[:, 1]

    def shap_top(self, row: pd.DataFrame, n: int = 10) -> dict[str, float]:
        """Top-n SHAP attributions for one decision row (H-12 — recorded in the
        fill journal line). TreeExplainer on LightGBM is cheap per row."""
        import shap

        if self.model is None:
            raise RuntimeError("PriceModel not fitted")
        explainer = shap.TreeExplainer(self.model)
        vals = explainer.shap_values(row[self.FEATURES])
        vals = vals[1] if isinstance(vals, list) else vals   # binary: class-1 column
        flat = np.asarray(vals).reshape(-1)[: len(self.FEATURES)]
        order = np.argsort(-np.abs(flat))[:n]
        return {self.FEATURES[i]: float(flat[i]) for i in order}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "PriceModel":
        blob = joblib.load(path)
        m = cls(blob["params"])
        m.model = blob["model"]
        return m
