"""Train a deployable daily model bundle per horizon (the production artifact the
screener/listing serves). Same fit protocol as a backtest fold — head/calib-tail
split → fit on head → isotonic calibrator on the OOF tail → refit on the FULL
window with frozen params — so train==serve by construction.

Bundle: models/daily/<horizon>/{model.joblib, blender.joblib, meta.json}. No
auto-promotion / no order side effects — this only writes model files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.daily import daily_path, load_daily_config, now_ist, today_ist
from src.daily.backtester import _inner_split, build_dataset
from src.daily.features import SCHEMA_VERSION
from src.daily.model import DailyModel
from src.intraday.blend import Blender

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    horizon: str
    trained_through: str
    n_rows: int
    schema_version: str
    bundle_dir: str


def _bundle_dir(horizon: str) -> Path:
    d = daily_path("state") / horizon
    d.mkdir(parents=True, exist_ok=True)
    return d


def _feature_flags() -> dict:
    """The feature-block toggles that change the feature semantics (train==serve)."""
    f = load_daily_config()["features"]
    return {k: bool(f.get(k)) for k in
            ("fno_enabled", "global_enabled", "macro_enabled", "sentiment_enabled")}


def train_production(horizon: str = "swing_1_5d", end_day: date | None = None,
                     train_months: int | None = None, tune: bool | None = None) -> TrainResult:
    cfg = load_daily_config()
    end_day = end_day or today_ist()
    months = train_months or (cfg["training"]["train_min_months"] + cfg["training"]["test_fold_months"] * 6)
    start = end_day - timedelta(days=int(months * 30.44))
    tune = cfg["training"]["tune"] if tune is None else tune

    data = build_dataset(start, end_day, horizon)
    head, tail = _inner_split(data)
    if head.empty or tail.empty:
        raise RuntimeError("inner split empty — need more history for production training")

    pm = DailyModel()
    if tune:
        pm.tune(head, head["label"])
    pm.fit(head, head["label"])
    blender = Blender(flow_active=False)
    blender.fit_calibration(pm.predict_proba(tail), tail["label"].values)
    pm = DailyModel(pm.params).fit(data, data["label"])     # full-window refit, frozen params

    bd = _bundle_dir(horizon)
    pm.save(bd / "model.joblib")
    blender.save(bd / "blender.joblib")
    meta = TrainResult(horizon=horizon, trained_through=str(end_day), n_rows=len(data),
                       schema_version=SCHEMA_VERSION, bundle_dir=str(bd))
    # record the feature flags the model was trained under — toggling e.g.
    # fno_enabled between train and serve silently changes the feature semantics.
    (bd / "meta.json").write_text(json.dumps(
        {**asdict(meta), "trained_at": now_ist().isoformat(),
         "feature_flags": _feature_flags()}, indent=2, default=str))
    logger.info("daily production bundle (%s) → %s (%d rows through %s)",
                horizon, bd, len(data), end_day)
    return meta


def load_bundle(horizon: str = "swing_1_5d") -> tuple[DailyModel, Blender, dict]:
    bd = _bundle_dir(horizon)
    mp = bd / "model.joblib"
    if not mp.exists():
        raise FileNotFoundError(f"no daily bundle for {horizon} at {bd} — run train first")
    pm = DailyModel.load(mp)
    blender = Blender.load(bd / "blender.joblib")
    meta = json.loads((bd / "meta.json").read_text()) if (bd / "meta.json").exists() else {}
    trained_flags = meta.get("feature_flags")
    if trained_flags and trained_flags != _feature_flags():
        logger.warning(
            "daily bundle %s trained with feature_flags=%s but current config is %s — "
            "train/serve feature mismatch; retrain or restore the flags",
            horizon, trained_flags, _feature_flags())
    return pm, blender, meta
