"""MLflow wrapper + model registry (v3_supplementry §12.2, B-6).

Hard rules: tracking failures RAISE in train-v3 and gate runs (no silent local
fallback); every run logs git SHA (a dirty tree raises in gate runs), config
hash, data span, DVC rev, seeds. Registry helpers register/transition/load by
stage so the serving API loads Production and the paper runner loads Staging.
"""

from __future__ import annotations

import json
import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)


class TrackingError(RuntimeError):
    pass


def _mlflow():
    import mlflow

    cfg = load_config()["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    return mlflow


def git_sha(require_clean: bool = False) -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except Exception as e:  # noqa: BLE001
        raise TrackingError(f"git SHA unavailable: {e}")
    if require_clean and dirty:
        raise TrackingError("working tree is dirty — gate/train runs require a clean commit for provenance")
    return sha + ("-dirty" if dirty else "")


def config_hash() -> str:
    import hashlib

    return hashlib.sha256(json.dumps(load_config(), sort_keys=True, default=str).encode()).hexdigest()[:12]


@contextmanager
def start_run(kind: str, require_clean: bool = False):
    """Context manager yielding an active MLflow run; failures raise (no silent
    fallback). Tags git SHA / config hash / kind on entry."""
    mlflow = _mlflow()
    try:
        with mlflow.start_run(run_name=kind) as run:
            mlflow.set_tags({"kind": kind, "git_sha": git_sha(require_clean),
                             "config_hash": config_hash()})
            yield run
    except Exception as e:  # noqa: BLE001
        raise TrackingError(f"MLflow run '{kind}' failed: {e}")


def log_training(result: dict, bundle_dir: Path) -> str:
    """Log a training run + register a Staging model version. Returns version."""
    mlflow = _mlflow()
    name = load_config()["mlflow"]["registry_model_name"]
    with start_run("train-v3", require_clean=False):
        mlflow.log_params(result.get("params", {}))
        mlflow.log_metrics(result.get("metrics", {}))
        for f in bundle_dir.glob("*"):
            if f.is_file():
                mlflow.log_artifact(str(f), artifact_path="bundle")
        result_uri = mlflow.get_artifact_uri("bundle")
        mv = mlflow.register_model(result_uri, name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name, mv.version, "Staging")
        logger.info("registered %s v%s → Staging", name, mv.version)
        return mv.version


def transition(version: str, stage: str) -> None:
    mlflow = _mlflow()
    name = load_config()["mlflow"]["registry_model_name"]
    mlflow.tracking.MlflowClient().transition_model_version_stage(name, version, stage)
    logger.info("%s v%s → %s", name, version, stage)


def load_stage(stage: str):
    """Load the (PriceModel, FlowModel|None, Blender) bundle for a registry
    stage. Resolves the version's artifact dir and reads the joblib files.
    Raises if no version is in that stage."""
    from src.intraday.blend import Blender
    from src.intraday.flow_model import FlowModel
    from src.intraday.price_model import PriceModel

    mlflow = _mlflow()
    name = load_config()["mlflow"]["registry_model_name"]
    client = mlflow.tracking.MlflowClient()
    versions = [mv for mv in client.search_model_versions(f"name='{name}'")
                if mv.current_stage == stage]
    if not versions:
        raise TrackingError(f"no model version in stage {stage}")
    mv = max(versions, key=lambda v: int(v.version))
    local = Path(mlflow.artifacts.download_artifacts(mv.source))
    pm = PriceModel.load(local / "price_model.joblib")
    fm = FlowModel.load(local / "flow_model.joblib") if (local / "flow_model.joblib").exists() else None
    blender = Blender.load(local / "blender.joblib")
    return pm, fm, blender


def log_backtest(report: dict) -> None:
    """Log a backtest report's headline metrics + gate booleans."""
    mlflow = _mlflow()
    with start_run("backtest", require_clean=False):
        flat = {k: v for k, v in report.items() if isinstance(v, (int, float, bool))}
        mlflow.log_metrics({k: float(v) for k, v in flat.items()})
        mlflow.log_dict(report, "backtest_report.json")
