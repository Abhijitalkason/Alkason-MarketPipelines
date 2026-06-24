"""Production training pipeline (v3_supplementry §9, B-1).

train_production must, on the synthetic repo: assemble the dataset, decide flow
activation from the real-data share, fit + save the full bundle, and return a
populated TrainResult — all without MLflow/DVC (track=False, push_dvc=False).
We also pin the flow-activation toggle (_set_flow_active) round-trip in config.
Each assertion targets a control that would fail if removed.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.intraday as si
from src.intraday import ROOT, load_config
from src.intraday.trainer import TrainResult, train_production, _set_flow_active


def _state_dir():
    return ROOT / load_config()["paths"]["state"]


@pytest.fixture
def _stub_provenance(monkeypatch):
    """The model-card writer reads the git SHA via tracking.git_sha(cwd=ROOT);
    under the fixture ROOT is a throwaway tmp dir with no .git, so git_sha would
    raise. Provenance is not the control under test here — stub it so the bundle/
    card path (which IS under test) can run."""
    import src.intraday.tracking as tracking
    monkeypatch.setattr(tracking, "git_sha", lambda *a, **k: "test-sha")


def test_set_flow_active_round_trip(repo):
    """_set_flow_active flips the single config line and clears the cache so the
    next load_config sees it — the bundle/gate/features depend on a consistent
    flag."""
    assert load_config()["gate"]["flow_model_active"] is False
    _set_flow_active(True)
    assert "flow_model_active: true" in si.CONFIG_PATH.read_text()
    assert load_config()["gate"]["flow_model_active"] is True
    _set_flow_active(False)
    assert "flow_model_active: false" in si.CONFIG_PATH.read_text()
    assert load_config()["gate"]["flow_model_active"] is False


def test_train_production_round_trip(repo, _stub_provenance):
    """End-to-end on a small in-fixture window: produces a TrainResult and writes
    the full bundle (price model, blender, fresh gate, drift reference, card)."""
    days = repo["days"]
    end_day = days[160]
    # ~5 months of train window so build_dataset has enough sessions but the run
    # stays fast; train_months overrides cfg.training.train_min_months.
    result = train_production(end_day=end_day, train_months=5, tune=False,
                              push_dvc=False, track=False)

    assert isinstance(result, TrainResult)
    # TrainResult contract fields
    assert result.flow_active is False                 # no real flow data on disk
    assert result.flow_real_share == pytest.approx(0.0)
    assert result.n_rows > 0
    assert isinstance(result.data_span, tuple) and len(result.data_span) == 2
    assert result.data_span[1] == end_day
    assert result.calib_brier == pytest.approx(result.calib_brier)  # finite float
    assert 0.0 <= result.calib_brier <= 1.0
    assert result.version is None                       # track=False → no registry version

    sd = _state_dir()
    for artifact in ("price_model.joblib", "blender.joblib", "gate_state.json",
                     "feature_matrix.parquet", "model_card.md"):
        assert (sd / artifact).exists(), f"missing training artifact {artifact}"

    # flow channel inactive → no flow model file written
    assert not (sd / "flow_model.joblib").exists()

    # config flag must reflect the (inactive) decision
    assert load_config()["gate"]["flow_model_active"] is False

    # drift reference == the training matrix used (drift compares against this)
    ref = pd.read_parquet(sd / "feature_matrix.parquet")
    assert len(ref) == result.n_rows
    assert "label" in ref.columns


def test_train_production_no_track_no_version(repo, _stub_provenance):
    """track=False must skip MLflow entirely (no version, no network) — retraining
    in CI/tests can't reach a tracking server."""
    days = repo["days"]
    result = train_production(end_day=days[160], train_months=5, tune=False,
                              push_dvc=False, track=False)
    assert result.version is None
    # a freshly-trained model card records the (off) flow channel
    card = (_state_dir() / "model_card.md").read_text()
    assert "price-only" in card or "OFF" in card
