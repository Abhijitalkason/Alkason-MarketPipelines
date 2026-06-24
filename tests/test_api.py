"""v3 Serving API (PLAN_v3 §15; v3_supplementry §13).

Every endpoint requires X-API-Key == os.environ[cfg.api.api_key_env]; the app
refuses to start with no key configured (B-8). These tests pin: 401 without/with
a wrong key on EVERY endpoint, 200 with the correct key, and the health/ready
contract. Each would fail if require_key (the auth control) were removed.

app.CFG is captured at import time, so we import src.api.app FRESH after the repo
fixture has pointed config at tmp_path and a bundle exists on disk.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from src.intraday import load_config


API_KEY = "test-secret-key-123"

# (method, path) for every authenticated endpoint
GET_ENDPOINTS = ["/health", "/signals/today", "/signals/history", "/screen/today",
                 "/performance", "/drift"]


@pytest.fixture
def api(repo, monkeypatch):
    """A TestClient with a trained bundle on disk and the API key set."""
    # 1. bundle on disk so lifespan's local-fallback load succeeds
    import src.intraday.tracking as tracking

    def _no_registry(*a, **k):
        raise tracking.TrackingError("no MLflow server in tests — use local bundle")

    monkeypatch.setattr(tracking, "git_sha", lambda *a, **k: "test-sha")
    # lifespan tries the Production registry first; with no MLflow server the
    # client retries for minutes. Force the local fallback (model_source=local).
    monkeypatch.setattr(tracking, "load_stage", _no_registry)
    from src.intraday.trainer import train_production
    train_production(end_day=repo["days"][160], train_months=5, tune=False,
                     push_dvc=False, track=False)

    # 2. API key in the env under the configured var name
    key_env = load_config()["api"]["api_key_env"]
    monkeypatch.setenv(key_env, API_KEY)

    # 3. import the app FRESH (its module-level CFG must read patched config)
    for mod in [m for m in sys.modules if m.startswith("src.api")]:
        del sys.modules[mod]
    app_mod = importlib.import_module("src.api.app")
    # registry load will fail (no MLflow) → local fallback, which is what we want
    with TestClient(app_mod.app) as client:
        yield client, app_mod
    for mod in [m for m in sys.modules if m.startswith("src.api")]:
        sys.modules.pop(mod, None)


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_get_endpoint_requires_key(api, path):
    client, _ = api
    # no key
    assert client.get(path).status_code == 401
    # wrong key
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
    # correct key
    r = client.get(path, headers={"X-API-Key": API_KEY})
    assert r.status_code == 200, f"{path} → {r.status_code}: {r.text}"


def test_post_backtest_requires_key(api):
    client, _ = api
    assert client.post("/backtest", params={"start": "2026-01-01", "end": "2026-01-02"}).status_code == 401
    assert client.post("/backtest", params={"start": "2026-01-01", "end": "2026-01-02"},
                       headers={"X-API-Key": "wrong"}).status_code == 401
    r = client.post("/backtest", params={"start": "2026-01-01", "end": "2026-01-02"},
                    headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_backtest_status_requires_key(api):
    client, _ = api
    # unknown job: 404 WITH key, but 401 WITHOUT key (auth runs before the handler)
    assert client.get("/backtest/nope").status_code == 401
    assert client.get("/backtest/nope", headers={"X-API-Key": API_KEY}).status_code == 404


def test_health_contract(api):
    client, _ = api
    r = client.get("/health", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    body = r.json()
    for field in ("status", "model_source", "model_age_min", "gate_tau",
                  "gate_days_updated", "halted", "journal_chain_ok", "uptime_seconds"):
        assert field in body, f"health missing {field}"
    assert body["status"] == "ok"
    assert body["halted"] is False
    assert body["model_source"] == "local:models/v3"   # registry unavailable → local fallback
    assert body["uptime_seconds"] >= 0.0


def test_app_refuses_to_start_without_key(repo, monkeypatch):
    """B-8: with no API key configured the app must refuse to start (lifespan
    raises). Proves the no-key startup guard."""
    import src.intraday.tracking as tracking
    monkeypatch.setattr(tracking, "git_sha", lambda *a, **k: "test-sha")
    from src.intraday.trainer import train_production
    train_production(end_day=repo["days"][160], train_months=5, tune=False,
                     push_dvc=False, track=False)

    key_env = load_config()["api"]["api_key_env"]
    monkeypatch.delenv(key_env, raising=False)
    for mod in [m for m in sys.modules if m.startswith("src.api")]:
        del sys.modules[mod]
    app_mod = importlib.import_module("src.api.app")
    with pytest.raises(RuntimeError):
        with TestClient(app_mod.app):
            pass
    for mod in [m for m in sys.modules if m.startswith("src.api")]:
        sys.modules.pop(mod, None)
