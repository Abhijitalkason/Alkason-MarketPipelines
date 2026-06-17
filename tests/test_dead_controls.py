"""Dead-control + dead-import sweeps (v3_supplementry §0 #2, §16 step 3).

These tests fail if a load-bearing control loses its live call site, or if a
deleted v1 module is still referenced — Gate 0 cannot pass otherwise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _grep_files(term: str, *paths: str) -> list[str]:
    out = subprocess.run(["grep", "-rl", "--include=*.py", term, *paths], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return out


def test_controls_have_external_call_sites():
    controls = {
        "check_kill_switches": "src/intraday/risk.py",
        "cross_check_bhavcopy": "src/intraday/bars.py",
        "run_check": "src/intraday/drift.py",
        "reconcile_day": "src/intraday/data_feed.py",
        "tune": "src/intraday/price_model.py",
        "check_schema": "src/intraday/features.py",
    }
    for name, owner in controls.items():
        hits = _grep_files(name, "src/intraday", "main.py", "src/api")
        external = [h for h in hits if h != owner]
        assert external, f"{name} has no live call site outside {owner} — dead control"


def test_no_v1_module_references():
    dead = ["news_scraper", "sentiment", "xgboost_model", "lstm_model",
            "chronos_model", "autogluon_model", "ensemble_model",
            "manual_trainer", "mlflow_trainer"]
    src = ROOT / "src"
    for mod in dead:
        hits = _grep_files(mod, "src/intraday", "main.py", "src/api", "src/slm")
        assert not hits, f"deleted v1 module {mod} still referenced in {hits}"
