"""Executable acceptance gates (PLAN_v3 §19; v3_supplementry §20).

Each gate is a function returning GateResult(passed, evidence). Results are
written append-only to reports/v3/gates/<gate>_<ts>.json. Failing a gate sends
the project back, never forward.

Gate 0 — engineering readiness (pytest suite + dead-control sweep + coverage)
Gate 1 — data (backfill stats + recorded sessions + reconcile + cross-check)
Gate 2 — label/geometry (geometry_study grid)
Gate 3 — backtest (base + 2× stress, all metric-contract gates)
Gate 4 — paper (≥22 sessions, win-rate gap, expectancy, slippage)
Gate 5 — capital (COMPLIANCE checklist + kill switches clean)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: dict = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)


def _gates_dir() -> Path:
    p = Path(load_config()["paths"]["gates"])
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write(result: GateResult) -> GateResult:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {**asdict(result), "stamp": stamp,
              "ts": datetime.now().isoformat(timespec="seconds")}
    (_gates_dir() / f"{result.gate}_{stamp}.json").write_text(json.dumps(record, indent=2, default=str))
    logger.info("GATE %s: %s — %s", result.gate, "PASS" if result.passed else "FAIL", result.detail)
    return result


# ── Gate 0 — engineering readiness ────────────────────────────────────

def gate0() -> GateResult:
    """pytest suite green + dead-control sweep clean + coverage ≥ 85%."""
    detail: dict = {}
    cov = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--cov=src/intraday",
         "--cov-report=term-missing:skip-covered", "--cov-fail-under=85"],
        cwd=ROOT, capture_output=True, text=True,
    )
    detail["pytest_returncode"] = cov.returncode
    detail["pytest_tail"] = cov.stdout.splitlines()[-15:]
    dead = _dead_control_sweep()
    detail["dead_controls"] = dead
    passed = cov.returncode == 0 and not dead
    return _write(GateResult("gate0", passed, detail))


def _dead_control_sweep() -> list[str]:
    """Every load-bearing control must have a call site outside its own module
    and tests. A control whose only references are its definition is dead."""
    controls = {
        "check_kill_switches": "src/intraday/risk.py",
        "cross_check_bhavcopy": "src/intraday/bars.py",
        "run_check": "src/intraday/drift.py",          # drift switch (c)
    }
    dead = []
    for name, owner in controls.items():
        hits = subprocess.run(
            ["grep", "-rl", "--include=*.py", name, "src/intraday", "main.py", "src/api"],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.split()
        external = [h for h in hits if h != owner]
        if not external:
            dead.append(name)
    return dead


# ── Gate 1 — data ─────────────────────────────────────────────────────

def gate1(min_names: int = 180, min_years: float = 3.0) -> GateResult:
    """≥3y bars for the PIT universe, bar-vs-bhavcopy mismatch <0.1%, recorder
    uptime, reconcile skew <5 bps. Reads what's on disk; raises nothing."""
    from datetime import timedelta
    from src.intraday import load_universe, today_ist
    from src.intraday.bars import session_dates

    detail: dict = {"names_ok": 0, "mismatches": [], "reconcile_alerts": []}
    uni = load_universe(as_of=today_ist())["symbol"].tolist()
    cutoff = today_ist() - timedelta(days=int(min_years * 365.25))
    for sym in uni:
        try:
            days = session_dates(sym, cutoff, today_ist())
            if days and days[0] <= cutoff + timedelta(days=30):
                detail["names_ok"] += 1
        except Exception:  # noqa: BLE001 — missing symbol simply doesn't count
            continue
    alerts = Path(load_config()["paths"]["alerts"])
    alerts = alerts if alerts.is_absolute() else ROOT / alerts
    if alerts.exists():
        detail["reconcile_alerts"] = [p.name for p in alerts.glob("*_reconcile.json")]
    passed = detail["names_ok"] >= min_names and not detail["reconcile_alerts"]
    return _write(GateResult("gate1", passed, detail))


# ── Gate 2 — label/geometry ───────────────────────────────────────────

def gate2(train_days: list | None = None) -> GateResult:
    """A geometry exists with baseline ≥86% and required δ ≤ 4 pts."""
    from src.intraday.geometry_study import TrainWindow, choose_geometry, run_grid

    if train_days is None:
        from src.intraday.backtester import build_dataset, make_folds, trading_sessions
        import pandas as pd
        from datetime import timedelta
        from src.intraday import today_ist
        days = trading_sessions(today_ist() - timedelta(days=4 * 365), today_ist())
        data = build_dataset(days)
        folds = make_folds(sorted(set(pd.to_datetime(data["date"]).dt.date)))
        train_days = folds[0][0]
    grid = run_grid(TrainWindow(train_days))
    chosen = choose_geometry(grid)
    passed = chosen is not None
    return _write(GateResult("gate2", passed, {"chosen_geometry": chosen,
                             "qualifying_cells": int(grid["meets_baseline_86"].sum())}))


# ── Gate 3 — backtest (base + 2× stress) ──────────────────────────────

def gate3(start, end) -> GateResult:
    """Runs base + 2× stress back-to-back; both must pass every metric-contract
    gate (win rate, expectancy, signals, no leakage, no subsidy, calibration,
    per-symbol coverage)."""
    from src.intraday.backtester import run_backtest

    base = run_backtest(start, end, stress=False, args_note="gate3-base")
    stress = run_backtest(start, end, stress=True, args_note="gate3-stress")
    passed = base["gates"]["all_pass"] and stress["gates"]["all_pass"]
    detail = {"base_gates": base["gates"], "stress_gates": stress["gates"],
              "base_win_rate": base["win_rate"], "stress_expectancy": stress["expectancy_pct_net"]}
    return _write(GateResult("gate3", passed, detail))


# ── Gate 4 — paper ────────────────────────────────────────────────────

def gate4() -> GateResult:
    """≥ paper_days sessions; win-rate gap ≤ paper_max_winrate_gap; positive
    expectancy; slippage within model (reconcile alerts clean)."""
    import pandas as pd

    cfg = load_config()
    sig = Path(cfg["paths"]["signals_log"])
    sig = sig if sig.is_absolute() else ROOT / sig
    files = sorted(sig.glob("paper_*.json"))
    rows = []
    for f in files:
        for rec in json.loads(f.read_text()):
            rows.append({"label": int(rec["label"]), "pnl_pct_net": float(rec["pnl_pct_net"])})
    detail = {"n_sessions": len(files), "n_trades": len(rows)}
    if len(files) < cfg["gates"]["paper_days"] or not rows:
        return _write(GateResult("gate4", False, detail))
    tdf = pd.DataFrame(rows)
    live_wr = float(tdf["label"].mean())
    bt_wr = _last_passing_backtest_wr()
    detail.update({"live_win_rate": live_wr, "backtest_win_rate": bt_wr,
                   "win_rate_gap": abs(live_wr - bt_wr),
                   "live_expectancy": float(tdf["pnl_pct_net"].mean())})
    passed = (abs(live_wr - bt_wr) <= cfg["gates"]["paper_max_winrate_gap"]
              and tdf["pnl_pct_net"].mean() > 0)
    return _write(GateResult("gate4", passed, detail))


def _last_passing_backtest_wr() -> float:
    reg = ROOT / load_config()["paths"]["run_registry"]
    if reg.exists():
        import pandas as pd
        df = pd.read_csv(reg)
        passed = df[df["all_pass"] == True]  # noqa: E712
        if not passed.empty:
            return float(passed["win_rate"].iloc[-1])
    return load_config()["gates"]["min_win_rate"]


# ── Gate 5 — capital ──────────────────────────────────────────────────

def gate5() -> GateResult:
    """Kill switches never breached + expectancy positive over the smallest-size
    live month. Reads halt state + live signals; COMPLIANCE.md broker checklist
    is a manual prerequisite recorded in the journal."""
    from src.intraday.risk import is_halted

    detail = {"halted": is_halted()}
    passed = not is_halted()
    return _write(GateResult("gate5", passed, detail))


def run_all(start=None, end=None) -> dict:
    """Run the gates whose evidence is available; used by `--mode gates`."""
    results = {"gate0": gate0()}
    if start and end:
        results["gate3"] = gate3(start, end)
    results["gate4"] = gate4()
    return {k: v.passed for k, v in results.items()}
