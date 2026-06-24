"""Paper-trading runner restart safety (PLAN_v3 §14 / Gate 4; v3_supplementry
§10.2, A.3).

The live loop persists positions, closed trades, and the risk DayState to
session_state_<date>.json after every fill/exit. A killed-and-resumed runner for
the same session MUST rehydrate identical state from that file — otherwise a
crash would silently drop open risk or double-count P&L. These tests simulate a
session's fills/exits, persist, then construct a FRESH runner and assert the
restored EOD state matches. They would fail if _restore_state / _persist_state
were removed.

PaperRunner builds a SessionRecorder whose feed needs UPSTOX_ACCESS_TOKEN at
construction (no network call until a poll), so we set a dummy token; the
restore path never polls.
"""

from __future__ import annotations

import json

import pytest

from src.intraday import ROOT, load_config


@pytest.fixture
def paper(repo, monkeypatch):
    """A trained bundle on disk + a dummy feed token so PaperRunner constructs."""
    import src.intraday.tracking as tracking

    def _no_registry(*a, **k):
        raise tracking.TrackingError("no MLflow server in tests — use local bundle")

    monkeypatch.setattr(tracking, "git_sha", lambda *a, **k: "test-sha")
    # PaperRunner._load_bundle tries the registry first; with no MLflow server the
    # client retries the network for minutes. Force the local-bundle fallback
    # (which is the path under test for restart safety).
    monkeypatch.setattr(tracking, "load_stage", _no_registry)
    from src.intraday.trainer import train_production
    train_production(end_day=repo["days"][160], train_months=5, tune=False,
                     push_dvc=False, track=False)
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "dummy-token-for-construction")
    return repo


def _simulate_fills(runner, day):
    """Mimic the runner taking two fills and closing one — exactly the dict shape
    _open / _close build, driving risk + persistence the same way the live loop
    does."""
    # open AAA (Tech) and BBB (Bank)
    for sym, direction, entry in (("AAA", 1, 1000.0), ("BBB", -1, 1250.0)):
        runner.positions[sym] = {
            "symbol": sym, "direction": direction, "entry": entry, "qty": 10,
            "target": entry * 1.01, "stop": entry * 0.99,
            "entry_ts": f"{day}T09:45:00", "t_bar": f"{day}T12:45:00",
            "p_cal": 0.93, "last_ltp": entry, "features": {"atr_pct": 0.01},
        }
        runner.risk.on_open(sym)
    runner._persist_state()
    # close AAA at a profit
    runner._persist_state()  # state after opens
    pos = runner.positions.pop("AAA")
    runner.risk.on_close("AAA", 500.0)
    runner.closed.append({
        "symbol": "AAA", "direction": 1, "entry": 1000.0, "qty": 10,
        "exit": 1010.0, "barrier": "target", "pnl_pct_gross": 0.01,
        "cost_pct": 0.001, "pnl_pct_net": 0.009, "label": 1,
        "features": pos["features"],
    })
    runner._persist_state()


def test_state_restore_after_kill(paper):
    from src.intraday.paper_runner import PaperRunner
    day = paper["days"][120]

    r1 = PaperRunner(1_000_000, day=day)
    _simulate_fills(r1, day)

    # the session-state file must exist (persisted on every fill/exit)
    sf = ROOT / load_config()["paths"]["state"] / f"session_state_{day.isoformat()}.json"
    assert sf.exists()

    # ── kill: a brand-new runner for the same day rehydrates ──────────
    r2 = PaperRunner(1_000_000, day=day)
    assert set(r2.positions) == set(r1.positions) == {"BBB"}
    assert [c["symbol"] for c in r2.closed] == ["AAA"]
    assert r2.closed[0]["label"] == 1
    # risk DayState restored exactly (open book + realized pnl + halt flag)
    assert r2.risk.state.open_positions == r1.risk.state.open_positions
    assert r2.risk.state.pnl_inr == pytest.approx(r1.risk.state.pnl_inr) == 500.0
    assert r2.risk.state.halted is False
    # the still-open BBB position survives the restart with its full geometry
    assert r2.positions["BBB"]["direction"] == -1
    assert r2.positions["BBB"]["entry"] == pytest.approx(1250.0)


def test_restored_state_is_not_a_fresh_session(paper):
    """A fresh runner (no prior state file) starts empty — proves the restore in
    the previous test is real, not vacuous."""
    from src.intraday.paper_runner import PaperRunner
    day = paper["days"][121]      # a day with no session-state file
    r = PaperRunner(1_000_000, day=day)
    assert r.positions == {}
    assert r.closed == []
    assert r.risk.state.pnl_inr == 0.0


def test_resume_preserves_daily_loss_halt(paper):
    """If the killed session had already tripped the daily-loss halt, the resumed
    runner must come back halted — the kill switch cannot be erased by a restart."""
    from src.intraday.paper_runner import PaperRunner
    day = paper["days"][122]
    r1 = PaperRunner(1_000_000, day=day)
    r1.risk.on_open("AAA")
    r1.risk.on_close("AAA", -20_000)     # -2% > 1.5% daily limit → halts
    assert r1.risk.state.halted is True
    r1._persist_state()

    r2 = PaperRunner(1_000_000, day=day)
    assert r2.risk.state.halted is True
    assert r2.risk.state.halt_reason == "daily_loss_limit"
    ok, reason = r2.risk.can_open("CCC", day)
    assert not ok and reason == "daily_loss_limit"


def test_persisted_file_round_trips_json(paper):
    """The persisted state is plain JSON the resume path reads back verbatim."""
    from src.intraday.paper_runner import PaperRunner
    day = paper["days"][123]
    r = PaperRunner(1_000_000, day=day)
    _simulate_fills(r, day)
    sf = ROOT / load_config()["paths"]["state"] / f"session_state_{day.isoformat()}.json"
    blob = json.loads(sf.read_text())
    assert set(blob) == {"positions", "closed", "day_state"}
    assert "BBB" in blob["positions"]
    assert blob["day_state"]["pnl_inr"] == 500.0
