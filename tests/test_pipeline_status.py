"""Tests for the pipeline observability journal (src/daily/status.py) —
writer lifecycle, failure capture, log routing, and the derived honesty
states (STALLED / STOPPED_UNEXPECTEDLY). All against a tmp dir, no network."""

import json
import logging
import os
from datetime import datetime, timedelta

import pytest

from src.daily.status import PipelineStatus, history, read_status


STEPS = [("a", "Step A"), ("b", "Step B")]


def test_lifecycle_completed(tmp_path):
    ps = PipelineStatus("job-x", STEPS, status_dir=tmp_path)
    with ps.step("a"):
        ps.update("doing A")
    with ps.step("b"):
        pass
    ps.done(result={"top": "PAYTM"})

    d = read_status(tmp_path)
    assert d["state"] == d["derived_state"] == "completed"
    assert [s["status"] for s in d["steps"]] == ["done", "done"]
    assert d["steps"][0]["seconds"] is not None
    assert d["result"] == {"top": "PAYTM"}
    h = history(status_dir=tmp_path)
    assert len(h) == 1 and h[0]["state"] == "completed"


def test_step_failure_marks_run_failed_and_reraises(tmp_path):
    ps = PipelineStatus("job-x", STEPS, status_dir=tmp_path)
    with pytest.raises(ValueError):
        with ps.step("a"):
            raise ValueError("boom")
    d = read_status(tmp_path)
    assert d["state"] == "failed"
    assert d["steps"][0]["status"] == "failed"
    assert "boom" in d["steps"][0]["one_liner"]
    assert "boom" in d["error"]


def test_log_lines_become_one_liners_and_tail(tmp_path):
    ps = PipelineStatus("job-x", STEPS, status_dir=tmp_path)
    lg = logging.getLogger("src.daily.panel")
    lg.setLevel(logging.INFO)     # main.py sets INFO globally in real runs
    with ps.step("a"):
        lg.info("RELIANCE: 750 rows written")
    ps.done()
    d = read_status(tmp_path)
    assert any("RELIANCE" in line for line in d["log_tail"])
    assert "RELIANCE" in d["steps"][0]["one_liner"]


def test_dead_pid_derives_stopped_unexpectedly(tmp_path):
    ps = PipelineStatus("job-x", STEPS, status_dir=tmp_path)
    # simulate a crash: file says running, but the recorded PID is dead
    ps.data["pid"] = 2 ** 22 + 12345          # extremely unlikely to exist
    ps._write(force=True)
    logging.getLogger().removeHandler(ps._handler)
    d = read_status(tmp_path)
    assert d["state"] == "running"
    assert d["derived_state"] == "stopped_unexpectedly"


def test_alive_but_silent_derives_stalled(tmp_path):
    ps = PipelineStatus("job-x", STEPS, status_dir=tmp_path)
    ps.data["pid"] = os.getpid()              # alive (this test process)
    ps._write(force=True)
    logging.getLogger().removeHandler(ps._handler)
    # age the heartbeat past the stall window
    f = tmp_path / "pipeline_status.json"
    d = json.loads(f.read_text())
    d["updated"] = (datetime.now() - timedelta(seconds=300)).isoformat(timespec="seconds")
    f.write_text(json.dumps(d))
    out = read_status(tmp_path)
    assert out["derived_state"] == "stalled"
    assert out["seconds_since_update"] >= 300


def test_mark_stopped_flips_running_to_stopped(tmp_path):
    from src.daily.status import mark_stopped
    ps = PipelineStatus("job-x", STEPS, status_dir=tmp_path)
    with ps.step("a"):
        # simulate the API stopping a run mid-step (file still says running)
        pass
    ps.data["state"] = "running"          # re-arm as if mid-run
    ps.data["steps"][1]["status"] = "running"
    ps._write(force=True)
    logging.getLogger().removeHandler(ps._handler)

    mark_stopped("stopped by user from UI", status_dir=tmp_path)
    d = read_status(tmp_path)
    assert d["state"] == d["derived_state"] == "stopped"
    assert d["steps"][1]["status"] == "failed"
    assert "stopped by user" in d["error"]
    assert history(status_dir=tmp_path)[0]["state"] == "stopped"


# ── per-item trace (src/daily/trace.py) ───────────────────────────────


def test_trace_roundtrip_and_filters(tmp_path):
    from src.daily.trace import Trace, read_trace
    tr = Trace(status_dir=tmp_path)
    tr.event("score", "RELIANCE", "ok", p_win=0.47,
             indicators={"rsi_14": 55.2}, blocks_present={"fno": True})
    tr.event("score", "PAYTM", "skip", reason="no feature row")
    tr.event("rank", "RELIANCE", "ok", rank=1, prob=0.47)
    tr.close()

    all_ev = read_trace(status_dir=tmp_path)
    assert all_ev[0]["step"] == "_meta"          # sources stamped first
    assert "data_sources" in all_ev[0]
    score = read_trace(step="score", status_dir=tmp_path)
    assert len(score) == 2
    rel = read_trace(item="reli", status_dir=tmp_path)   # substring, case-insensitive
    assert {e["step"] for e in rel} == {"score", "rank"}
    assert score[0]["indicators"]["rsi_14"] == 55.2
    assert score[1]["reason"] == "no feature row"


def test_trace_truncates_on_new_run(tmp_path):
    from src.daily.trace import Trace, read_trace
    t1 = Trace(status_dir=tmp_path)
    t1.event("panel", "OLD", "ok")
    t1.close()
    t2 = Trace(status_dir=tmp_path)   # new run — journal restarts
    t2.event("panel", "NEW", "ok")
    t2.close()
    items = [e["item"] for e in read_trace(step="panel", status_dir=tmp_path)]
    assert items == ["NEW"]


def test_story_builder_narrates_tiebreak_and_conclusion(tmp_path):
    from src.daily.story import build_story
    from src.daily.trace import Trace

    ps = PipelineStatus("daily-pipeline", STEPS, status_dir=tmp_path)
    tr = Trace(status_dir=tmp_path)
    tr.event("universe", "2 stocks", "ok", source="universe_top100.csv",
             rule="PIT top-100 by turnover", as_of="2026-07-14", n=2)
    tr.event("fetch_bhavcopy", "2026-07-14", "ok",
             source="nsearchives.nseindia.com", detail="eq + fo written")
    tr.event("fetch_global", "spx", "ok", source="Yahoo Finance (^GSPC)", rows=900)
    tr.event("panel", "AAA", "ok", rows=700, source="local bhavcopy store")
    # both stocks tie on P → story must explain the momentum tie-break
    tr.event("score", "AAA", "ok", p_win=0.31, indicators={"ret_20d": 0.09})
    tr.event("score", "BBB", "ok", p_win=0.31, indicators={"ret_20d": 0.01})
    tr.event("rank", "AAA", "ok", rank=1, prob=0.31, target=110.0, stop=90.0, why="x")
    tr.close()
    ps.done(result={"summary": {"headline": "top pick AAA",
                                "result_meaning": ["m1"], "next_steps": ["n1"]}})

    story = build_story(tmp_path)
    assert story["state"] == "completed"
    titles = [s["title"] for s in story["steps"]]
    assert titles[0].startswith("Stock selection")
    sel = story["steps"][0]
    assert sel["why"].startswith("PIT top-100")
    rank_step = story["steps"][-1]
    assert "SAME calibrated P(win)" in rank_step["why"]
    assert "AAA" in rank_step["why"] and "+9.0%" in rank_step["why"]
    assert story["conclusion"]["headline"] == "top pick AAA"
