"""Pipeline observability (feature request 2026-07-17) — a live, file-backed
status journal for the AI/ML pipeline jobs, so "what is running, which step,
which stock/API, what result" is visible in the UI instead of a mystery.

Design:
  - The running job holds a PipelineStatus; each step is a context manager.
  - A logging.Handler is attached for the run's duration, so every existing
    log line from src.daily/src.intraday (per-day bhavcopy fetches, per-series
    global pulls, per-symbol feature skips, screen results...) becomes the
    current step's live one-liner AND a rolling log tail — no invasive
    instrumentation of the pipeline internals.
  - State is written ATOMICALLY (tmp + rename) to reports/daily/
    pipeline_status.json; readers (API/UI) can poll it safely mid-write.
  - Completed runs append one summary line to pipeline_runs.jsonl (history).
  - read_status() derives honesty states the file alone can't know:
    a "running" file whose PID is dead → STOPPED_UNEXPECTEDLY; alive but no
    heartbeat for 90s → STALLED.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from src.daily import daily_path

HEARTBEAT_STALL_S = 90          # running + no write for this long → STALLED
_WRITE_MIN_INTERVAL_S = 0.5     # throttle mid-run writes
_LOG_TAIL_LINES = 40


def _status_file(status_dir: Path | None = None) -> Path:
    return (status_dir or daily_path("reports")) / "pipeline_status.json"


def _history_file(status_dir: Path | None = None) -> Path:
    return (status_dir or daily_path("reports")) / "pipeline_runs.jsonl"


class _StatusLogHandler(logging.Handler):
    """Routes every log record emitted during the run into the status journal."""

    def __init__(self, run: "PipelineStatus"):
        super().__init__(level=logging.INFO)
        self.run = run
        self.setFormatter(logging.Formatter("%(name)s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.run._log_line(self.format(record))
        except Exception:  # noqa: BLE001 — observability must never kill the job
            pass


class PipelineStatus:
    """Writer side. Usage:

        ps = PipelineStatus("daily-pipeline", [("fetch", "Fetch EOD"), ...])
        with ps.step("fetch"):
            ...            # log lines become live one-liners automatically
            ps.update("bhavcopy 2026-07-17 on disk")   # or set one explicitly
        ps.done(result={...})   # or ps.fail("reason") — step ctx does it on raise
    """

    def __init__(self, job: str, steps: list[tuple[str, str]],
                 status_dir: Path | None = None):
        self._dir = status_dir
        self._file = _status_file(status_dir)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._last_write = 0.0
        self._current: str | None = None
        self.data = {
            "job": job, "pid": os.getpid(),
            "state": "running",
            "started": datetime.now().isoformat(timespec="seconds"),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "steps": [{"name": n, "label": lbl, "status": "pending",
                       "one_liner": "", "started": None, "seconds": None}
                      for n, lbl in steps],
            "log_tail": [], "result": None, "error": None,
        }
        self._handler = _StatusLogHandler(self)
        logging.getLogger().addHandler(self._handler)
        self._write(force=True)

    # ── steps ─────────────────────────────────────────────────────────
    def step(self, name: str) -> "_StepCtx":
        return _StepCtx(self, name)

    def _step_obj(self, name: str) -> dict:
        for s in self.data["steps"]:
            if s["name"] == name:
                return s
        raise KeyError(f"unknown pipeline step {name!r}")

    def update(self, msg: str) -> None:
        """Explicit one-liner for the current step."""
        if self._current:
            self._step_obj(self._current)["one_liner"] = msg[:240]
        self._write()

    def _log_line(self, msg: str) -> None:
        tail = self.data["log_tail"]
        tail.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}"[:300])
        del tail[:-_LOG_TAIL_LINES]
        if self._current:
            self._step_obj(self._current)["one_liner"] = msg[:240]
        self._write()

    # ── run termination ───────────────────────────────────────────────
    def done(self, result: dict | None = None) -> None:
        self.data["state"], self.data["result"] = "completed", result
        self._finish()

    def fail(self, error: str) -> None:
        self.data["state"], self.data["error"] = "failed", error[:500]
        self._finish()

    def _finish(self) -> None:
        logging.getLogger().removeHandler(self._handler)
        self._write(force=True)
        summary = {k: self.data[k] for k in ("job", "state", "started", "updated",
                                             "result", "error")}
        summary["steps"] = [{"name": s["name"], "status": s["status"],
                             "seconds": s["seconds"]} for s in self.data["steps"]]
        with open(_history_file(self._dir), "a") as f:
            f.write(json.dumps(summary, default=str) + "\n")

    # ── atomic, throttled writes ──────────────────────────────────────
    def _write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < _WRITE_MIN_INTERVAL_S:
            return
        self._last_write = now
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        tmp.replace(self._file)


class _StepCtx:
    def __init__(self, run: PipelineStatus, name: str):
        self.run, self.name = run, name

    def __enter__(self):
        s = self.run._step_obj(self.name)
        s["status"], s["started"] = "running", datetime.now().isoformat(timespec="seconds")
        self.run._current = self.name
        self._t0 = time.time()
        self.run._write(force=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        s = self.run._step_obj(self.name)
        s["seconds"] = round(time.time() - self._t0, 1)
        if exc_type is not None:
            s["status"] = "failed"
            s["one_liner"] = f"{exc_type.__name__}: {exc}"[:240]
            self.run._current = None
            self.run.fail(f"step {self.name}: {exc_type.__name__}: {exc}")
            return False                      # re-raise — the job must still fail loudly
        s["status"] = "done"
        self.run._current = None
        self.run._write(force=True)
        return False


def mark_stopped(reason: str = "stopped by user",
                 status_dir: Path | None = None) -> dict | None:
    """Called by the API's stop endpoint AFTER signalling the pipeline process:
    flips a 'running' journal to an explicit STOPPED state (instead of letting
    it surface as stopped_unexpectedly) and records the run in history."""
    f = _status_file(status_dir)
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if d["state"] == "running":
        d["state"], d["error"] = "stopped", reason[:300]
        d["updated"] = datetime.now().isoformat(timespec="seconds")
        for s in d["steps"]:
            if s["status"] == "running":
                s["status"] = "failed"
                s["one_liner"] = reason[:240]
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, default=str))
        tmp.replace(f)
        summary = {k: d[k] for k in ("job", "state", "started", "updated", "result", "error")}
        summary["steps"] = [{"name": s["name"], "status": s["status"],
                             "seconds": s["seconds"]} for s in d["steps"]]
        with open(_history_file(status_dir), "a") as fh:
            fh.write(json.dumps(summary, default=str) + "\n")
    return d


# ── reader side (API/UI) ──────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return False


def read_status(status_dir: Path | None = None) -> dict | None:
    """Current status + derived truth: a 'running' file whose process died is
    STOPPED_UNEXPECTEDLY; alive but silent past the heartbeat window is STALLED."""
    f = _status_file(status_dir)
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    derived = d["state"]
    age = (datetime.now() - datetime.fromisoformat(d["updated"])).total_seconds()
    if d["state"] == "running":
        if not _pid_alive(int(d.get("pid", -1))):
            derived = "stopped_unexpectedly"
        elif age > HEARTBEAT_STALL_S:
            derived = "stalled"
    d["derived_state"] = derived
    d["seconds_since_update"] = round(age)
    return d


def history(n: int = 8, status_dir: Path | None = None) -> list[dict]:
    f = _history_file(status_dir)
    if not f.exists():
        return []
    lines = f.read_text().strip().splitlines()
    return [json.loads(x) for x in lines[-n:]][::-1]
