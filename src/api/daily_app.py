"""Daily prediction & listing API (read-only). Serves the top-N watchlist the
daily job produces — no model-mutating or order-placing endpoint exists.

Mirrors src/api/app.py conventions: X-API-Key (refuses to start without it),
bundles loaded once at startup, GET-only. Distinct port (config_daily api.port).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from src.daily import daily_path, enabled_horizons, load_daily_config, now_ist, today_ist

load_dotenv()
logger = logging.getLogger(__name__)
CFG = load_daily_config()
_state: dict = {}


def _require_api_key_env() -> str:
    key_env = CFG["api"]["api_key_env"]
    key = os.environ.get(key_env, "")
    if not key:
        raise RuntimeError(f"{key_env} not set — daily API refuses to start without an API key")
    return key


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["api_key"] = _require_api_key_env()
    _state["started"] = now_ist()
    logger.info("daily API up — horizons %s", enabled_horizons())
    yield


app = FastAPI(title="Daily Prediction & Listing API", version="1.0.0", lifespan=lifespan)


async def require_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != _state.get("api_key"):
        raise HTTPException(401, detail="invalid or missing X-API-Key")


def _latest_list() -> dict | None:
    lists = sorted(daily_path("lists").glob("list_*.json")) if daily_path("lists").exists() else []
    return json.loads(lists[-1].read_text()) if lists else None


@app.get("/health", dependencies=[Depends(require_key)])
async def health():
    latest = _latest_list()
    return {"status": "ok", "enabled_horizons": enabled_horizons(),
            "latest_list_date": latest["date"] if latest else None,
            "uptime_seconds": (now_ist() - _state["started"]).total_seconds()}


@app.get("/daily/list", dependencies=[Depends(require_key)])
async def daily_list(horizon: str | None = None):
    """Today's list if present, else the most recent. Optionally filter to one horizon."""
    today = today_ist().isoformat()
    p = daily_path("lists") / f"list_{today}.json"
    data = json.loads(p.read_text()) if p.exists() else _latest_list()
    if data is None:
        raise HTTPException(404, detail="no daily list yet — run `daily-list`")
    if horizon:
        data = {**data, "horizons": {horizon: data.get("horizons", {}).get(horizon, {"picks": []})}}
    return data


@app.get("/daily/list/history", dependencies=[Depends(require_key)])
async def daily_list_history(days: int = 30):
    cutoff = today_ist() - timedelta(days=days)
    out = []
    for f in sorted(daily_path("lists").glob("list_*.json")):
        try:
            d = date.fromisoformat(f.stem.replace("list_", ""))
        except ValueError:
            continue
        if d >= cutoff:
            out.append(json.loads(f.read_text()))
    return {"days": days, "n": len(out), "lists": out}


@app.get("/daily/scoreboard", dependencies=[Depends(require_key)])
async def scoreboard(horizon: str = "swing_1_5d", source: str = "oos"):
    """How the model's picks actually do. source=oos → at-scale lookahead-free
    backtest record; source=live → the running day-by-day forward scoreboard."""
    from src.daily.evaluate import scoreboard as live_scoreboard
    from src.daily.evaluate import scoreboard_from_oos
    try:
        return live_scoreboard(horizon) if source == "live" else scoreboard_from_oos(horizon)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))


@app.get("/daily/chart/{symbol}", dependencies=[Depends(require_key)])
async def daily_chart(symbol: str, horizon: str = "swing_1_5d", day: str | None = None):
    d = day or today_ist().isoformat()
    path = daily_path("charts") / d / f"{symbol}_{horizon}.png"
    if not path.exists():
        raise HTTPException(404, detail=f"no chart for {symbol} {horizon} on {d}")
    return FileResponse(str(path), media_type="image/png")


# ── pipeline observability (feature request 2026-07-17) ──────────────

@app.get("/pipeline/status", dependencies=[Depends(require_key)])
async def pipeline_status():
    """Live status of the AI/ML pipeline: current run (job, steps, one-line
    updates, log tail, derived state incl. STALLED / STOPPED_UNEXPECTEDLY when
    the writing process died) + recent run history."""
    from src.daily.status import history, read_status
    return {"current": read_status(), "history": history(8)}


@app.get("/pipeline/trace", dependencies=[Depends(require_key)])
async def pipeline_trace(step: str | None = None, item: str | None = None,
                         limit: int = 3000):
    """The current run's per-item trace: every data source hit (with URL/
    ticker), every symbol's panel build, every stock's indicator values +
    calibrated score (or skip reason), the final ranking — filterable by
    step and item substring. Rendered at /ui/trace."""
    from src.daily.trace import read_trace
    return {"events": read_trace(step=step, item=item, limit=limit)}


@app.get("/pipeline/stream", dependencies=[Depends(require_key)])
async def pipeline_stream():
    """LIVE event stream (feature request 2026-07-22): holds the connection
    open and pushes one JSON line per event AS IT HAPPENS — every trace event
    (per-series fetch, per-stock panel/score/rank) plus step start/complete
    transitions and run-state changes. Connect first, then press ▶ Start:
    the feed announces the new run and streams it end to end."""
    import asyncio
    import json as _json

    from starlette.responses import StreamingResponse

    from src.daily.status import read_status
    from src.daily.trace import _trace_file

    async def gen():
        f = _trace_file()
        pos = f.stat().st_size if f.exists() else 0      # tail from NOW, no replay
        st = read_status()
        last_steps = {s["name"]: s["status"] for s in (st or {}).get("steps", [])}
        last_state = (st or {}).get("derived_state")
        yield _json.dumps({"kind": "hello",
                           "msg": f"live feed connected (current state: {last_state or 'never run'}) "
                                  "— press ▶ Start to watch a run stream"}) + "\n"
        while True:
            try:
                if f.exists():
                    size = f.stat().st_size
                    if size < pos:                        # truncation = a NEW run began
                        pos = 0
                        yield _json.dumps({"kind": "info", "msg": "🚀 new run started"}) + "\n"
                    if size > pos:
                        with open(f) as fh:
                            fh.seek(pos)
                            chunk = fh.read()
                            pos = fh.tell()
                        for line in chunk.splitlines():
                            try:
                                yield _json.dumps({"kind": "trace", **_json.loads(line)}) + "\n"
                            except ValueError:
                                continue
                st = read_status()
                if st:
                    for s in st.get("steps", []):
                        if last_steps.get(s["name"]) != s["status"]:
                            last_steps[s["name"]] = s["status"]
                            msg = {"running": f"▶ STEP started: {s['label']}",
                                   "done": f"✔ STEP completed: {s['label']} ({s['seconds']}s)",
                                   "failed": f"✖ STEP FAILED: {s['label']} — {s['one_liner']}"}.get(s["status"])
                            if msg:
                                yield _json.dumps({"kind": "step", "msg": msg}) + "\n"
                    if st.get("derived_state") != last_state:
                        last_state = st["derived_state"]
                        res = st.get("result") or {}
                        tp = res.get("top_pick")
                        yield _json.dumps({
                            "kind": "state", "state": last_state,
                            "msg": f"🏁 RUN {last_state.upper()}"
                                   + (f" — top pick {tp['symbol']} P={tp['prob']}" if tp else "")
                                   + (f" — {st.get('error')}" if st.get("error") else "")}) + "\n"
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a read race must not kill the stream
                await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/pipeline/story", dependencies=[Depends(require_key)])
async def pipeline_story():
    """The run as a numbered narrative: stock selection (source + why) →
    each API call + its result → indicator/scoring stage → WHY the top pick
    won (derived from the real numbers) → conclusion + next steps."""
    from src.daily.story import build_story
    return build_story()


@app.get("/ui/trace")
async def trace_ui():
    """The dedicated pipeline-trace dashboard (separate page — full-volume
    per-stock detail; the main UI's pipeline panel links here)."""
    return HTMLResponse((Path(__file__).parent / "ui" / "trace.html").read_text())


@app.post("/pipeline/run", dependencies=[Depends(require_key)])
async def pipeline_run(live: bool = False):
    """Start a fresh daily-pipeline run (detached subprocess). Job control
    only — spawns the same CLI a human/cron would; refuses a second
    concurrent run. `live=true` adds --live (provisional Upstox intraday
    snapshot). NOT a model-mutating endpoint (training/promotion stay
    CLI-only per repo policy)."""
    import subprocess
    import sys as _sys

    from src.daily.status import read_status
    from src.intraday import ROOT
    cur = read_status()
    if cur and cur.get("derived_state") == "running":
        raise HTTPException(409, detail=f"pipeline already running (pid {cur['pid']})")
    cmd = [_sys.executable, "main.py", "--mode", "daily-pipeline"]
    if live:
        cmd.append("--live")
    log = open(ROOT / "reports" / "daily" / "pipeline_ui_runs.log", "ab")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,   # survives server restarts; killed only via /pipeline/stop
    )
    logger.info("pipeline started from UI: pid %d (live=%s)", proc.pid, live)
    return {"started": True, "pid": proc.pid, "live": live}


@app.post("/pipeline/stop", dependencies=[Depends(require_key)])
async def pipeline_stop():
    """Stop the running pipeline (SIGTERM to its recorded pid) and flip the
    journal to an explicit STOPPED state."""
    import signal

    from src.daily.status import mark_stopped, read_status
    cur = read_status()
    if not cur or cur.get("derived_state") != "running":
        raise HTTPException(409, detail="no running pipeline to stop")
    pid = int(cur["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    mark_stopped(f"stopped by user from UI (SIGTERM → pid {pid})")
    logger.info("pipeline pid %d stopped from UI", pid)
    return {"stopped": True, "pid": pid}


# ── single-stock deep analyzer (feature request 2026-07-16) ───────────

_UI_PATH = Path(__file__).parent / "ui" / "analyze.html"


@app.get("/ui")
async def analyzer_ui():
    """The analyzer dashboard (static page; data endpoints still need the key)."""
    return HTMLResponse(_UI_PATH.read_text())


@app.get("/symbols", dependencies=[Depends(require_key)])
async def symbols():
    """All NSE symbols in the latest bhavcopy — powers the UI autocomplete."""
    import pandas as pd
    from src.intraday import ROOT
    files = sorted((ROOT / "data" / "bhavcopy").glob("eq_*.parquet"))
    if not files:
        raise HTTPException(404, detail="no bhavcopy on disk")
    syms = pd.read_parquet(files[-1], columns=["symbol"])["symbol"].sort_values().tolist()
    return {"as_of": files[-1].stem[3:], "n": len(syms), "symbols": syms}


@app.get("/analyze/{symbol}", dependencies=[Depends(require_key)])
async def analyze_symbol(symbol: str):
    """Full background analysis + transparent BUY/HOLD/SELL scorecard for one
    NSE stock. Network sections (news, fundamentals) run live and degrade
    honestly when unreachable. CPU/network-bound → executed off the event loop."""
    import anyio
    from src.analysis.engine import analyze
    try:
        return await anyio.to_thread.run_sync(analyze, symbol)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, detail=str(e))


@app.get("/analyze/{symbol}/chart.png", dependencies=[Depends(require_key)])
async def analyze_chart(symbol: str):
    from src.intraday import ROOT
    path = ROOT / "reports" / "analysis" / f"{symbol.upper()}_{today_ist().isoformat()}.png"
    if not path.exists():
        raise HTTPException(404, detail="chart not rendered yet — call /analyze/{symbol} first")
    return FileResponse(str(path), media_type="image/png")
