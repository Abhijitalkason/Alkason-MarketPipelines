"""v3 Serving API (PLAN_v3 §15; v3_supplementry §13).

FastAPI, Pydantic v2. Every endpoint requires X-API-Key matching
os.environ[cfg.api.api_key_env]; with no key configured the app refuses to
start (B-8). Models load once at startup from the registry Production stage
(fallback models/v3 with WARN); inference runs off the event loop via
run_in_executor; check_schema runs before every predict. No model-mutating
endpoint exists — training/promotion are CLI + registry only.

The signal endpoints read the append-only journal and signals log — the live
track record is whatever the paper/live runner has actually recorded.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from src.intraday import ROOT, load_config, now_ist, today_ist
from src.intraday.journal import journal_days, journal_verify, read_journal

load_dotenv()
logger = logging.getLogger(__name__)
CFG = load_config()

_state: dict = {}


def _require_api_key_env() -> str:
    key_env = CFG["api"]["api_key_env"]
    key = os.environ.get(key_env, "")
    if not key:
        raise RuntimeError(
            f"{key_env} not set — the serving API refuses to start without an API key (B-8)"
        )
    return key


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["api_key"] = _require_api_key_env()
    try:
        from src.intraday.tracking import load_stage
        _state["bundle"] = load_stage("Production")
        _state["model_source"] = "registry:Production"
    except Exception as e:  # noqa: BLE001 — local fallback with WARN
        logger.warning("Production registry load failed (%s) — loading models/v3 files", e)
        _state["bundle"] = _load_local_bundle()
        _state["model_source"] = "local:models/v3"
    from src.intraday.conformal_gate import ACIGate
    _state["gate"] = ACIGate.load()
    _state["started"] = now_ist()
    logger.info("v3 API up — models from %s", _state["model_source"])
    yield


def _load_local_bundle():
    from src.intraday.blend import Blender
    from src.intraday.flow_model import FlowModel
    from src.intraday.price_model import PriceModel

    sd = ROOT / CFG["paths"]["state"]
    pm = PriceModel.load(sd / "price_model.joblib")
    fm = FlowModel.load(sd / "flow_model.joblib") if (sd / "flow_model.joblib").exists() else None
    blender = Blender.load(sd / "blender.joblib")
    return pm, fm, blender


app = FastAPI(
    title="Intraday Selective Signal API (v4)",
    version="4.0.0",
    description=(
        "Selective NSE intraday signal API (PLAN_v4). Every endpoint requires the "
        "`X-API-Key` header — click **Authorize** and paste the key (the value of "
        "the env var named by `api.api_key_env`, default `V3_API_KEY`) to try the "
        "endpoints from this page."
    ),
    lifespan=lifespan,
    docs_url="/docs",          # Swagger UI (explicit)
    redoc_url="/redoc",        # ReDoc (explicit)
    openapi_url="/openapi.json",
)

# Declaring the scheme makes Swagger UI render an Authorize button that injects
# X-API-Key on every "Try it out" call (auto_error=False so we return our own 401).
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_key(x_api_key: str = Security(_api_key_scheme)) -> None:
    if x_api_key != _state.get("api_key"):
        raise HTTPException(401, detail="invalid or missing X-API-Key")


def _halt_guard() -> None:
    from src.intraday.risk import halt_reason, is_halted
    if is_halted():
        raise HTTPException(503, detail={"halted": True, "reasons": halt_reason()})


# ── models ────────────────────────────────────────────────────────────

class Health(BaseModel):
    status: str
    model_source: str
    model_age_min: float
    gate_tau: float
    gate_days_updated: int
    halted: bool
    journal_chain_ok: bool
    uptime_seconds: float


# ── endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=Health, dependencies=[Depends(require_key)])
async def health():
    from src.intraday.risk import is_halted
    sd = ROOT / CFG["paths"]["state"]
    mp = sd / "price_model.joblib"
    age_min = ((now_ist().timestamp() - mp.stat().st_mtime) / 60) if mp.exists() else -1.0
    today = today_ist()
    chain_ok = journal_verify(today) if today in journal_days() else True
    g = _state["gate"].state
    return Health(
        status="ok", model_source=_state["model_source"], model_age_min=age_min,
        gate_tau=g.tau, gate_days_updated=g.days_updated, halted=is_halted(),
        journal_chain_ok=chain_ok,
        uptime_seconds=(now_ist() - _state["started"]).total_seconds(),
    )


@app.get("/signals/today", dependencies=[Depends(require_key)])
async def signals_today():
    day = today_ist()
    fills = read_journal(day, "fill")
    exits = read_journal(day, "exit")
    explains = {e["payload"].get("symbol"): e["payload"] for e in read_journal(day, "explain")}
    return {"date": str(day), "fills": [f["payload"] for f in fills],
            "exits": [e["payload"] for e in exits], "explanations": explains}


@app.get("/signals/history", dependencies=[Depends(require_key)])
async def signals_history(days: int = 30):
    sig_dir = Path(CFG["paths"]["signals_log"])
    sig_dir = sig_dir if sig_dir.is_absolute() else ROOT / sig_dir
    cutoff = today_ist() - timedelta(days=days)
    rows = []
    import json as _json
    for f in sorted(sig_dir.glob("paper_*.json")):
        try:
            d = date.fromisoformat(f.stem.replace("paper_", ""))
        except ValueError:
            continue
        if d < cutoff:
            continue
        for rec in _json.loads(f.read_text()):
            rows.append({"date": str(d), "symbol": rec["symbol"], "barrier": rec["barrier"],
                         "label": rec["label"], "pnl_pct_net": rec["pnl_pct_net"]})
    if not rows:
        return {"days": days, "n": 0, "win_rate": None, "expectancy": None, "signals": []}
    tdf = pd.DataFrame(rows)
    daily = tdf.groupby("date")["pnl_pct_net"].sum()
    equity = (1 + daily).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    return {"days": days, "n": len(tdf), "win_rate": float(tdf["label"].mean()),
            "expectancy": float(tdf["pnl_pct_net"].mean()), "max_drawdown": dd,
            "signals": rows}


@app.get("/screen/today", dependencies=[Depends(require_key)])
async def screen_today():
    day = today_ist()
    screens = read_journal(day, "screen")
    excl_path = ROOT / CFG["paths"]["state"] / "excluded_symbols.json"
    import json as _json
    excluded = _json.loads(excl_path.read_text()) if excl_path.exists() else []
    return {"date": str(day), "screen": screens[-1]["payload"] if screens else None,
            "excluded_symbols": excluded}


@app.get("/performance", dependencies=[Depends(require_key)])
async def performance():
    from src.intraday.conformal_gate import coverage_report
    hist = await signals_history(days=90)
    g = CFG["gates"]
    sig_dir = Path(CFG["paths"]["signals_log"])
    sig_dir = sig_dir if sig_dir.is_absolute() else ROOT / sig_dir
    import json as _json
    rows = []
    for f in sorted(sig_dir.glob("paper_*.json")):
        for rec in _json.loads(f.read_text()):
            rows.append({"symbol": rec["symbol"], "label": int(rec["label"])})
    cov = coverage_report(pd.DataFrame(rows)) if rows else {}
    return {
        "rolling": {k: hist.get(k) for k in ("win_rate", "expectancy", "max_drawdown", "n")},
        "thresholds": {"min_win_rate": g["min_win_rate"], "min_expectancy_pct": g["min_expectancy_pct"]},
        "per_symbol_coverage": cov,
    }


@app.get("/drift", dependencies=[Depends(require_key)])
async def drift_status():
    p = Path(CFG["paths"]["drift"])
    p = p if p.is_absolute() else ROOT / p
    import json as _json
    files = sorted(p.glob("*.json")) if p.exists() else []
    latest = _json.loads(files[-1].read_text()) if files else None
    aci = read_journal(today_ist(), "aci_update")
    return {"latest_drift": latest, "last_aci_update": aci[-1]["payload"] if aci else None}


@app.post("/backtest", dependencies=[Depends(require_key)])
async def trigger_backtest(start: str, end: str):
    """Start a background backtest job; returns a job id. (Read-only on models —
    no training/promotion via the API.)"""
    _halt_guard()
    job_id = now_ist().strftime("%Y%m%d%H%M%S")
    _state.setdefault("jobs", {})[job_id] = {"status": "running"}

    async def _run():
        from src.intraday.backtester import run_backtest
        try:
            rep = await asyncio.get_event_loop().run_in_executor(
                None, lambda: run_backtest(date.fromisoformat(start), date.fromisoformat(end),
                                           args_note=f"api-job-{job_id}"))
            _state["jobs"][job_id] = {"status": "done", "report": rep["gates"]}
        except Exception as e:  # noqa: BLE001 — surfaced via job status
            _state["jobs"][job_id] = {"status": "failed", "error": str(e)}

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/backtest/{job_id}", dependencies=[Depends(require_key)])
async def backtest_status(job_id: str):
    job = _state.get("jobs", {}).get(job_id)
    if not job:
        raise HTTPException(404, detail="unknown job id")
    return {"job_id": job_id, **job}
