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
from fastapi.responses import FileResponse

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
async def scoreboard(horizon: str = "next_day", source: str = "oos"):
    """How the model's picks actually do. source=oos → at-scale lookahead-free
    backtest record; source=live → the running day-by-day forward scoreboard."""
    from src.daily.evaluate import scoreboard as live_scoreboard
    from src.daily.evaluate import scoreboard_from_oos
    try:
        return live_scoreboard(horizon) if source == "live" else scoreboard_from_oos(horizon)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))


@app.get("/daily/chart/{symbol}", dependencies=[Depends(require_key)])
async def daily_chart(symbol: str, horizon: str = "next_day", day: str | None = None):
    d = day or today_ist().isoformat()
    path = daily_path("charts") / d / f"{symbol}_{horizon}.png"
    if not path.exists():
        raise HTTPException(404, detail=f"no chart for {symbol} {horizon} on {d}")
    return FileResponse(str(path), media_type="image/png")
