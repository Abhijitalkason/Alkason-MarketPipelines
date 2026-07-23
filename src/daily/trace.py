"""Per-item pipeline trace (feature request 2026-07-17) — the deep companion
to status.py. Status answers "which step, what state"; the trace answers
"which stock / which series / which URL, with what parameters, and what came
out" — one structured JSONL event per item:

    {"ts": ..., "step": "score", "item": "RELIANCE", "status": "ok",
     "p_win": 0.4744, "indicators": {"ret_5d": 0.012, "rsi_14": 55.3, ...},
     "blocks_present": {"fno": true, "global": true, ...}}

The file (reports/daily/pipeline_trace.jsonl) is truncated at the start of
each run — it is always the CURRENT run's full trace (~220 events for the
top-100 universe), served at GET /pipeline/trace and rendered at /ui/trace.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.daily import daily_path

# the interpretable indicator subset traced per stock at scoring time
KEY_INDICATORS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_pct", "vol_ratio_20", "ma_dist_20", "ma_dist_50",
    "pos_52w", "gap_prev", "deliv_z", "turnover_z",
    "oi_change_z", "basis_pct", "fii_5d_z",
    "overnight_spx", "asia_ret", "india_vix", "usdinr_chg", "crude_chg",
]

# representative _present flag per optional feature block
BLOCK_FLAGS = {"fno": "oi_change_z_present", "global": "overnight_spx_present",
               "macro": "days_to_macro_present", "sentiment": "sent_mean_5d_present"}

# what feeds each step — shown in the trace UI header so "which source?"
# never needs asking. News is deliberately listed as NOT part of this pipeline.
DATA_SOURCES = {
    "fetch_bhavcopy": "NSE official archives (nsearchives.nseindia.com) — EOD equity + F&O",
    "fetch_global": "Yahoo Finance (yfinance) — one ticker per series",
    "panel": "local bhavcopy store → per-symbol daily parquet",
    "score": "feature schema daily-v1 (~40 indicators/stock) → LightGBM swing_1_5d + isotonic calibration",
    "rank": "calibrated P(win) desc, momentum tie-break (ret_20d, ret_5d)",
    "news": "NOT part of this pipeline (sentiment block disabled in config_daily). "
            "News is fetched on demand by the Analyzer (GDELT DOC API, 30-day window).",
}


def _trace_file(status_dir: Path | None = None) -> Path:
    return (status_dir or daily_path("reports")) / "pipeline_trace.jsonl"


class Trace:
    """Writer. Truncates the journal and stamps a _meta event on creation."""

    def __init__(self, status_dir: Path | None = None):
        self._file = _trace_file(status_dir)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._file, "w")
        self.event("_meta", "run", "ok", started=datetime.now().isoformat(timespec="seconds"),
                   data_sources=DATA_SOURCES)

    def event(self, step: str, item: str, status: str, **detail) -> None:
        rec = {"ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
               "step": step, "item": str(item), "status": status, **detail}
        try:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001 — tracing must never kill the pipeline
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


def read_trace(step: str | None = None, item: str | None = None,
               limit: int = 3000, status_dir: Path | None = None) -> list[dict]:
    f = _trace_file(status_dir)
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if step and rec.get("step") != step:
            continue
        if item and item.upper() not in rec.get("item", "").upper():
            continue
        out.append(rec)
    return out[-limit:]
