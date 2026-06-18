"""Drift detection → ACI recalibration flag → halt (v3_supplementry §12.4,
B-7 / PLAN_v3 §13 switch c).

reference = production model's feature_matrix.parquet (training artifact)
current   = last drift.current_window_sessions of live feature rows (journal)
report    = Evidently DataDriftPreset → reports/v3/drift/<date>.{html,json}
breach    = share of drifted columns ≥ drift.share_drifted_columns_alert
on breach : (1) force ACI recalibration flag for next session
            (2) append alert to reports/v3/alerts/<date>.json
            (3) consecutive breaches ≥ threshold → _persist_halt(["drift"])

Runs post-market from the paper runner and standalone via `--mode drift`.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.intraday import ROOT, load_config, now_ist, today_ist
from src.intraday.features import PRICE_FEATURES

logger = logging.getLogger(__name__)

RECALIB_FLAG = "recalibrate_next_session.flag"


class DriftError(RuntimeError):
    pass


def _state_dir() -> Path:
    p = ROOT / load_config()["paths"]["state"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _drift_dir() -> Path:
    p = Path(load_config()["paths"]["drift"])
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reference_matrix() -> pd.DataFrame:
    p = _state_dir() / "feature_matrix.parquet"
    if not p.exists():
        raise DriftError("drift reference (models/v3/feature_matrix.parquet) missing — train first")
    return pd.read_parquet(p)


def _current_window() -> pd.DataFrame:
    """Recent live feature rows from the signals log (paper/live fills carry the
    full feature vector)."""
    cfg = load_config()
    sig_dir = Path(cfg["paths"]["signals_log"])
    sig_dir = sig_dir if sig_dir.is_absolute() else ROOT / sig_dir
    n = cfg["drift"]["current_window_sessions"]
    files = sorted(sig_dir.glob("paper_*.json"))[-n:]
    rows = []
    for f in files:
        for rec in json.loads(f.read_text()):
            feats = rec.get("features", {})
            if feats:
                rows.append(feats)
    return pd.DataFrame(rows)


def run_check(day: date | None = None) -> dict:
    """Compute drift, write reports, set recalib flag / alert / halt as needed.
    Returns a small summary dict."""
    from src.intraday.risk import _persist_halt

    cfg = load_config()["drift"]
    day = day or today_ist()
    ref = _reference_matrix()
    cur = _current_window()
    cols = [c for c in PRICE_FEATURES if c in ref.columns and c in cur.columns]
    if cur.empty or len(cols) < 3:
        logger.info("drift: insufficient live rows (%d) — skipping", len(cur))
        return {"checked": False, "n_current": len(cur)}

    share, html_path = _evidently_drift(ref[cols], cur[cols], day)
    breach = share >= cfg["share_drifted_columns_alert"]
    summary = {"checked": True, "share_drifted": share, "breach": breach,
               "n_current": len(cur), "report": str(html_path)}
    (_drift_dir() / f"{day.isoformat()}.json").write_text(json.dumps(summary, indent=2, default=str))

    if breach:
        _set_recalib_flag()
        _append_alert(day, summary)
        if _consecutive_breaches(day) >= cfg["consecutive_breaches_to_halt"]:
            _persist_halt(["drift"])
            summary["halted"] = True
    else:
        _clear_breach_streak()
    from src.intraday.journal import journal_write
    journal_write("drift", summary, day=day)
    return summary


def _evidently_drift(ref: pd.DataFrame, cur: pd.DataFrame, day: date) -> tuple[float, Path]:
    """Run Evidently DataDriftPreset; return (share_drifted_columns, html path).
    Evidently import failure raises (drift is a kill-switch input, not optional)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception as e:  # noqa: BLE001
        raise DriftError(f"Evidently unavailable — drift is a kill-switch input: {e}")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)
    html_path = _drift_dir() / f"{day.isoformat()}.html"
    try:
        snapshot.save_html(str(html_path))
    except Exception:  # noqa: BLE001 — html is a nicety; the share is what gates
        pass
    result = snapshot.dict() if hasattr(snapshot, "dict") else {}
    share = _extract_share(result, ref.columns)
    return share, html_path


def _extract_share(result: dict, columns) -> float:
    """Pull the drifted-column share out of an Evidently snapshot dict, tolerant
    of version differences."""
    for metric in result.get("metrics", []):
        val = metric.get("value", {})
        if isinstance(val, dict):
            if "share_of_drifted_columns" in val:
                return float(val["share_of_drifted_columns"])
            if "drifted_columns" in val and "number_of_columns" in val:
                return float(val["drifted_columns"]) / max(1, float(val["number_of_columns"]))
    return 0.0


def _set_recalib_flag() -> None:
    (_state_dir() / RECALIB_FLAG).write_text(now_ist().isoformat())


def recalib_flag_set() -> bool:
    return (_state_dir() / RECALIB_FLAG).exists()


def clear_recalib_flag() -> None:
    f = _state_dir() / RECALIB_FLAG
    if f.exists():
        f.unlink()


def _append_alert(day: date, summary: dict) -> None:
    p = Path(load_config()["paths"]["alerts"])
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{day.isoformat()}.json"
    existing = json.loads(f.read_text()) if f.exists() else []
    existing.append({"ts": now_ist().isoformat(), "kind": "drift", **summary})
    f.write_text(json.dumps(existing, indent=2, default=str))


def _streak_file() -> Path:
    return _state_dir() / "drift_breach_streak.json"


def _consecutive_breaches(day: date) -> int:
    """Count consecutive breaching SESSIONS, keyed by date — so running the
    drift check twice in one day (e.g. `--mode drift` plus the paper post-market)
    cannot double-increment the streak and trip the halt on a single day."""
    f = _streak_file()
    state = json.loads(f.read_text()) if f.exists() else {"streak": 0, "last_day": None}
    if state.get("last_day") == day.isoformat():
        return state["streak"]            # already counted this session
    n = state.get("streak", 0) + 1
    f.write_text(json.dumps({"streak": n, "last_day": day.isoformat()}))
    return n


def _clear_breach_streak() -> None:
    _streak_file().write_text(json.dumps({"streak": 0, "last_day": None}))
