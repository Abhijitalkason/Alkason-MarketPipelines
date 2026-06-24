"""Production training pipeline (v3_supplementry §9, B-1).

train_production builds the dataset over the train window, decides flow-channel
activation from the real-data share, fits screen weights, fits the price (+flow
when active) models with a day-boundary inner calibration tail, saves the
bundle, logs to MLflow, registers a Staging version, and DVC-pushes — every
step journaled. Promotion to Production is a separate, operator-confirmed CLI
action so retraining can never silently go live.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.intraday import ROOT, load_config, today_ist
from src.intraday.backtester import build_dataset, fit_fold, trading_sessions
from src.intraday.blend import Blender
from src.intraday.conformal_gate import ACIGate
from src.intraday.features import FLOW_PRESENT, flow_real_share
from src.intraday.journal import journal_write
from src.intraday.screener import save_weights

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    flow_active: bool
    flow_real_share: float
    n_rows: int
    data_span: tuple[date, date]
    calib_brier: float
    version: str | None = None


def _state_dir() -> Path:
    p = ROOT / load_config()["paths"]["state"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_production(end_day: date | None = None, train_months: int | None = None,
                     tune: bool | None = None, push_dvc: bool = True,
                     track: bool = True) -> TrainResult:
    cfg = load_config()
    end_day = end_day or today_ist()
    train_months = train_months or cfg["training"]["train_min_months"]
    start_day = end_day - timedelta(days=int(train_months * 30.44))
    days = trading_sessions(start_day, end_day)  # real NSE sessions, not raw weekdays

    # 1–2. dataset (all integrity checks live in build_dataset; flow inactive so
    # build_matrix does not yet require the flow floor)
    data = build_dataset(days)
    data["date"] = pd.to_datetime(data["date"]).dt.date

    # 3. flow activation decision
    share = flow_real_share(data) if all(c in data.columns for c in FLOW_PRESENT) else 0.0
    activate = share >= cfg["gate"]["flow_activation_share"]
    logger.info("flow_real_share=%.3f → flow_model_active=%s", share, activate)
    _set_flow_active(activate)

    # 4. screen weights on the full train window
    from src.intraday.backtester import _fit_fold_screen_weights
    weights = _fit_fold_screen_weights(data)
    if weights:
        save_weights(weights)

    # 5. fit (single "fold" = the full window with inner calib tail)
    pm, fm, blender, record = fit_fold(data, tune=tune)

    # 6. save bundle
    state = _state_dir()
    pm.save(state / "price_model.joblib")
    if fm is not None:
        fm.save(state / "flow_model.joblib")
    blender.save(state / "blender.joblib")
    _save_fresh_gate(state)
    data.to_parquet(state / "feature_matrix.parquet", index=False)  # drift reference
    _write_model_card(state, share, activate, record, (start_day, end_day), len(data))

    result = TrainResult(flow_active=activate, flow_real_share=share, n_rows=len(data),
                         data_span=(start_day, end_day), calib_brier=record["calib_brier"])

    # 7–8. MLflow run + Staging registration
    if track:
        try:
            from src.intraday.tracking import log_training
            metrics = {"flow_real_share": share, "calib_brier": record["calib_brier"],
                       "n_rows": float(len(data))}
            params = {"flow_active": activate, "train_months": train_months,
                      "blend_weight": record["weight"],
                      "data_span": f"{start_day}..{end_day}"}
            result.version = log_training({"metrics": metrics, "params": params}, state)
        except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
            logger.error("MLflow logging failed: %s", e)
            raise

    # 9. DVC push (raises on failure)
    if push_dvc:
        from src.intraday.dvc_utils import add_and_push, is_initialized
        if is_initialized():
            add_and_push(["models/v3", "data/bars", "data/bhavcopy"])
        else:
            logger.warning("DVC not initialised — skipping push (init before Gate 1)")

    journal_write("train", {"version": result.version, "flow_active": activate,
                            "flow_real_share": share, "n_rows": len(data),
                            "span": f"{start_day}..{end_day}"})
    logger.info("train_production done: %s", result)
    return result


def _set_flow_active(active: bool) -> None:
    """Persist the flow-activation decision back into config so blend/gate/features
    read a consistent flag. Surgical, comment-preserving, atomic:
      - edits only the single `flow_model_active:` line (a full YAML re-dump would
        strip every comment in the file);
      - writes via a temp file + os.replace so a concurrent reader never sees a
        half-written config.
    Note: load_config.cache_clear() only affects THIS process — a separately
    running serve/paper process keeps its cached flag until restarted (training
    and serving are distinct lifecycle events; the bundle also carries the flag)."""
    import os
    import re

    from src.intraday import CONFIG_PATH

    text = CONFIG_PATH.read_text()
    new = re.sub(r"(^\s*flow_model_active:\s*)(true|false)",
                 lambda m: f"{m.group(1)}{str(active).lower()}", text, count=1, flags=re.M)
    if new == text:
        return  # already at the desired value (or key absent)
    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(new)
    os.replace(tmp, CONFIG_PATH)
    load_config.cache_clear()


def _save_fresh_gate(state: Path) -> None:
    """Fresh τ₀ unless a live gate state already exists — never clobber live
    state (v3_supplementry §9.1 step 6)."""
    if not (state / "gate_state.json").exists():
        ACIGate().save()


def _write_model_card(state: Path, share: float, active: bool, record: dict,
                      span: tuple[date, date], n_rows: int) -> None:
    from src.intraday.tracking import config_hash, git_sha
    from src.intraday.dvc_utils import current_rev

    cfg = load_config()
    excl_path = state / "excluded_symbols.json"
    excluded = json.loads(excl_path.read_text()) if excl_path.exists() else []
    card = f"""# Model Card — intraday-v3-signal

**Intended use:** 2–3h NSE large-cap selective trading signals; entries
09:30–11:00 IST, square-off 14:45, always flat overnight.

## Training data
- Span: {span[0]} → {span[1]}
- Rows: {n_rows}
- flow_real_share: {share:.3f} → flow channel **{'ACTIVE' if active else 'OFF (price-only)'}**

## Geometry & gate
- target/stop ATR: {cfg['geometry']['target_atr']}/{cfg['geometry']['stop_atr']}
- time barrier: {cfg['geometry']['time_barrier_hours']}h, square-off {cfg['geometry']['squareoff']}
- τ₀: {cfg['gate']['fire_threshold_init']}, target win rate {cfg['gate']['target_win_rate']}
- agreement floor: {cfg['gate']['model_agreement_floor']}

## Metrics
- calibration-tail Brier: {record['calib_brier']:.4f}
- blend weight (price): {record['weight']:.2f}

## Exclusions
- per-symbol coverage exclusions: {excluded or 'none'}

## Known limitations
- flow channel {'on' if active else 'OFF until flow_real_share ≥ '
                + str(cfg['gate']['flow_activation_share'])}
- pre-open / L2 archive depth grows forward only
- index_alignment uses the universe-median proxy (no index bars yet)

## Provenance
- git SHA: {git_sha()}
- config hash: {config_hash()}
- DVC rev: {current_rev()}
"""
    (state / "model_card.md").write_text(card)


# ── promotion / rollback (approval gate, §9.2) ────────────────────────

def promote(version: str, confirm: bool = False) -> dict:
    """Staging → Production. Requires a passing Gate-3 report and paper evidence
    on disk plus an explicit --confirm; retraining can never silently go live."""
    cfg = load_config()
    if not confirm:
        raise RuntimeError("promotion requires --confirm (operator approval gate)")
    gate3 = _latest_gate_result("gate3")
    if not (gate3 and gate3.get("passed")):
        raise RuntimeError("no passing Gate-3 result on disk — run `--mode gates` first")
    paper_days = _count_paper_sessions()
    if paper_days < cfg["gates"]["paper_days"]:
        raise RuntimeError(
            f"only {paper_days} paper sessions (< {cfg['gates']['paper_days']}) — Gate 4 not met"
        )
    from src.intraday.tracking import transition
    transition(version, "Production")
    journal_write("promote", {"version": version, "gate3": gate3.get("stamp"),
                             "paper_days": paper_days})
    return {"status": "promoted", "version": version}


def rollback(confirm: bool = False) -> dict:
    """Re-stage the previous Production version (demote current). Journaled."""
    if not confirm:
        raise RuntimeError("rollback requires --confirm")
    import mlflow
    cfg = load_config()
    name = cfg["mlflow"]["registry_model_name"]
    client = mlflow.tracking.MlflowClient()
    prod = [mv for mv in client.search_model_versions(f"name='{name}'")
            if mv.current_stage == "Production"]
    if not prod:
        raise RuntimeError("no Production version to roll back")
    current = max(prod, key=lambda mv: int(mv.version))
    from src.intraday.tracking import transition
    transition(current.version, "Archived")
    others = sorted((mv for mv in client.search_model_versions(f"name='{name}'")
                     if mv.version != current.version), key=lambda mv: int(mv.version), reverse=True)
    if others:
        transition(others[0].version, "Production")
    journal_write("rollback", {"demoted": current.version,
                              "restored": others[0].version if others else None})
    return {"status": "rolled_back", "demoted": current.version}


def _latest_gate_result(name: str) -> dict | None:
    p = Path(load_config()["paths"]["gates"])
    p = p if p.is_absolute() else ROOT / p
    files = sorted(p.glob(f"{name}_*.json")) if p.exists() else []
    return json.loads(files[-1].read_text()) if files else None


def _count_paper_sessions() -> int:
    p = Path(load_config()["paths"]["signals_log"])
    p = p if p.is_absolute() else ROOT / p
    return len(list(p.glob("paper_*.json"))) if p.exists() else 0


# ── rolling recalibration (H-4) ───────────────────────────────────────

def recalibrate() -> dict:
    """Refit isotonic on the trailing calibration_window_days of MATURED live/
    paper signals (features+outcomes from the journal/signals log). Aborts
    (no-op + alert) below recalib_min_signals. Journaled."""
    cfg = load_config()
    win = cfg["gate"]["calibration_window_days"]
    min_n = cfg["training"]["recalib_min_signals"]
    sig = _load_matured_signals(win)
    if len(sig) < min_n:
        logger.warning("recalibrate: only %d matured signals (< %d) — aborting", len(sig), min_n)
        journal_write("recalibrate", {"status": "aborted", "n": len(sig)})
        return {"status": "aborted", "n": len(sig)}

    state = _state_dir()
    blender = Blender.load(state / "blender.joblib")
    from src.intraday.price_model import PriceModel
    pm = PriceModel.load(state / "price_model.joblib")
    X = pd.DataFrame(list(sig["features"]))
    p_price = pm.predict_proba(X)
    p_flow = None
    if blender.flow_active and (state / "flow_model.joblib").exists():
        from src.intraday.flow_model import FlowModel
        fm = FlowModel.load(state / "flow_model.joblib")
        p_flow = fm.predict_proba(X)
    y = sig["label"].to_numpy()
    before = float(((blender.calibrated(p_price, p_flow) - y) ** 2).mean())
    blender.fit_calibration(blender.blend(p_price, p_flow), y)
    after = float(((blender.calibrated(p_price, p_flow) - y) ** 2).mean())
    blender.save(state / "blender.joblib")
    out = {"status": "ok", "n": len(sig), "brier_before": before, "brier_after": after}
    journal_write("recalibrate", out)
    logger.info("recalibrate: Brier %.4f → %.4f over %d signals", before, after, len(sig))
    return out


def _load_matured_signals(window_days: int) -> pd.DataFrame:
    """Matured paper/live signals with feature vectors + labels from the signals
    log over the trailing window."""
    cfg = load_config()
    sig_dir = Path(cfg["paths"]["signals_log"])
    sig_dir = sig_dir if sig_dir.is_absolute() else ROOT / sig_dir
    cutoff = today_ist() - timedelta(days=window_days)
    rows = []
    for f in sorted(sig_dir.glob("paper_*.json")):
        try:
            d = date.fromisoformat(f.stem.replace("paper_", ""))
        except ValueError:
            continue
        if d < cutoff:
            continue
        for rec in json.loads(f.read_text()):
            if rec.get("features") and "label" in rec:
                rows.append({"features": rec["features"], "label": int(rec["label"])})
    return pd.DataFrame(rows)
