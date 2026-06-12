"""Event-driven purged walk-forward backtest (PLAN_v3 Sections 12, 19 Gate 3).

Per fold: train models on the fold's train window (with an internal chronological
OOF tail for blend-weight + isotonic fitting), then replay each test day:
screen 09:30 → features per decision bar → blend → calibrate → ACI gate →
fill next bar open → triple-barrier outcome on the 1-min path → costs → daily
ACI update. One trade per symbol per day.

Outputs: reports/v3/backtest_<ts>.json + trades parquet. Includes the executable
LEAKAGE ALARM: win rate > 92% at coverage > 10% fails the run.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config, load_universe
from src.intraday.blend import Blender
from src.intraday.conformal_gate import ACIGate
from src.intraday.costs import round_trip_cost_pct
from src.intraday.features import build_matrix, check_schema, FEATURE_ORDER
from src.intraday.flow_model import FlowModel
from src.intraday.labeler import geometric_baseline, label_day, outcomes_frame
from src.intraday.price_model import PriceModel
from src.intraday.risk import is_blackout
from src.intraday.screener import screen_day

logger = logging.getLogger(__name__)


class LeakageAlarm(RuntimeError):
    pass


# ── dataset assembly ──────────────────────────────────────────────────

def build_dataset(days: list[date], symbols: list[str]) -> pd.DataFrame:
    """Screen → features → labels for a list of trading days.
    Returns one row per (symbol, decision bar) with feature columns + label."""
    rows = []
    for day in days:
        if is_blackout(day):
            continue
        try:
            top = screen_day(day, symbols)
        except Exception as e:  # noqa: BLE001
            logger.warning("dataset: screen %s failed (%s)", day, e)
            continue
        plan = top.assign(date=day)[["symbol", "date", "direction"]]
        feats = build_matrix(plan)
        if feats.empty:
            continue
        for sym, g in feats.groupby("symbol"):
            direction = int(plan[plan.symbol == sym]["direction"].iloc[0])
            outs = outcomes_frame(label_day(sym, day, {ts: direction for ts in g["ts"]}))
            if outs.empty:
                continue
            # decision bar ts + 15min == entry_ts → join on that
            g = g.copy()
            g["entry_ts"] = g["ts"] + pd.Timedelta("15min")
            merged = g.merge(outs[["entry_ts", "label", "pnl_pct", "barrier", "entry"]],
                             on="entry_ts", how="inner")
            rows.append(merged)
    if not rows:
        raise RuntimeError("dataset assembly produced 0 rows — check bar store coverage")
    out = pd.concat(rows, ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    logger.info("dataset: %d rows, %d days, base win rate %.3f",
                len(out), out.date.nunique(), out.label.mean())
    return out


# ── fold machinery ────────────────────────────────────────────────────

def make_folds(all_days: list[date]) -> list[tuple[list[date], list[date]]]:
    """Expanding-window folds: train ≥ train_min_months, test = test_fold_months,
    embargo_days dropped from the train end (purge handled at row level too)."""
    tr_cfg = load_config()["training"]
    days = sorted(all_days)
    start = days[0]
    folds = []
    test_len = tr_cfg["test_fold_months"]
    first_test = start + timedelta(days=int(tr_cfg["train_min_months"] * 30.44))
    t0 = first_test
    while t0 < days[-1]:
        t1 = t0 + timedelta(days=int(test_len * 30.44))
        embargo = timedelta(days=tr_cfg["embargo_days"])
        train = [d for d in days if d < t0 - embargo]
        test = [d for d in days if t0 <= d < t1]
        if train and test:
            folds.append((train, test))
        t0 = t1
    if not folds:
        raise RuntimeError("not enough history for a single walk-forward fold")
    return folds


def fit_fold(train_df: pd.DataFrame) -> tuple[PriceModel, FlowModel, Blender]:
    """Train models on the fold. Internal chronological 80/20: models on the
    first 80%, blend weight + isotonic on the 20% OOF tail; then refit on 100%."""
    cut = int(len(train_df) * 0.8)
    head, tail = train_df.iloc[:cut], train_df.iloc[cut:]

    pm = PriceModel().fit(head, head["label"])
    fm = FlowModel().fit(head, head["label"])
    blender = Blender()
    p_p, p_f = pm.predict_proba(tail), fm.predict_proba(tail)
    blender.fit_weight(p_p, p_f, tail["label"].values)
    blender.fit_calibration(blender.blend(p_p, p_f), tail["label"].values)

    pm = PriceModel(pm.params).fit(train_df, train_df["label"])   # full-train refit
    fm = FlowModel(fm.params).fit(train_df, train_df["label"])
    return pm, fm, blender


# ── the walk-forward run ──────────────────────────────────────────────

def run_backtest(start: date, end: date, stress: bool = False,
                 capital_per_trade_inr: float = 100_000) -> dict:
    cfg = load_config()
    symbols = load_universe()["symbol"].tolist()
    all_days = pd.bdate_range(start, end).date.tolist()

    logger.info("assembling dataset %s → %s (%d candidate days)", start, end, len(all_days))
    data = build_dataset(all_days, symbols)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    folds = make_folds(sorted(data["date"].unique()))
    logger.info("walk-forward: %d folds, geometric baseline %.3f", len(folds), geometric_baseline())

    trades = []
    fold_stats = []
    for k, (train_days, test_days) in enumerate(folds, 1):
        train_df = data[data.date.isin(train_days)]
        check_schema(train_df)
        pm, fm, blender = fit_fold(train_df)
        gate = ACIGate()  # fresh τ per fold, adapts through the test window

        for day in test_days:
            day_rows = data[data.date == day]
            if day_rows.empty:
                continue
            p_p, p_f = pm.predict_proba(day_rows), fm.predict_proba(day_rows)
            p_cal = blender.calibrated(p_p, p_f)
            mask = gate.fire(p_cal, p_p, p_f)
            fired = day_rows[mask].copy()
            fired["p_cal"] = p_cal[mask]
            # one trade per symbol/day: first fired decision bar
            fired = fired.sort_values("entry_ts").groupby("symbol", as_index=False).first()
            day_labels = []
            for _, r in fired.iterrows():
                qty = max(1, int(capital_per_trade_inr / r["entry"]))
                cost = round_trip_cost_pct(r["entry"], qty, stress=stress)
                trades.append({
                    "fold": k, "date": day, "symbol": r["symbol"],
                    "entry_ts": r["entry_ts"], "p_cal": r["p_cal"],
                    "label": int(r["label"]), "barrier": r["barrier"],
                    "pnl_pct_gross": r["pnl_pct"], "cost_pct": cost,
                    "pnl_pct_net": r["pnl_pct"] - cost,
                })
                day_labels.append(int(r["label"]))
            gate.update(day_labels)

        fold_trades = [t for t in trades if t["fold"] == k]
        if fold_trades:
            ft = pd.DataFrame(fold_trades)
            fold_stats.append({
                "fold": k, "n_trades": len(ft),
                "win_rate": float(ft.label.mean()),
                "expectancy_pct": float(ft.pnl_pct_net.mean()),
                "signals_per_day": len(ft) / max(1, len(test_days)),
            })
            logger.info("fold %d: %s", k, fold_stats[-1])

    return _finalize(trades, fold_stats, data, stress)


def _finalize(trades: list[dict], fold_stats: list[dict], data: pd.DataFrame, stress: bool) -> dict:
    cfg = load_config()
    g = cfg["gates"]
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        raise RuntimeError("backtest emitted 0 trades — gate too tight or edge absent (Gate 3 fails honestly)")

    test_rows = len(data[data.date.isin(tdf.date.unique())])
    coverage = len(tdf) / max(1, test_rows)
    win_rate = float(tdf.label.mean())
    expectancy = float(tdf.pnl_pct_net.mean())
    daily = tdf.groupby("date")["pnl_pct_net"].sum()
    equity = (1 + daily).cumprod()
    dd = float((equity / equity.cummax() - 1).min())

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stress_slippage": stress,
        "geometric_baseline": geometric_baseline(),
        "n_trades": len(tdf),
        "win_rate": win_rate,
        "edge_over_baseline": win_rate - geometric_baseline(),
        "expectancy_pct_net": expectancy,
        "signals_per_day": float(tdf.groupby("date").size().mean()),
        "coverage_of_candidates": coverage,
        "profit_factor": float(tdf[tdf.pnl_pct_net > 0].pnl_pct_net.sum()
                               / abs(tdf[tdf.pnl_pct_net < 0].pnl_pct_net.sum() or 1)),
        "max_drawdown": dd,
        "worst_day_pct": float(daily.min()),
        "fold_stats": fold_stats,
        "gates": {
            "win_rate_ok": win_rate >= g["min_win_rate"],
            "expectancy_ok": expectancy >= g["min_expectancy_pct"],
            "signals_ok": float(tdf.groupby("date").size().mean()) >= g["min_signals_per_day"],
        },
    }

    # ── executable leakage alarm (Section 12.2) ───────────────────────
    if win_rate > g["leakage_alarm_win_rate"] and coverage > g["leakage_alarm_coverage"]:
        raise LeakageAlarm(
            f"win rate {win_rate:.3f} at coverage {coverage:.3f} — investigate lookahead before trusting this run"
        )

    out_dir = ROOT / cfg["paths"]["reports"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"backtest_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    tdf.to_parquet(out_dir / f"trades_{stamp}.parquet", index=False)
    logger.info("backtest report → %s", out_dir / f"backtest_{stamp}.json")
    return report
