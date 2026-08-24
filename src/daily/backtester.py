"""Daily purged walk-forward backtest + the STAGED gate (the decisive artifact).

Per fold: train on the fold's train window with a day-boundary inner calibration
tail (isotonic via the reused intraday Blender, single-channel), refit on the full
window with frozen params, predict each test day OOS. Then score the OOS panel:

  Milestone 1 (PREDICTIVE — gates whether the list ships at all):
    OOS AUC vs 0.5 (+ per-fold consistency), log-loss vs 0.693, calibration gap,
    and precision@N / lift for the top-N daily BUY list. NO costs read here.
  Milestone 2 (AFTER-COST — gates the 'actionable' label):
    net-of-cost expectancy of the top-N picks across folds, base + stress.

Outputs: reports/daily/backtest_<horizon>_<ts>.json + OOS parquet + reliability/
precision PNGs + one append-only run_registry line (git sha + config hash → any
gate-loosening is visible). The gate is ALLOWED to say "no" — that is the honest
finding, exactly as the intraday system found for its horizon.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.daily import (ROOT, daily_path, horizon_config, load_daily_config,
                       load_universe, now_ist)
from src.daily.costs import round_trip_cost_pct
from src.daily.features import build_matrix, check_schema
from src.daily.labeler import geometric_baseline, label_day, outcomes_frame
from src.daily.model import DailyModel
from src.daily.panel import load_symbol_daily, trading_days
from src.intraday.blend import Blender

logger = logging.getLogger(__name__)

LOG_LOSS_FLOOR = float(np.log(2))   # 0.6931 — no-skill binary log-loss
# Leakage alarm: a daily direction model materially above this is implausible —
# investigate before trusting (mirrors the intraday leakage alarm philosophy).
LEAKAGE_AUC_ALARM = 0.80


class LeakageAlarm(RuntimeError):
    pass


# ── dataset assembly ──────────────────────────────────────────────────

def _load_raw_panels(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    base = daily_path("daily_path")
    for s in symbols:
        p = base / f"{s}.parquet"
        if p.exists():
            out[s] = pd.read_parquet(p)
    return out


def build_dataset(start: date, end: date, horizon: str = "swing_1_5d") -> pd.DataFrame:
    """Assemble one row per (symbol, decision day): daily features + triple-barrier
    label/pnl for `horizon`. Universe is point-in-time per day (survivorship-proof).
    """
    cfg = load_daily_config()
    days = trading_days(start, end)
    if not days:
        raise RuntimeError(f"no trading days on disk for {start}..{end}")
    # monthly PIT membership: the universe is recomputed each month (top-100 by
    # trailing turnover), so a row only counts when its symbol was a member in
    # its month — survivorship-proof AND selection-drift-proof.
    month_starts = sorted({date(d.year, d.month, 1) for d in days})
    membership: dict[date, set[str]] = {}
    for m in month_starts:
        try:
            membership[m] = set(load_universe(as_of=m)["symbol"])
        except ValueError:                 # months before the first universe month
            membership[m] = set()
    all_syms = sorted(set().union(*membership.values()))
    panels = _load_raw_panels(all_syms)
    if not panels:
        raise RuntimeError("no per-symbol daily panels on disk — run build_panel first")

    from src.daily.features import features_for_symbol
    from src.daily.labeler import label_symbol

    merged_frames, failed = [], 0
    for sym, panel in panels.items():
        try:
            feats = features_for_symbol(sym, days, df_daily=panel)
            if feats.empty:
                continue
            labels = label_symbol(sym, days, horizon, df_daily=panel)
            if not labels:
                continue
            ldf = pd.DataFrame([
                {"date": pd.Timestamp(d), "label": o.label, "pnl_pct": o.pnl_pct,
                 "entry": o.entry, "barrier": o.barrier, "entry_ts": o.entry_ts}
                for d, o in labels.items()
            ])
            merged = feats.merge(ldf, on="date", how="inner")
            if not merged.empty:
                merged_frames.append(merged)
        except Exception as e:  # noqa: BLE001 — counted; raises over tolerance
            failed += 1
            logger.warning("dataset: %s failed (%s)", sym, e)
    if len(panels) and failed > len(panels) * cfg["data"]["max_skip_share"]:
        raise RuntimeError(f"{failed}/{len(panels)} symbols failed assembly (> tolerance)")
    if not merged_frames:
        raise RuntimeError("dataset assembly produced 0 rows — check the daily panel")
    df = pd.concat(merged_frames, ignore_index=True)
    from src.daily.features import MARKET_FEATURES
    n0 = len(df)
    df = df.dropna(subset=MARKET_FEATURES).reset_index(drop=True)   # belt-and-suspenders
    if n0 - len(df):
        logger.info("dataset: dropped %d/%d rows with incomplete market features", n0 - len(df), n0)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # keep only rows whose symbol was a universe member in that row's month
    mkey = df["date"].map(lambda d: date(d.year, d.month, 1))
    keep = pd.Series([s in membership.get(m, set())
                      for s, m in zip(df["symbol"], mkey)], index=df.index)
    n_pre = len(df)
    df = df[keep].reset_index(drop=True)
    if n_pre - len(df):
        logger.info("dataset: dropped %d/%d rows outside monthly PIT membership",
                    n_pre - len(df), n_pre)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    logger.info("daily dataset: %d rows, %d days, base up-rate %.3f (%d skips)",
                len(df), df["date"].nunique(), df["label"].mean(), failed)
    return df


# ── fold machinery ────────────────────────────────────────────────────

def make_folds(all_days: list[date], horizon: str) -> list[tuple[list[date], list[date]]]:
    """Expanding-window day folds: train ≥ train_min_months, test = test_fold_months,
    embargo = max(training.embargo_days, horizon.embargo_days). Raises if < min_folds."""
    tr = load_daily_config()["training"]
    embargo_days = max(tr["embargo_days"], horizon_config(horizon)["embargo_days"])
    days = sorted(all_days)
    first_test = days[0] + timedelta(days=int(tr["train_min_months"] * 30.44))
    test_len = tr["test_fold_months"]
    folds, t0 = [], first_test
    while t0 < days[-1]:
        t1 = t0 + timedelta(days=int(test_len * 30.44))
        train = [d for d in days if d < t0 - timedelta(days=embargo_days)]
        test = [d for d in days if t0 <= d < t1]
        if train and test:
            folds.append((train, test))
        t0 = t1
    if len(folds) < tr["min_folds"]:
        raise RuntimeError(
            f"only {len(folds)} folds < min_folds={tr['min_folds']} — need more history "
            f"(≥{tr['train_min_months']}m train + {tr['min_folds']}×{test_len}m OOS)"
        )
    return folds


def _inner_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_daily_config()["training"]
    days = sorted(pd.to_datetime(train_df["date"]).dt.date.unique())
    calib_days = days[-cfg["inner_calib_days"]:]
    calib_start = calib_days[0]
    head_days = [d for d in days if d < calib_start - timedelta(days=cfg["embargo_days"])]
    head = train_df[pd.to_datetime(train_df["date"]).dt.date.isin(head_days)]
    tail = train_df[pd.to_datetime(train_df["date"]).dt.date.isin(calib_days)]
    return head, tail


def fit_fold(train_df: pd.DataFrame, tune: bool | None = None) -> tuple[DailyModel, Blender, dict]:
    """Train one fold: head/calib split → fit on head → OOF on tail → isotonic
    calibrator on tail OOF → refit on FULL train with frozen params. The forbidden
    pattern (calibrator on in-fold predictions) is structurally excluded."""
    cfg = load_daily_config()
    tune = cfg["training"]["tune"] if tune is None else tune
    head, tail = _inner_split(train_df)
    if head.empty or tail.empty:
        raise RuntimeError("inner split produced an empty head or calibration tail")

    pm = DailyModel()
    if tune:
        pm.tune(head, head["label"])
    pm.fit(head, head["label"])
    p_tail = pm.predict_proba(tail)
    blender = Blender(flow_active=False)          # single channel → isotonic only
    blender.fit_calibration(p_tail, tail["label"].values)
    calib_brier = float(np.mean((blender.calibrated(p_tail) - tail["label"].values) ** 2))

    pm = DailyModel(pm.params).fit(train_df, train_df["label"])   # full-train refit, frozen
    return pm, blender, {"calib_brier": calib_brier, "params": pm.params}


# ── the walk-forward run + staged gate ────────────────────────────────

def run_backtest(start: date, end: date, horizon: str = "swing_1_5d",
                 capital_per_trade_inr: float = 100_000, tune: bool | None = None,
                 args_note: str = "") -> dict:
    cfg = load_daily_config()
    data = build_dataset(start, end, horizon)
    check_schema(data)
    folds = make_folds(sorted(data["date"].unique()), horizon)
    logger.info("daily walk-forward: %d folds, geometric baseline %.3f",
                len(folds), geometric_baseline(horizon))

    oos_rows, fold_stats = [], []
    for k, (train_days, test_days) in enumerate(folds, 1):
        train_df = data[data.date.isin(train_days)]
        pm, blender, rec = fit_fold(train_df, tune=tune)
        test_df = data[data.date.isin(test_days)].copy()
        if test_df.empty:
            continue
        p = pm.predict_proba(test_df)
        p_cal = blender.calibrated(p)
        test_df = test_df.assign(fold=k, p=p, p_cal=p_cal)
        oos_rows.append(test_df)
        if test_df["label"].nunique() > 1:
            fold_auc = float(roc_auc_score(test_df["label"], test_df["p_cal"]))
        else:
            fold_auc = float("nan")
        fold_stats.append({"fold": k, "n": len(test_df), "auc": fold_auc,
                           "up_rate": float(test_df["label"].mean()),
                           "calib_brier": rec["calib_brier"]})
        logger.info("fold %d: n=%d auc=%.3f", k, len(test_df), fold_auc)

    oos = pd.concat(oos_rows, ignore_index=True)
    return _finalize(oos, fold_stats, horizon, start, end, capital_per_trade_inr, args_note)


def _milestone1(oos: pd.DataFrame, fold_stats: list[dict], horizon: str) -> dict:
    g1 = load_daily_config()["gate"]["milestone1"]
    top_n = g1["top_n"]
    # single-class OOS ⇒ AUC undefined; report NaN (gate fails) rather than a
    # misleading number (mirrors the per-fold guard in run_backtest).
    auc = float(roc_auc_score(oos["label"], oos["p_cal"])) if oos["label"].nunique() > 1 else float("nan")
    ll = float(log_loss(oos["label"], np.clip(oos["p_cal"], 1e-6, 1 - 1e-6), labels=[0, 1]))
    brier = float(brier_score_loss(oos["label"], oos["p_cal"]))
    calib_gap = float(abs(oos["p_cal"].mean() - oos["label"].mean()))
    aucs = [f["auc"] for f in fold_stats if np.isfinite(f["auc"])]
    consistency = float(np.mean([a > 0.5 for a in aucs])) if aucs else 0.0

    # precision@N for the daily BUY list: top-N by p_cal each day, pick-weighted
    # (micro-average) — a day with fewer than N candidates counts by its picks, and
    # it agrees with the live scoreboard's aggregation.
    base_rate = float(oos["label"].mean())
    picked = pd.concat(
        [g.nlargest(min(top_n, len(g)), "p_cal") for _, g in oos.groupby("date")],
        ignore_index=True,
    ) if len(oos) else oos
    precision_at_n = float(picked["label"].mean()) if len(picked) else 0.0
    lift = precision_at_n / base_rate if base_rate > 0 else 0.0

    criteria = {
        "auc_ok": auc >= g1["min_auc"],
        "fold_consistency_ok": consistency >= g1["min_auc_fold_consistency"],
        "calibration_ok": calib_gap <= g1["calibration_max_gap"],
        "lift_ok": lift >= g1["min_precision_at_n_lift"],
    }
    return {
        "oos_auc": auc, "auc_per_fold": aucs, "fold_consistency": consistency,
        "log_loss": ll, "log_loss_floor": LOG_LOSS_FLOOR, "brier": brier,
        "calibration_gap": calib_gap, "base_rate": base_rate,
        "precision_at_n": precision_at_n, "lift": lift, "top_n": top_n,
        "criteria": criteria, "passed": all(criteria.values()),
    }


def _milestone2(oos: pd.DataFrame, horizon: str, capital_per_trade_inr: float) -> dict:
    g2 = load_daily_config()["gate"]["milestone2"]
    top_n = load_daily_config()["gate"]["milestone1"]["top_n"]

    def net_expectancy(stress: bool) -> tuple[float, int, float, float]:
        picks = []
        for _, g in oos.groupby("date"):
            top = g.nlargest(min(top_n, len(g)), "p_cal")
            for _, r in top.iterrows():
                if not (r["entry"] > 0):             # guard bad/zero/NaN entry price
                    continue
                qty = max(1, int(capital_per_trade_inr / r["entry"]))
                cost = round_trip_cost_pct(horizon, r["entry"], qty, stress=stress)
                picks.append({"gross": r["pnl_pct"], "cost": cost,
                              "net": r["pnl_pct"] - cost})
        if not picks:
            return 0.0, 0, 0.0, 0.0
        pdf = pd.DataFrame(picks)
        return float(pdf["net"].mean()), len(pdf), float(pdf["gross"].mean()), float(pdf["cost"].mean())

    net, n_picks, gross, mean_cost = net_expectancy(stress=False)
    net_stress, _, _, _ = net_expectancy(stress=True)
    criteria = {
        "expectancy_ok": net >= g2["min_expectancy_pct"],
        "stress_ok": net_stress >= g2["min_expectancy_pct"],
    }
    return {
        "net_expectancy": net, "net_expectancy_stress": net_stress,
        "gross_expectancy": gross, "mean_cost": mean_cost, "n_picks": n_picks,
        "criteria": criteria, "passed": all(criteria.values()),
    }


def _finalize(oos: pd.DataFrame, fold_stats: list[dict], horizon: str, start: date,
              end: date, capital_per_trade_inr: float, args_note: str) -> dict:
    m1 = _milestone1(oos, fold_stats, horizon)
    m2 = _milestone2(oos, horizon, capital_per_trade_inr)

    if m1["oos_auc"] > LEAKAGE_AUC_ALARM:
        raise LeakageAlarm(
            f"OOS AUC {m1['oos_auc']:.3f} > {LEAKAGE_AUC_ALARM} — implausible at the daily "
            f"direction horizon; investigate lookahead before trusting this run"
        )

    verdict = (
        "M1 PASS — measurable predictive edge; list ships (predictive-only)"
        if m1["passed"] else
        "M1 FAIL — no measurable predictive edge at this horizon/data (honest no)"
    )
    if m1["passed"]:
        verdict += "; " + ("M2 PASS — after-cost edge, list is actionable"
                           if m2["passed"] else "M2 FAIL — not cost-validated, list stays predictive-only")

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "horizon": horizon,
        "window": {"start": str(start), "end": str(end)},
        "n_oos_rows": len(oos), "n_folds": len(fold_stats),
        "geometric_baseline": geometric_baseline(horizon),
        "milestone1_predictive": m1,
        "milestone2_after_cost": m2,
        "fold_stats": fold_stats,
        "verdict": verdict,
    }

    out_dir = daily_path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"backtest_{horizon}_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    oos.to_parquet(out_dir / f"oos_{horizon}_{stamp}.parquet", index=False)
    _write_artifacts(oos, out_dir, horizon, stamp)
    _append_run_registry(report, args_note, stamp)
    logger.info("daily backtest → %s | %s", out_dir / f"backtest_{horizon}_{stamp}.json", verdict)
    return report


def _write_artifacts(oos: pd.DataFrame, out_dir: Path, horizon: str, stamp: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        logger.warning("matplotlib unavailable, skipping PNGs: %s", e)
        return
    # reliability diagram
    fig, ax = plt.subplots()
    bins = np.linspace(oos["p_cal"].min(), oos["p_cal"].max(), 8)
    rel = oos.assign(b=pd.cut(oos["p_cal"], bins)).groupby("b", observed=True).agg(
        pred=("p_cal", "mean"), obs=("label", "mean"))
    ax.plot(rel["pred"], rel["obs"], marker="o", label="model")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="ideal")
    ax.set_xlabel("predicted P(up)"); ax.set_ylabel("realized up-rate")
    ax.set_title(f"Daily calibration — {horizon}"); ax.legend()
    fig.savefig(out_dir / f"reliability_{horizon}_{stamp}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # precision@N vs N (ranking quality)
    fig, ax = plt.subplots()
    ns = list(range(1, 16))
    base = oos["label"].mean()
    prec = []
    for n in ns:
        hits = [g.nlargest(min(n, len(g)), "p_cal")["label"].mean() for _, g in oos.groupby("date")]
        prec.append(np.mean(hits))
    ax.plot(ns, prec, marker="o", label="precision@N")
    ax.axhline(base, ls="--", color="r", label="base up-rate")
    ax.set_xlabel("N (top picks/day)"); ax.set_ylabel("hit rate")
    ax.set_title(f"Daily ranking quality — {horizon}"); ax.legend()
    fig.savefig(out_dir / f"precision_{horizon}_{stamp}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _config_hash() -> str:
    return hashlib.sha256(
        json.dumps(load_daily_config(), sort_keys=True, default=str).encode()).hexdigest()[:12]


def _append_run_registry(report: dict, args_note: str, stamp: str) -> None:
    reg = daily_path("run_registry")
    reg.parent.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    m1, m2 = report["milestone1_predictive"], report["milestone2_after_cost"]
    line = {
        "ts": report["generated"], "stamp": stamp, "horizon": report["horizon"],
        "git_sha": sha, "config_hash": _config_hash(), "args": args_note,
        "oos_auc": m1["oos_auc"], "lift": m1["lift"], "calibration_gap": m1["calibration_gap"],
        "m1_pass": m1["passed"], "net_expectancy": m2["net_expectancy"], "m2_pass": m2["passed"],
    }
    pd.DataFrame([line]).to_csv(reg, mode="a", header=not reg.exists(), index=False)
