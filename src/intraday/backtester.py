"""Event-driven purged walk-forward backtest (PLAN_v3 §12, §19 Gate 3;
v3_supplementry §8).

Per fold: train on the fold's train window with a day-boundary inner
calibration tail (blend weight + isotonic), refit on the full window with
frozen params, then replay each test day: screen 09:30 → features per decision
bar → blend → calibrate → ACI gate → fill next bar open → triple-barrier
outcome on the 1-min path → costs → daily ACI update. One trade per symbol/day.

Outputs: reports/v3/backtest_<ts>.json + trades parquet + frontier/reliability
PNGs; one append-only line to reports/v3/run_registry.csv (M-5). Executable
leakage alarm (per fold AND aggregate); distributional fairness rules (§8.3).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday import ROOT, load_config, load_universe
from src.intraday.blend import Blender
from src.intraday.conformal_gate import ACIGate, coverage_report
from src.intraday.costs import round_trip_cost_pct
from src.intraday.features import build_matrix, check_schema
from src.intraday.flow_model import FlowModel
from src.intraday.labeler import geometric_baseline, label_day, outcomes_frame
from src.intraday.price_model import PriceModel
from src.intraday.risk import is_blackout
from src.intraday.screener import (SCORE_COMPONENTS, fit_weights, screen_day,
                                   score_rows)
from src.intraday.bars import median_1min_volume

logger = logging.getLogger(__name__)


class LeakageAlarm(RuntimeError):
    pass


# ── dataset assembly ──────────────────────────────────────────────────

def build_dataset(days: list[date], symbols: list[str] | None = None) -> pd.DataFrame:
    """Screen → features → labels for a list of trading days. Returns one row
    per (symbol, decision bar) with feature columns, screen components, label.

    Blackout is checked PER screened symbol (M-6) — results-day names drop out,
    the rest of the day still trades."""
    rows = []
    skipped_days = 0
    for day in days:
        if is_blackout(day):                       # market-wide blackout → skip day
            continue
        syms = load_universe(as_of=day)["symbol"].tolist() if symbols is None else symbols
        try:
            top = screen_day(day, syms)
        except Exception as e:  # noqa: BLE001 — counted; raises over tolerance below
            skipped_days += 1
            logger.warning("dataset: screen %s failed (%s)", day, e)
            continue
        # per-symbol blackout (results ±1)
        top = top[~top["symbol"].apply(lambda s: is_blackout(day, s))]
        if top.empty:
            continue
        plan = top.assign(date=day)[["symbol", "date", "direction"]]
        feats = build_matrix(plan)
        if feats.empty:
            continue
        screen_cols = top.set_index("symbol")[SCORE_COMPONENTS + ["score"]]
        for sym, g in feats.groupby("symbol"):
            direction = int(plan[plan.symbol == sym]["direction"].iloc[0])
            outs = outcomes_frame(label_day(sym, day, {ts: direction for ts in g["ts"]}))
            if outs.empty:
                continue
            g = g.copy()
            g["entry_ts"] = g["ts"] + pd.Timedelta("15min")
            merged = g.merge(outs[["entry_ts", "label", "pnl_pct", "barrier", "entry"]],
                             on="entry_ts", how="inner")
            for col in SCORE_COMPONENTS:
                merged[f"screen_{col}"] = screen_cols.loc[sym, col]
            rows.append(merged)
    if skipped_days > len(days) * load_config()["data"]["max_skip_share"]:
        raise RuntimeError(
            f"{skipped_days}/{len(days)} days failed screening "
            f"(> {load_config()['data']['max_skip_share']:.0%}) — check bar store coverage"
        )
    if not rows:
        raise RuntimeError("dataset assembly produced 0 rows — check bar store coverage")
    out = pd.concat(rows, ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    logger.info("dataset: %d rows, %d days, base win rate %.3f",
                len(out), out.date.nunique(), out.label.mean())
    return out


# ── fold machinery ────────────────────────────────────────────────────

def make_folds(all_days: list[date]) -> list[tuple[list[date], list[date]]]:
    """Expanding-window day-level folds: train ≥ train_min_months, test =
    test_fold_months, embargo_days dropped from the train end. Raises if fewer
    than training.min_folds folds are available (≥6 OOS, PLAN_v3 §12.2)."""
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
    if len(folds) < tr_cfg["min_folds"]:
        raise RuntimeError(
            f"only {len(folds)} walk-forward folds < min_folds={tr_cfg['min_folds']} — "
            f"need more history (≥{tr_cfg['train_min_months']}m train + "
            f"{tr_cfg['min_folds']}×{test_len}m OOS)"
        )
    return folds


def _inner_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day-boundary inner split (M-3/H-4): head = train days except the last
    inner_calib_days calendar days (+1 embargo); tail = those calib days. Never
    a row-index cut — a trading day must not straddle the boundary."""
    cfg = load_config()["training"]
    days = sorted(pd.to_datetime(train_df["date"]).dt.date.unique())
    calib_days = days[-cfg["inner_calib_days"]:]
    embargo = timedelta(days=cfg["embargo_days"])
    calib_start = calib_days[0]
    head_days = [d for d in days if d < calib_start - embargo]
    head = train_df[pd.to_datetime(train_df["date"]).dt.date.isin(head_days)]
    tail = train_df[pd.to_datetime(train_df["date"]).dt.date.isin(calib_days)]
    return head, tail


def fit_fold(train_df: pd.DataFrame, tune: bool | None = None) -> tuple[PriceModel, FlowModel | None, Blender, dict]:
    """Train models for one fold (v3_supplementry §8.1):
      1. day-boundary head/calib-tail split (+embargo)
      2. (optional) PriceModel.tune(head) — params frozen now
      3. price/flow fit on head → predict tail (true OOF)
      4. Blender.fit_weight + fit_calibration on tail OOF
      5. price/flow refit on FULL train window with frozen params
    Step 5/4 pairing is legitimate OOF-stacking; the forbidden pattern
    (calibrator on in-fold predictions) is structurally excluded."""
    cfg = load_config()
    tune = cfg["training"]["tune"] if tune is None else tune
    flow_active = bool(cfg["gate"]["flow_model_active"])
    head, tail = _inner_split(train_df)
    if head.empty or tail.empty:
        raise RuntimeError("inner split produced an empty head or calibration tail")

    pm = PriceModel()
    if tune:
        pm.tune(head, head["label"])
    pm.fit(head, head["label"])
    fm = None
    p_f_tail = None
    if flow_active:
        fm = FlowModel().fit(head, head["label"])
        p_f_tail = fm.predict_proba(tail)

    p_p_tail = pm.predict_proba(tail)
    blender = Blender(flow_active=flow_active)
    blender.fit_weight(p_p_tail, p_f_tail, tail["label"].values)
    blender.fit_calibration(blender.blend(p_p_tail, p_f_tail), tail["label"].values)
    calib_brier = float(np.mean((blender.calibrated(p_p_tail, p_f_tail) - tail["label"].values) ** 2))

    pm = PriceModel(pm.params).fit(train_df, train_df["label"])   # full-train refit, frozen params
    if flow_active:
        fm = FlowModel(fm.params).fit(train_df, train_df["label"])

    # per-fold screen weights, fit on TRAIN days only
    weights = _fit_fold_screen_weights(train_df)
    record = {"weight": blender.weight, "calib_brier": calib_brier,
              "price_params": pm.params, "screen_weights": weights}
    return pm, fm, blender, record


def _fit_fold_screen_weights(train_df: pd.DataFrame) -> dict | None:
    """Refit screen weights on train days (H-8). 'day_won' = symbol produced ≥1
    win that day. Returns None if degenerate (single-class)."""
    cols = [f"screen_{c}" for c in SCORE_COMPONENTS]
    if not all(c in train_df.columns for c in cols):
        return None
    grp = train_df.groupby(["date", "symbol"])
    rows = grp[cols].first().rename(columns={f"screen_{c}": c for c in SCORE_COMPONENTS})
    won = (grp["label"].max() > 0).astype(int)
    try:
        return fit_weights(rows.reset_index(), won.reset_index()["label"])
    except Exception as e:  # noqa: BLE001 — degenerate fold falls back to default weights
        logger.warning("fold screen-weight fit skipped: %s", e)
        return None


# ── the walk-forward run ──────────────────────────────────────────────

def run_backtest(start: date, end: date, stress: bool = False,
                 capital_per_trade_inr: float = 100_000, tune: bool | None = None,
                 args_note: str = "") -> dict:
    cfg = load_config()
    all_days = pd.bdate_range(start, end).date.tolist()

    logger.info("assembling dataset %s → %s (%d candidate days)", start, end, len(all_days))
    data = build_dataset(all_days)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    folds = make_folds(sorted(data["date"].unique()))
    logger.info("walk-forward: %d folds, geometric baseline %.3f", len(folds), geometric_baseline())

    trades = []
    fold_stats = []
    vol_cache: dict[tuple[str, date], float] = {}
    for k, (train_days, test_days) in enumerate(folds, 1):
        train_df = data[data.date.isin(train_days)]
        check_schema(train_df)
        pm, fm, blender, record = fit_fold(train_df, tune=tune)
        gate = ACIGate()  # fresh τ per fold, adapts through the test window
        refit_weights = record["screen_weights"]

        for day in test_days:
            day_rows = data[data.date == day].copy()
            if day_rows.empty:
                continue
            if refit_weights is not None:  # re-rank test day with fold weights
                day_rows = _refit_rank(day_rows, refit_weights, cfg["screen"]["top_n"])
            p_p = pm.predict_proba(day_rows)
            p_f = fm.predict_proba(day_rows) if fm is not None else None
            p_cal = blender.calibrated(p_p, p_f)
            mask = gate.fire(p_cal, p_p, p_f)
            fired = day_rows[mask].copy()
            fired["p_cal"] = p_cal[mask]
            fired = fired.sort_values("entry_ts").groupby("symbol", as_index=False).first()
            day_labels = []
            for _, r in fired.iterrows():
                key = (r["symbol"], day)
                if key not in vol_cache:
                    try:
                        vol_cache[key] = median_1min_volume(r["symbol"], day)
                    except Exception:  # noqa: BLE001 — fall back to a conservative floor
                        vol_cache[key] = float(cfg["universe"]["min_median_1min_volume"])
                qty = max(1, int(capital_per_trade_inr / r["entry"]))
                cost = round_trip_cost_pct(r["entry"], qty, vol_cache[key], stress=stress)
                trades.append({
                    "fold": k, "date": day, "symbol": r["symbol"],
                    "sector": _sector(r["symbol"], day),
                    "entry_ts": r["entry_ts"], "p_cal": r["p_cal"],
                    "label": int(r["label"]), "barrier": r["barrier"],
                    "atr_pct": float(r["atr_pct"]),
                    "tod_bucket": _tod_bucket(r["entry_ts"]),
                    "pnl_pct_gross": r["pnl_pct"], "cost_pct": cost,
                    "pnl_pct_net": r["pnl_pct"] - cost,
                })
                day_labels.append(int(r["label"]))
            gate.update(day_labels)

        fold_trades = [t for t in trades if t["fold"] == k]
        if fold_trades:
            ft = pd.DataFrame(fold_trades)
            fold_cov = len(ft) / max(1, len(data[data.date.isin(test_days)]))
            fs = {
                "fold": k, "n_trades": len(ft),
                "win_rate": float(ft.label.mean()),
                "expectancy_pct": float(ft.pnl_pct_net.mean()),
                "signals_per_day": len(ft) / max(1, len(test_days)),
                "coverage": fold_cov,
                "calib_brier": record["calib_brier"],
            }
            # per-fold leakage alarm (PLAN_v3 §12.2 — "any fold")
            if fs["win_rate"] > cfg["gates"]["leakage_alarm_win_rate"] and \
               fold_cov > cfg["gates"]["leakage_alarm_coverage"]:
                raise LeakageAlarm(
                    f"fold {k}: win rate {fs['win_rate']:.3f} at coverage {fold_cov:.3f} "
                    f"— investigate lookahead before trusting this run"
                )
            fold_stats.append(fs)
            logger.info("fold %d: %s", k, fs)

    # Union of EVERY test-fold day (incl. days that fired zero signals) — the
    # honest coverage denominator (M-4 / PLAN_v3 §12.2). Days with no trades must
    # still count as "candidate rows we could have traded but didn't."
    all_test_days = sorted({d for _, test_days in folds for d in test_days})
    return _finalize(trades, fold_stats, data, stress, start, end, args_note, all_test_days)


def _refit_rank(day_rows: pd.DataFrame, weights: dict, top_n: int) -> pd.DataFrame:
    """Re-rank a test day's candidates with fold-fitted screen weights and keep
    top_n. Built from the screen_<component> columns into a clean frame so the
    component names don't collide with same-named flow features (e.g.
    delivery_z is both a screen component and a flow feature)."""
    per_symbol = day_rows.drop_duplicates("symbol")
    comp = pd.DataFrame({c: per_symbol[f"screen_{c}"].to_numpy() for c in SCORE_COMPONENTS})
    comp["symbol"] = per_symbol["symbol"].to_numpy()
    comp["refit_score"] = score_rows(comp, weights)
    keep = set(comp.sort_values("refit_score", ascending=False).head(top_n)["symbol"])
    return day_rows[day_rows["symbol"].isin(keep)]


def _sector(symbol: str, day: date) -> str:
    try:
        uni = load_universe(as_of=day)
        row = uni[uni.symbol == symbol]
        return str(row["sector"].iloc[0]) if not row.empty else "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _tod_bucket(entry_ts) -> str:
    t = pd.Timestamp(entry_ts).time()
    from datetime import time as _t
    if t < _t(10, 0):
        return "0930-1000"
    if t < _t(10, 30):
        return "1000-1030"
    return "1030-1100"


def _finalize(trades, fold_stats, data, stress, start, end, args_note, all_test_days=None) -> dict:
    cfg = load_config()
    g = cfg["gates"]
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        raise RuntimeError("backtest emitted 0 trades — gate too tight or edge absent (Gate 3 fails honestly)")

    # denominator = ALL candidate rows across ALL test windows (not just days
    # that produced a trade) — otherwise zero-signal days vanish and coverage is
    # overstated. Fall back to traded days only if the caller didn't pass folds.
    test_days = set(all_test_days) if all_test_days is not None else set(tdf.date.unique())
    test_rows = len(data[data.date.isin(test_days)])
    coverage = len(tdf) / max(1, test_rows)
    win_rate = float(tdf.label.mean())
    expectancy = float(tdf.pnl_pct_net.mean())
    daily = tdf.groupby("date")["pnl_pct_net"].sum()
    equity = (1 + daily).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    brier = float(np.mean((tdf["p_cal"] - tdf["label"]) ** 2))

    fairness = _fairness_tables(tdf)
    cov_report = coverage_report(tdf)
    calib_gap = _calibration_gap(tdf)
    excluded = [s for s, r in cov_report.items() if r["below_target"]]
    subsidy = _subsidy_breaches(fairness)

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "window": {"start": str(start), "end": str(end)},
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
        "sharpe_daily": sharpe,
        "brier_score": brier,
        "calibration_gap": calib_gap,
        "fold_stats": fold_stats,
        "fairness": fairness,
        "per_symbol_coverage": cov_report,
        "excluded_symbols": excluded,
        "gates": {
            "win_rate_ok": win_rate >= g["min_win_rate"],
            "expectancy_ok": expectancy >= g["min_expectancy_pct"],
            "signals_ok": float(tdf.groupby("date").size().mean()) >= g["min_signals_per_day"],
            "calibration_ok": calib_gap <= g["calibration_max_gap"],
            "no_subsidy_ok": not subsidy,
            "per_symbol_coverage_ok": len(excluded) == 0,
        },
    }
    report["gates"]["all_pass"] = all(report["gates"].values())

    # ── leakage alarm (aggregate; per-fold checked in the loop) ───────
    if win_rate > g["leakage_alarm_win_rate"] and coverage > g["leakage_alarm_coverage"]:
        raise LeakageAlarm(
            f"win rate {win_rate:.3f} at coverage {coverage:.3f} — investigate lookahead before trusting this run"
        )

    out_dir = ROOT / cfg["paths"]["reports"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"backtest_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    tdf.to_parquet(out_dir / f"trades_{stamp}.parquet", index=False)
    _write_artifacts(tdf, test_rows, out_dir, stamp)
    _excluded_symbols_file(excluded)
    _append_run_registry(report, stress, args_note, stamp)
    logger.info("backtest report → %s (gates: %s)", out_dir / f"backtest_{stamp}.json", report["gates"])
    return report


def _fairness_tables(tdf: pd.DataFrame) -> dict:
    """Per symbol/sector/vol-regime/time-of-day: n, win rate, expectancy, mean p_cal (H-5)."""
    tdf = tdf.copy()
    # qcut into 3 vol terciles, but a fold with few trades / near-constant ATR may
    # not yield 3 distinct edges — with explicit labels that raises ValueError and
    # would abort the whole report. Fall back to a single regime in that case.
    try:
        tdf["vol_regime"] = pd.qcut(tdf["atr_pct"], 3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        tdf["vol_regime"] = "all"
    out = {}
    for key in ["symbol", "sector", "vol_regime", "tod_bucket"]:
        grp = tdf.groupby(key, observed=True)
        out[key] = {
            str(k): {"n": int(v), "win_rate": float(wr), "expectancy": float(ex),
                     "mean_p_cal": float(mp)}
            for k, v, wr, ex, mp in zip(
                grp.size().index, grp.size(),
                grp["label"].mean(), grp["pnl_pct_net"].mean(), grp["p_cal"].mean())
        }
    return out


def _subsidy_breaches(fairness: dict) -> list[str]:
    """Any symbol/sector with ≥30 trades and negative expectancy ⇒ subsidy alarm."""
    breaches = []
    for key in ["symbol", "sector"]:
        for name, row in fairness.get(key, {}).items():
            if row["n"] >= 30 and row["expectancy"] < 0:
                breaches.append(f"{key}:{name}")
    return breaches


def _calibration_gap(tdf: pd.DataFrame) -> float:
    """|expected − realized| win rate in the fired region (reliability gap)."""
    if tdf.empty:
        return 1.0
    return float(abs(tdf["p_cal"].mean() - tdf["label"].mean()))


def _excluded_symbols_file(excluded: list[str]) -> None:
    p = ROOT / load_config()["paths"]["state"] / "excluded_symbols.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(excluded, indent=2))


def _append_run_registry(report: dict, stress: bool, args_note: str, stamp: str) -> None:
    """One append-only CSV line per run (M-5) — tune-until-green becomes visible."""
    import subprocess

    reg = ROOT / load_config()["paths"]["run_registry"]
    reg.parent.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    cfg_hash = _config_hash()
    line = {
        "ts": report["generated"], "stamp": stamp, "git_sha": sha, "config_hash": cfg_hash,
        "stress": stress, "args": args_note,
        "win_rate": report["win_rate"], "expectancy": report["expectancy_pct_net"],
        "coverage": report["coverage_of_candidates"], "all_pass": report["gates"]["all_pass"],
    }
    header = not reg.exists()
    pd.DataFrame([line]).to_csv(reg, mode="a", header=header, index=False)


def _config_hash() -> str:
    import hashlib

    return hashlib.sha256(json.dumps(load_config(), sort_keys=True, default=str).encode()).hexdigest()[:12]


def _write_artifacts(tdf: pd.DataFrame, test_rows: int, out_dir: Path, stamp: str) -> None:
    """Win-rate-vs-coverage frontier + calibration reliability diagram (PNGs).
    Coverage axis uses the same denominator as the headline metric: candidate
    rows on test days only (not train+test)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 — artifacts are nice-to-have, not gating
        logger.warning("matplotlib unavailable, skipping artifact PNGs: %s", e)
        return

    # frontier: sweep p_cal threshold, plot (coverage, win rate)
    thresholds = np.linspace(tdf["p_cal"].min(), tdf["p_cal"].max(), 25)
    total = max(1, test_rows)
    cov, wr = [], []
    for t in thresholds:
        sub = tdf[tdf["p_cal"] >= t]
        if len(sub) >= 10:
            cov.append(len(sub) / total)
            wr.append(sub["label"].mean())
    fig, ax = plt.subplots()
    ax.plot(cov, wr, marker="o")
    ax.set_xlabel("coverage"); ax.set_ylabel("win rate"); ax.set_title("Win-rate vs coverage frontier")
    ax.axhline(load_config()["gates"]["min_win_rate"], ls="--", color="r")
    fig.savefig(out_dir / f"frontier_{stamp}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # reliability diagram
    fig, ax = plt.subplots()
    bins = np.linspace(tdf["p_cal"].min(), tdf["p_cal"].max(), 8)
    tdf2 = tdf.assign(b=pd.cut(tdf["p_cal"], bins))
    rel = tdf2.groupby("b", observed=True).agg(pred=("p_cal", "mean"), obs=("label", "mean"))
    ax.plot(rel["pred"], rel["obs"], marker="o", label="model")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="ideal")
    ax.set_xlabel("predicted P(win)"); ax.set_ylabel("realized win rate")
    ax.set_title("Calibration reliability"); ax.legend()
    fig.savefig(out_dir / f"reliability_{stamp}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
