"""main.py — v3 CLI for the Intraday Selective Signal System (PLAN_v3.md).

Reads config_v3.yaml only. v1 modes are gone (decommissioned with v1).

Modes:
  backfill | bhavcopy | corp-actions | record | screen | geometry-study
  | backtest [--stress] | train-v3 | promote --version N --confirm | rollback
  | recalibrate | paper --capital N | drift | gates | serve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _today() -> date:
    from src.intraday import today_ist
    return today_ist()


# ── data ──────────────────────────────────────────────────────────────

def cmd_backfill(args):
    from src.intraday.data_feed import backfill
    backfill(symbols=args.symbols, years=args.years)


def cmd_bhavcopy(args):
    from src.intraday import load_config
    from src.intraday.bhavcopy import backfill_bhavcopies, fetch_fii_dii
    years = args.years or load_config()["data"]["backfill_years"]
    start = _today() - timedelta(days=int(years * 365.25))
    backfill_bhavcopies(start, _today())
    fetch_fii_dii()


def cmd_corp_actions(args):
    from src.intraday import load_config
    from src.intraday.corporate_actions import build_table
    years = args.years or load_config()["data"]["backfill_years"]
    build_table(_today() - timedelta(days=int(years * 365.25)), _today())


def cmd_record(args):
    from src.intraday.recorder import run
    run()


# ── pipeline ──────────────────────────────────────────────────────────

def cmd_screen(args):
    from src.intraday.screener import screen_day
    day = date.fromisoformat(args.date) if args.date else _today()
    print(screen_day(day).to_string(index=False))


def cmd_geometry_study(args):
    import pandas as pd
    from src.intraday.backtester import build_dataset, make_folds, trading_sessions
    from src.intraday.geometry_study import TrainWindow, choose_geometry, run_grid
    end = date.fromisoformat(args.end) if args.end else _today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=4 * 365)
    data = build_dataset(trading_sessions(start, end))
    folds = make_folds(sorted(set(pd.to_datetime(data["date"]).dt.date)))
    grid = run_grid(TrainWindow(folds[0][0]))
    print(grid.to_string(index=False))
    print("\nChosen geometry:", choose_geometry(grid))


def cmd_backtest(args):
    from src.intraday.backtester import run_backtest
    end = date.fromisoformat(args.end) if args.end else _today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=3 * 365)
    report = run_backtest(start, end, stress=args.stress, tune=args.tune,
                          symbols=args.symbols,   # PLAN_v4 §6.3 Run D: restrict universe
                          args_note=f"cli start={start} end={end} stress={args.stress} "
                                    f"symbols={args.symbols}")
    print(json.dumps(report["gates"], indent=2, default=str))


# ── PLAN_v4: pre-open regime + news channels ──────────────────────────

def cmd_regime_capture(args):
    from src.intraday.regime_data import capture_today
    day = date.fromisoformat(args.date) if args.date else _today()
    n = capture_today(day)
    print(json.dumps({"mode": "regime-capture", "day": str(day), "rows": n}))


def cmd_regime_backfill(args):
    from src.intraday import load_config
    from src.intraday.regime_data import backfill
    end = date.fromisoformat(args.end) if args.end else _today()
    years = args.years or load_config()["data"]["backfill_years"]
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=int(years * 365.25))
    n = backfill(start, end)
    print(json.dumps({"mode": "regime-backfill", "start": str(start), "end": str(end), "rows": n}))


def cmd_news_capture(args):
    from src.intraday.news_regime import capture
    day = date.fromisoformat(args.date) if args.date else _today()
    n = capture(day)
    print(json.dumps({"mode": "news-capture", "day": str(day), "headlines": n}))


def cmd_train_v3(args):
    from src.intraday.trainer import train_production
    end = date.fromisoformat(args.end) if args.end else _today()
    result = train_production(end_day=end, train_months=args.months, tune=args.tune,
                              push_dvc=not args.no_dvc, track=not args.no_track)
    print(json.dumps(result.__dict__, indent=2, default=str))


def cmd_promote(args):
    from src.intraday.trainer import promote
    print(json.dumps(promote(args.version, confirm=args.confirm), indent=2))


def cmd_rollback(args):
    from src.intraday.trainer import rollback
    print(json.dumps(rollback(confirm=args.confirm), indent=2))


def cmd_recalibrate(args):
    from src.intraday.trainer import recalibrate
    print(json.dumps(recalibrate(), indent=2, default=str))


def cmd_paper(args):
    from src.intraday.paper_runner import run
    run(capital_inr=args.capital)


def cmd_drift(args):
    from src.intraday.drift import run_check
    print(json.dumps(run_check(), indent=2, default=str))


def cmd_gates(args):
    from src.intraday import gates
    end = date.fromisoformat(args.end) if args.end else None
    start = date.fromisoformat(args.start) if args.start else None
    print(json.dumps(gates.run_all(start, end), indent=2, default=str))


def cmd_serve(args):
    import uvicorn
    from src.intraday import load_config
    cfg = load_config()["api"]
    uvicorn.run("src.api.app:app", host=cfg["host"], port=cfg["port"])


# ── daily prediction & listing system (src/daily) ─────────────────────

def _daily_horizon(args) -> str:
    if args.horizon:
        return args.horizon
    from src.daily import enabled_horizons
    hs = enabled_horizons()
    if not hs:
        raise SystemExit("no enabled horizons in config_daily.yaml")
    return hs[0]


def cmd_daily_universe(args):
    import subprocess
    import sys as _sys
    cmd = [_sys.executable, "scripts/build_daily_universe.py"]
    if args.no_rebuild_panel:
        cmd.append("--no-rebuild-panel")
    raise SystemExit(subprocess.call(cmd))


def cmd_daily_panel(args):
    from src.daily import load_daily_config
    from src.daily.panel import build_panel
    years = args.years or load_daily_config()["data"]["backfill_years"]
    start = _today() - timedelta(days=int(years * 365.25))
    print(json.dumps({"symbols_written": len(build_panel(start, _today(), symbols=args.symbols))}))


def cmd_daily_global(args):
    from src.daily.global_data import fetch_all
    print(json.dumps(fetch_all(), indent=2, default=str))


def cmd_daily_macro(args):
    from src.daily.macro import build_calendar
    print(json.dumps(build_calendar(), indent=2, default=str))


def cmd_daily_fii_backfill(args):
    from src.daily.fii_backfill import backfill
    print(json.dumps(backfill(), indent=2, default=str))


def cmd_daily_backtest(args):
    from src.daily.backtester import run_backtest
    horizon = _daily_horizon(args)
    end = date.fromisoformat(args.end) if args.end else _today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=3 * 365)
    rep = run_backtest(start, end, horizon=horizon,
                       args_note=f"cli {horizon} {start}..{end}")
    print(json.dumps({"verdict": rep["verdict"],
                      "milestone1": rep["milestone1_predictive"]["criteria"],
                      "milestone2": rep["milestone2_after_cost"]["criteria"]},
                     indent=2, default=str))


def cmd_daily_train(args):
    from src.daily.trainer import train_production
    horizon = _daily_horizon(args)
    end = date.fromisoformat(args.end) if args.end else _today()
    print(json.dumps(train_production(horizon=horizon, end_day=end, tune=args.tune).__dict__,
                     indent=2, default=str))


def cmd_daily_list(args):
    from src.daily.run_list import run
    day = date.fromisoformat(args.date) if args.date else _today()
    horizons = [args.horizon] if args.horizon else None
    out = run(day=day, horizons=horizons)
    for h, block in out["horizons"].items():
        print(f"\n=== {h} (actionable={block['actionable']}) ===")
        for pick in block["picks"]:
            tgt = pick.get("target_price"); stp = pick.get("stop_price")
            ref = pick.get("ref_close"); qty = pick.get("qty_ref")
            print(f"  {pick['symbol']:12s} P(win)={pick['prob']:.3f}"
                  f"  ref={ref}  target={tgt}  stop={stp}  qty@1L={qty}"
                  f"  hold<={pick.get('max_hold_days')}d  {', '.join(pick['why'])}")


def cmd_daily_eval(args):
    from src.daily.evaluate import evaluate_day
    day = date.fromisoformat(args.date) if args.date else _today() - timedelta(days=1)
    print(json.dumps(evaluate_day(day, horizon=_daily_horizon(args)), indent=2, default=str))


def cmd_daily_scoreboard(args):
    from src.daily.evaluate import scoreboard, scoreboard_from_oos
    horizon = _daily_horizon(args)
    out = scoreboard(horizon) if args.start == "live" else scoreboard_from_oos(horizon)
    print(json.dumps(out, indent=2, default=str))


def cmd_daily_serve(args):
    import uvicorn
    from src.daily import load_daily_config
    cfg = load_daily_config()["api"]
    uvicorn.run("src.api.daily_app:app", host=cfg["host"], port=cfg["port"])


def main():
    p = argparse.ArgumentParser(description="Intraday Selective Signal System (v3)")
    p.add_argument("--mode", required=True, choices=[
        "backfill", "bhavcopy", "corp-actions", "record", "screen", "geometry-study",
        "backtest", "train-v3", "promote", "rollback", "recalibrate", "paper",
        "drift", "gates", "serve",
        "regime-capture", "regime-backfill", "news-capture",   # PLAN_v4 §5/§8
        "daily-universe", "daily-panel", "daily-global", "daily-macro",   # src/daily
        "daily-fii-backfill", "daily-backtest", "daily-train", "daily-list",
        "daily-eval", "daily-scoreboard", "daily-serve",
    ])
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--stress", action="store_true", default=False)
    p.add_argument("--tune", action="store_true", default=None,
                   help="force Optuna tuning on (overrides config training.tune)")
    p.add_argument("--no-dvc", action="store_true", default=False)
    p.add_argument("--no-track", action="store_true", default=False)
    p.add_argument("--version", type=str, default=None)
    p.add_argument("--confirm", action="store_true", default=False)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--horizon", type=str, default=None,
                   help="daily-* modes: horizon block from config_daily.yaml "
                        "(default: first enabled)")
    p.add_argument("--no-rebuild-panel", action="store_true", default=False,
                   help="daily-universe: reuse the cached v6 panel")
    args = p.parse_args()

    dispatch = {
        "backfill": cmd_backfill, "bhavcopy": cmd_bhavcopy, "corp-actions": cmd_corp_actions,
        "record": cmd_record, "screen": cmd_screen, "geometry-study": cmd_geometry_study,
        "backtest": cmd_backtest, "train-v3": cmd_train_v3, "promote": cmd_promote,
        "rollback": cmd_rollback, "recalibrate": cmd_recalibrate, "paper": cmd_paper,
        "drift": cmd_drift, "gates": cmd_gates, "serve": cmd_serve,
        "regime-capture": cmd_regime_capture, "regime-backfill": cmd_regime_backfill,
        "news-capture": cmd_news_capture,
        "daily-universe": cmd_daily_universe, "daily-panel": cmd_daily_panel,
        "daily-global": cmd_daily_global, "daily-macro": cmd_daily_macro,
        "daily-fii-backfill": cmd_daily_fii_backfill, "daily-backtest": cmd_daily_backtest,
        "daily-train": cmd_daily_train, "daily-list": cmd_daily_list,
        "daily-eval": cmd_daily_eval, "daily-scoreboard": cmd_daily_scoreboard,
        "daily-serve": cmd_daily_serve,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
