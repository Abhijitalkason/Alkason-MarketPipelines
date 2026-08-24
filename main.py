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


def _pick_verdict(prob: float, actionable: bool, strong_min: float) -> tuple[str, str, str]:
    """Final-status verdict for one pick → (action, colour, label). This is a
    LONG-ONLY BUY watchlist, so the call is BUY (act) vs HOLD (watch) — SELL is
    never emitted here (that is the per-stock Analyzer's job). Colours are the
    real-money confidence signal:
      green  = high confidence — a strong signal, calibrated P(win) ≥ strong_min;
      yellow = moderate — the list is actionable (after-cost backtest gate passed)
               but the pick is below the strong bar;
      red    = low / no proven edge — not actionable or P near the ~0.50 base rate,
               i.e. watch only, do NOT size up.
    """
    if prob >= strong_min:
        return "BUY", "green", f"high confidence — strong signal (P≥{strong_min:.2f})"
    if actionable:
        return "BUY", "yellow", "moderate — actionable (after-cost gate passed), below strong bar"
    return "HOLD", "red", "low / no proven edge — watch only, do not size up"


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

    # tier 1 — strong signals (rare by design): speak first and loudest
    for s in out.get("strong_signals", []):
        print(f"\n{'!' * 66}\n!! STRONG SIGNAL  {s['symbol']}  ({s['horizon']})  "
              f"P(win)={s['prob']:.3f}\n!! {s.get('confidence_note', 'clears the 0.80 selective bar')}"
              f"\n{'!' * 66}")

    # tier 2 — the day's single top pick (always present)
    tp = out.get("top_pick")
    if tp:
        print(f"\n{'=' * 66}\n TODAY'S TOP PICK ({tp['horizon']}, tier={tp['tier']})\n"
              f"   {tp['symbol']}  {tp['direction']}  P(win)={tp['prob']:.3f}\n"
              f"   entry: {tp.get('entry_rule')}  ref={tp.get('ref_close')}  "
              f"target={tp.get('target_price')}  stop={tp.get('stop_price')}  "
              f"qty@1L={tp.get('qty_ref')}  hold<={tp.get('max_hold_days')}d\n"
              f"   why: {', '.join(tp['why'])}\n"
              f"   note: {tp['confidence_note']}\n{'=' * 66}")

    for h, block in out["horizons"].items():
        print(f"\n=== {h} (actionable={block['actionable']}) ===")
        for pick in block["picks"]:
            tgt = pick.get("target_price"); stp = pick.get("stop_price")
            ref = pick.get("ref_close"); qty = pick.get("qty_ref")
            print(f"  #{pick['rank']:<2d} {pick['symbol']:12s} P(win)={pick['prob']:.3f}"
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


def cmd_daily_pipeline(args):
    """Observable end-to-end recommendation job (feature 2026-07-17):
    fetch EOD → fetch global → rebuild panel → generate picks → notify.
    Every step's state + live one-liners stream to reports/daily/
    pipeline_status.json (GET /pipeline/status; Pipeline panel in the UI).
    If today's data isn't published yet, falls back to the latest session
    on disk instead of emitting an empty list."""
    from src.daily import daily_path
    from src.daily.status import PipelineStatus
    from src.daily.trace import Trace

    from src.daily import load_daily_config as _ldc
    live_on = bool(getattr(args, "live", False)) or _ldc().get("live_data", {}).get("enabled", False)
    day = date.fromisoformat(args.date) if args.date else _today()
    _steps = [
        ("fetch_bhavcopy", "Fetch NSE EOD (bhavcopy)"),
        ("fetch_global", "Fetch global/macro series (yfinance)"),
        ("rebuild_panel", "Rebuild per-symbol daily panel"),
    ]
    if live_on:
        _steps.append(("live_snapshot", "Live Upstox snapshot (provisional today bar)"))
    _steps += [("generate_list", "Score universe → picks + top pick"),
               ("notify", "Desktop notification")]
    ps = PipelineStatus("daily-pipeline", _steps)
    tr = Trace()   # per-item journal → GET /pipeline/trace · /ui/trace
    try:
        fetched_today = False
        with ps.step("fetch_bhavcopy"):
            # trailing-week window: self-heals gaps after idle days — files
            # land even when the sweep then raises about today's missing file
            fetch_from = day - timedelta(days=7)
            try:
                from src.intraday.bhavcopy import backfill_bhavcopies
                backfill_bhavcopies(fetch_from, day)
                fetched_today = True
                ps.update(f"bhavcopy {fetch_from}..{day} on disk")
                tr.event("fetch_bhavcopy", f"{fetch_from}..{day}", "ok",
                         source="nsearchives.nseindia.com", detail="eq + fo parquet written")
            except Exception as e:  # noqa: BLE001 — today unpublished/holiday; earlier days still landed
                ps.update(f"partial — {str(e)[:140]}")
                tr.event("fetch_bhavcopy", f"{fetch_from}..{day}", "skip",
                         source="nsearchives.nseindia.com",
                         detail="fetched what exists in the window; missing day(s) noted in error",
                         error=str(e)[:400])

        with ps.step("fetch_global"):
            try:
                from src.daily.global_data import SINGLE_SERIES, fetch_all
                written = fetch_all()
                for name, rows in sorted(written.items()):
                    ticker = SINGLE_SERIES[name][0] if name in SINGLE_SERIES else "composite"
                    tr.event("fetch_global", name, "ok" if rows else "skip",
                             source=f"Yahoo Finance ({ticker})", rows=rows)
                ps.update(f"{len(written)} global series refreshed")
            except Exception as e:  # noqa: BLE001 — optional channel, fail-soft
                ps.update(f"skipped — {type(e).__name__} (features degrade to _present=0)")
                tr.event("fetch_global", "all", "fail", error=str(e)[:400])

        with ps.step("rebuild_panel"):
            from src.daily import load_daily_config
            from src.daily.panel import build_panel
            years = load_daily_config()["data"]["backfill_years"]
            written = build_panel(day - timedelta(days=int(years * 365.25)), day)
            for sym, rows in sorted(written.items()):
                tr.event("panel", sym, "ok" if rows else "skip",
                         rows=rows, source="local bhavcopy store")
            ps.update(f"{len(written)} symbols written")

        live_injected = False
        if live_on:
            with ps.step("live_snapshot"):
                from src.daily.live_snapshot import inject_live_rows, token_ok
                if not token_ok():
                    ps.update("Upstox token missing/expired — cannot fetch live data; "
                              "falling back to the latest EOD session")
                    tr.event("live", "token", "fail", error="UPSTOX_ACCESS_TOKEN invalid/expired")
                else:
                    res = inject_live_rows(day, trace=tr.event)
                    live_injected = res["injected"] > 0
                    ps.update(f"provisional live bars for {res['injected']}/"
                              f"{res['injected'] + res['skipped']} symbols "
                              "(PREVIEW — close not final, flow features absent)")

        with ps.step("generate_list"):
            import pandas as pd
            from src.daily.run_list import run as run_list
            # fallback: last session actually on disk (data lags the calendar).
            # With a live snapshot injected, today's provisional row IS present,
            # so eff advances to `day` and the pipeline scores the live bar.
            eff = day
            probe = [daily_path("daily_path") / f"{s}.parquet"
                     for s in ("RELIANCE", "TCS", "HDFCBANK")]
            dates = [pd.read_parquet(p, columns=["date"])["date"].max()
                     for p in probe if p.exists()]
            if dates:
                eff = min(day, max(d.date() for d in dates))
            if eff != day:
                ps.update(f"no data for {day} yet → using last session {eff}")
                tr.event("generate_list", str(day), "skip",
                         reason=f"session data not published — analysing {eff} instead")
            out = run_list(day=eff, trace=tr.event)
            tp = out.get("top_pick")
            strong = out.get("strong_signals", [])
            ps.update(f"day {eff}: top pick "
                      f"{tp['symbol'] + ' P=' + format(tp['prob'], '.3f') if tp else 'NONE'}"
                      f" · {len(strong)} strong signal(s)")

        with ps.step("notify"):
            import subprocess
            try:
                for s in strong:
                    subprocess.run(["osascript", "-e",
                                    f'display notification "{s["symbol"]} {s["direction"]} '
                                    f'P={s["prob"]:.3f} — verify before acting" '
                                    f'with title "🚨 STRONG SIGNAL (>=0.80)" sound name "Sosumi"'],
                                   check=False, timeout=10)
                if tp:
                    subprocess.run(["osascript", "-e",
                                    f'display notification "{tp["symbol"]} {tp["direction"]} '
                                    f'P={tp["prob"]:.3f} | tgt {tp.get("target_price")} '
                                    f'stop {tp.get("stop_price")}" '
                                    f'with title "Today\'s top pick ({tp["tier"]})"'],
                                   check=False, timeout=10)
                ps.update("notification sent" if tp else "nothing to notify")
            except Exception as e:  # noqa: BLE001
                ps.update(f"notify skipped — {type(e).__name__}")

        n_picks = sum(len(b["picks"]) for b in out["horizons"].values())
        # ── plain-English summary: what happened / what it means / what's next ──
        if live_injected:
            data_line = (f"⚠️ PROVISIONAL LIVE run: used an Upstox intraday snapshot for "
                         f"{eff} (prices are current, NOT the final close; delivery %/OI "
                         "features are absent). Treat as a preview, not a final signal.")
        elif fetched_today:
            data_line = f"Fetched today's ({day}) NSE end-of-day data."
        else:
            data_line = (f"Today's ({day}) NSE data is not published yet — analysed the last "
                         f"completed session ({eff}) instead.")
        what_happened = [
            data_line,
            "Refreshed global/macro series (S&P, Nasdaq, crude, USD/INR, India VIX…).",
            "Rebuilt the daily price panel for the top-100 universe.",
            f"Scored all universe stocks with the swing model → ranked top {n_picks} picks.",
        ]
        if tp:
            meaning = [
                f"TOP PICK = {tp['symbol']} ({tp['direction']}) — rank #1 of the universe: "
                f"the day's RELATIVE best, picked by model score + momentum tie-break.",
                f"Its calibrated P(win) is {tp['prob']:.3f} vs the 0.80 strong-signal bar → "
                f"NOT a high-confidence signal; the model's own gate marks it "
                f"actionable=false.",
                f"Main drivers: {', '.join(tp['why'])}.",
                f"If traded anyway: entry next session open ~₹{tp.get('ref_close')}, "
                f"target ₹{tp.get('target_price')}, stop ₹{tp.get('stop_price')}, "
                f"qty@₹1L {tp.get('qty_ref')}, hold ≤{tp.get('max_hold_days')} days.",
                f"Strong signals fired: {len(strong)} (a macOS notification shouts if one ever does).",
            ]
        else:
            meaning = ["No picks were produced — see the step one-liners above for why."]
        next_steps = [
            (f"NSE publishes today's data ~18:30 IST — press ▶ Start (or let the 19:30 cron "
             f"run) after that to get the pick for the NEXT session's open."
             if not fetched_today else
             f"Data is current through {eff}; the next pick lands after the next market close."),
            f"To dig deeper: type {tp['symbol'] if tp else 'a symbol'} into the Analyzer (left) "
            "for full evidence, news and the 1–5 day trade plan.",
            "Track the pick's honesty over time: python main.py --mode daily-scoreboard.",
        ]
        # ── enrich the final status so the UI shows the full, real-money-safe
        # recommendation card (action · confidence colour · buy/target/stop ·
        # duration · data provenance) instead of a bare one-liner ──────────────
        from src.daily import now_ist
        strong_min = float(_ldc()["listing"].get("top_pick", {}).get("strong_signal_min_prob", 0.80))

        def _enrich(p: dict) -> dict:
            action, color, clabel = _pick_verdict(p["prob"], bool(p.get("actionable")), strong_min)
            return {"rank": p.get("rank"), "symbol": p["symbol"],
                    "direction": p.get("direction", "LONG"), "prob": p["prob"],
                    "action": action, "color": color, "confidence_label": clabel,
                    "confidence_pct": round(p["prob"] * 100, 1),
                    "buy_price": p.get("ref_close"), "stop_price": p.get("stop_price"),
                    "target_price": p.get("target_price"),
                    "duration_days": p.get("max_hold_days"), "qty_at_1L": p.get("qty_ref")}

        enriched_picks = sorted(
            (_enrich(p) for b in out["horizons"].values() for p in b["picks"]),
            key=lambda x: (x["rank"] is None, x["rank"]))
        enriched_tp = ({**_enrich(tp), "tier": tp["tier"], "why": tp.get("why", [])}
                       if tp else None)
        provenance = {
            "session_date": str(eff),
            "used_upstox_live": live_injected,
            "captured_at": now_ist().isoformat(timespec="seconds"),
            "source": ("Upstox live intraday snapshot — PROVISIONAL (current price, NOT the "
                       "final close; delivery %/OI absent)" if live_injected else
                       "NSE EOD official close — FINAL (today's session)" if fetched_today else
                       "NSE EOD official close — last available session (today not published yet)"),
        }
        result = {"day": str(eff),
                  "data_provenance": provenance,
                  "top_pick": enriched_tp,
                  "picks": enriched_picks,
                  "n_picks": n_picks, "n_strong_signals": len(strong),
                  "summary": {
                      "headline": (f"Session {eff}: top pick {tp['symbol']} ({tp['direction']}), "
                                   f"{len(strong)} strong signal(s)" if tp else
                                   f"Session {eff}: no picks produced"),
                      "what_happened": what_happened,
                      "result_meaning": meaning,
                      "next_steps": next_steps,
                  }}
        ps.done(result=result)
        tr.close()
        print(json.dumps({"pipeline": "completed", **result}, indent=2))
    except Exception as e:
        # step ctx already recorded the failure; make the CLI loud too
        tr.event("pipeline", "run", "fail", error=f"{type(e).__name__}: {e}"[:400])
        tr.close()
        print(json.dumps({"pipeline": "failed", "error": f"{type(e).__name__}: {e}"}))
        raise


def cmd_analyze(args):
    """Single-stock deep analysis (feature request 2026-07-16): full background
    check + transparent BUY/HOLD/SELL scorecard. `--symbols X` picks the stock."""
    from src.analysis.engine import analyze
    if not args.symbols:
        raise SystemExit("usage: --mode analyze --symbols RELIANCE")
    rep = analyze(args.symbols[0])
    v = rep["verdict"]
    print(f"\n{'=' * 70}\n {rep['identity']['name']} ({rep['symbol']})  →  "
          f"{v['action']}  (score {v['composite_score']:+.3f}, "
          f"confidence {v['confidence']}, evidence {v['data_completeness']:.0%})\n{'=' * 70}")
    tp = rep.get("trade_plan")
    if tp:
        print(f"\nTRADE PLAN ({tp['direction']})")
        print(f"  entry: {tp['entry_rule']}")
        print(f"  ref {tp['ref_close']}  stop {tp['stop_price']}  "
              f"T1 {tp['target_1']} (1:1)  T2 {tp['target_2']} (2:1)  [R={tp['risk_per_share']}]")
        print(f"  hold: {tp['suggested_holding']}")
        print(f"  qty@1L {tp['qty_ref']} | qty@1%-risk {tp['qty_if_risking_1pct']} | "
              f"cost ~{tp['round_trip_cost_pct']}% round trip")
    print("\nREASONING")
    for i, r in enumerate(rep["reasoning"], 1):
        print(f"  {i}. {r}")
    print("\nSCORECARD")
    for r in rep["scorecard"]:
        flag = "" if r["present"] else "  [absent]"
        print(f"  {r['component']:<15s} w={r['weight']:.2f} s={r['score']:+.2f}{flag}  {r['reason']}")
    if rep["news"].get("present"):
        print(f"\nNEWS ({rep['news']['n_articles']} articles, engine {rep['news']['engine']})")
        for a in rep["news"]["articles"][:6]:
            print(f"  [{a['sentiment']:+.2f}] {a['title'][:90]}\n          {a['url']}")
    print("\nREFERENCES")
    for r in rep["references"][:6]:
        print(f"  {r['label']}: {r['url']}")
    if rep.get("chart_path"):
        print(f"\nchart: {rep['chart_path']}")
    print(f"\n⚠️  {rep['disclaimer']}")


def main():
    p = argparse.ArgumentParser(description="Intraday Selective Signal System (v3)")
    p.add_argument("--mode", required=True, choices=[
        "backfill", "bhavcopy", "corp-actions", "record", "screen", "geometry-study",
        "backtest", "train-v3", "promote", "rollback", "recalibrate", "paper",
        "drift", "gates", "serve",
        "regime-capture", "regime-backfill", "news-capture",   # PLAN_v4 §5/§8
        "daily-universe", "daily-panel", "daily-global", "daily-macro",   # src/daily
        "daily-fii-backfill", "daily-backtest", "daily-train", "daily-list",
        "daily-eval", "daily-scoreboard", "daily-serve", "daily-pipeline", "analyze",
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
    p.add_argument("--live", action="store_true", default=False,
                   help="daily-pipeline: run on a PROVISIONAL Upstox intraday snapshot "
                        "instead of waiting for the NSE EOD close (needs a valid token)")
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
        "daily-serve": cmd_daily_serve, "daily-pipeline": cmd_daily_pipeline, "analyze": cmd_analyze,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
