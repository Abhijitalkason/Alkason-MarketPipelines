"""Paper-trading runner (PLAN_v3 §14 / Gate 4; v3_supplementry §10.2).

Full live loop with simulated orders at real quoted prices:
recorder (live bars) → 09:30 screen → features per 15-min close → blend → ACI
gate → simulated fills at bid/ask → barrier tracking → 14:45 square-off →
post-market label maturation + ACI update + kill-switch + drift checks.

Critical fixes vs the skeleton:
  - B-2: the screen and features consume recorder.history_with_today(), so the
    09:30 screen sees today's first bar and features see intraday bars live;
  - H-10: check_schema before every serve-time predict;
  - restart safety (A.3): positions, closed trades, risk DayState, and the tick
    buffer all rehydrate from session_state_<date>.json + the recorder spill;
  - B-7: check_kill_switches and drift.run_check have their first/live callers
    here, post-market;
  - every screen / gate-eval / fill / exit / aci_update is journaled.

Loads the Staging bundle from the registry (fallback: models/v3 files).
"""

from __future__ import annotations

import json
import logging
import threading
import time as time_mod
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from src.intraday import ROOT, load_config, now_ist, today_ist
from src.intraday.blend import Blender
from src.intraday.conformal_gate import ACIGate
from src.intraday.costs import round_trip_cost_pct
from src.intraday.features import check_schema, features_for_day
from src.intraday.flow_model import FlowModel
from src.intraday.journal import journal_write, seal_day
from src.intraday.price_model import PriceModel
from src.intraday.recorder import SessionRecorder, _wait_until
from src.intraday.risk import (DayState, RiskManager, check_kill_switches, is_blackout,
                               is_halted, position_size)
from src.intraday.screener import screen_day

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 45


class PaperRunner:
    def __init__(self, capital_inr: float, day: date | None = None):
        self.cfg = load_config()
        self.day = day or today_ist()
        self.capital = capital_inr
        self.state_dir = ROOT / self.cfg["paths"]["state"]
        self._load_bundle()
        self.recorder = SessionRecorder(day=self.day)
        self.positions: dict[str, dict] = {}
        self.closed: list[dict] = []
        self.screened: pd.DataFrame | None = None
        self.risk = RiskManager(capital_inr, as_of=self.day)
        self.log_dir = Path(self.cfg["paths"]["signals_log"])
        self.log_dir = self.log_dir if self.log_dir.is_absolute() else ROOT / self.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_state = self.state_dir / f"session_state_{self.day.isoformat()}.json"
        self._restore_state()

    def _load_bundle(self) -> None:
        """Staging from the registry; fallback to models/v3 files with a WARN."""
        try:
            from src.intraday.tracking import load_stage  # type: ignore[attr-defined]
            bundle = load_stage("Staging")
            self.price_model, self.flow_model, self.blender = bundle
        except Exception as e:  # noqa: BLE001 — registry optional in local paper runs
            logger.warning("registry Staging load failed (%s) — using models/v3 files", e)
            self.price_model = PriceModel.load(self.state_dir / "price_model.joblib")
            self.flow_model = (FlowModel.load(self.state_dir / "flow_model.joblib")
                               if (self.state_dir / "flow_model.joblib").exists() else None)
            self.blender = Blender.load(self.state_dir / "blender.joblib")
        self.gate = ACIGate.load()

    # ── restart safety ────────────────────────────────────────────────

    def _restore_state(self) -> None:
        if self._session_state.exists():
            s = json.loads(self._session_state.read_text())
            self.positions = s.get("positions", {})
            self.closed = s.get("closed", [])
            self.risk.state = DayState.from_dict(s["day_state"])
            logger.info("paper: resumed session %s (%d open, %d closed)",
                        self.day, len(self.positions), len(self.closed))

    def _persist_state(self) -> None:
        self._session_state.write_text(json.dumps({
            "positions": self.positions, "closed": self.closed,
            "day_state": self.risk.state.to_dict(),
        }, default=str))

    # ── live evaluation at each 15-min close ──────────────────────────

    def evaluate(self) -> None:
        regime_active = bool(self.cfg.get("regime", {}).get("active", False))
        for _, s in self.screened.iterrows():
            sym = s["symbol"]
            if sym in self.positions:
                continue
            ok, reason = self.risk.can_open(sym, self.day)
            if not ok:
                continue
            df_1min = self.recorder.history_with_today(sym, LOOKBACK_DAYS)
            if df_1min.empty:
                continue
            feats = features_for_day(sym, self.day, int(s["direction"]), df_1min=df_1min)
            if feats.empty:
                continue
            row = feats.iloc[[-1]]                          # latest closed decision bar
            check_schema(row)                               # H-10 serve-time guard
            p_p = self.price_model.predict_proba(row)
            p_f = self.flow_model.predict_proba(row) if self.flow_model is not None else None
            p_cal = self.blender.calibrated(p_p, p_f)
            # PLAN_v4 §7 — per-candidate direction veto immediately before the gate.
            # Evaluated AFTER features/probabilities so the journal line carries the
            # full counterfactual scaffold: once the day's bars mature, this exact
            # row can be labeled and the vetoed signal scored (would it have won?).
            if regime_active:
                from src.intraday.regime import veto
                vetoed, vreason = veto(self.day, int(s["direction"]))
                if vetoed:
                    journal_write("gate_eval", {
                        "symbol": sym, "direction": int(s["direction"]),
                        "p_price": float(p_p[0]),
                        "p_flow": float(p_f[0]) if p_f is not None else None,
                        "p_cal": float(p_cal[0]), "tau": self.gate.state.tau,
                        "fired": False, "skipped": vreason,
                        "features": row.drop(columns=["symbol", "ts", "date"],
                                             errors="ignore").iloc[0].to_dict(),
                    }, day=self.day)
                    continue
            fired = bool(self.gate.fire(p_cal, p_p, p_f)[0])
            journal_write("gate_eval", {
                "symbol": sym, "p_price": float(p_p[0]),
                "p_flow": float(p_f[0]) if p_f is not None else None,
                "p_cal": float(p_cal[0]), "tau": self.gate.state.tau, "fired": fired,
            }, day=self.day)
            if not fired:
                continue
            self._open(sym, int(s["direction"]), row, float(p_cal[0]))

    def _open(self, sym: str, direction: int, row: pd.DataFrame, p_cal: float) -> None:
        geo = self.cfg["geometry"]
        quote = self.recorder.poll_quotes_once()
        q = quote[quote.symbol == sym]
        if q.empty or pd.isna(q["ask"].iloc[0]) or pd.isna(q["bid"].iloc[0]):
            journal_write("gate_eval", {"symbol": sym, "fired": True, "skipped": "no_quote"}, day=self.day)
            return
        entry = float(q["ask"].iloc[0] if direction == 1 else q["bid"].iloc[0])
        atr = float(row["atr_pct"].iloc[0]) * entry
        stop = entry - direction * geo["stop_atr"] * atr
        qty = position_size(self.capital, entry, stop)
        if qty == 0:
            journal_write("gate_eval", {"symbol": sym, "fired": True, "skipped": "zero_qty"}, day=self.day)
            return
        try:
            shap = self.price_model.shap_top(row, n=10)
        except Exception as e:  # noqa: BLE001 — SHAP is explanatory, not gating
            shap = {}
            logger.warning("SHAP failed for %s: %s", sym, e)
        self.positions[sym] = {
            "symbol": sym, "direction": direction, "entry": entry, "qty": qty,
            "target": entry + direction * geo["target_atr"] * atr, "stop": stop,
            "entry_ts": now_ist().isoformat(),
            "t_bar": (now_ist() + timedelta(hours=geo["time_barrier_hours"])).isoformat(),
            "p_cal": p_cal,
            "last_ltp": entry,   # last known mark — square-off falls back to it if the quote dies
            "features": row.drop(columns=["symbol", "ts", "date"], errors="ignore").iloc[0].to_dict(),
        }
        self.risk.on_open(sym)
        fill_payload = {"symbol": sym, "direction": direction, "entry": entry,
                        "qty": qty, "p_cal": p_cal, "shap_top": shap,
                        "quoted_bid": float(q["bid"].iloc[0]),
                        "quoted_ask": float(q["ask"].iloc[0])}
        journal_write("fill", fill_payload, day=self.day)
        self._explain_async(fill_payload)        # PLAN_v3 §15: fired-signal narration, off the decision path
        self._persist_state()
        logger.info("PAPER OPEN %s dir=%+d entry=%.2f qty=%d p=%.3f", sym, direction, entry, qty, p_cal)

    def _explain_async(self, fill_payload: dict) -> None:
        """Granite narration of a FIRED signal, generated on a background thread
        (never on the decision path) and appended as a journal `explain` event,
        surfaced by GET /signals/today (PLAN_v3 §15). Failure is non-fatal."""
        day = self.day

        def _run() -> None:
            try:
                from src.slm.explainer import explain_signal
                text = explain_signal(fill_payload)
                journal_write("explain", {"symbol": fill_payload["symbol"], "text": text}, day=day)
            except Exception as e:  # noqa: BLE001 — explanation is UX, never gating
                logger.warning("explain failed for %s: %s", fill_payload.get("symbol"), e)

        threading.Thread(target=_run, daemon=True).start()

    def manage_positions(self) -> None:
        if not self.positions:
            return
        quotes = self.recorder.poll_quotes_once()
        squareoff = time.fromisoformat(self.cfg["geometry"]["squareoff"])
        for sym in list(self.positions):
            pos = self.positions[sym]
            q = quotes[quotes.symbol == sym]
            if q.empty or pd.isna(q["ltp"].iloc[0]):
                logger.warning("no quote to manage %s this cycle", sym)
                continue
            ltp = float(q["ltp"].iloc[0])
            pos["last_ltp"] = ltp   # remember the mark for a stale-quote square-off
            now = now_ist()
            d = pos["direction"]
            reason = None
            if (d == 1 and ltp >= pos["target"]) or (d == -1 and ltp <= pos["target"]):
                reason = "target"
            elif (d == 1 and ltp <= pos["stop"]) or (d == -1 and ltp >= pos["stop"]):
                reason = "stop"
            elif now >= datetime.fromisoformat(pos["t_bar"]) or now.time() >= squareoff:
                reason = "time"
            if reason:
                self._close(sym, ltp, reason)

    def _close(self, sym: str, exit_price: float, reason: str) -> None:
        pos = self.positions.pop(sym)
        gross = pos["direction"] * (exit_price - pos["entry"]) / pos["entry"]
        try:
            mv = self.recorder.history_with_today(sym, LOOKBACK_DAYS)["volume"].tail(20).median()
            mv = float(mv) if mv and mv > 0 else self.cfg["universe"]["min_median_1min_volume"]
        except Exception:  # noqa: BLE001
            mv = self.cfg["universe"]["min_median_1min_volume"]
        cost = round_trip_cost_pct(pos["entry"], pos["qty"], mv, entry_is_quoted=True)
        net = gross - cost
        pnl_inr = net * pos["entry"] * pos["qty"]
        self.risk.on_close(sym, pnl_inr)
        # Label MUST match the backtest/labeler convention exactly (train==serve):
        # time-barrier exit is labelled on GROSS PnL sign (pre-cost), never net —
        # net would make the win/loss feeding the ACI gate and kill switches mean
        # something different live than in training. Expectancy uses net separately.
        rec = {**{k: v for k, v in pos.items() if k != "features"},
               "exit": exit_price, "barrier": reason, "exit_ts": now_ist().isoformat(),
               "pnl_pct_gross": gross, "cost_pct": cost, "pnl_pct_net": net,
               "label": int(gross > 0) if reason == "time" else int(reason == "target"),
               "features": pos["features"]}
        self.closed.append(rec)
        journal_write("exit", {"symbol": sym, "barrier": reason, "exit": exit_price,
                              "pnl_pct_net": net, "label": rec["label"]}, day=self.day)
        self._persist_state()
        logger.info("PAPER CLOSE %s %s net=%.3f%%", sym, reason, net * 100)

    # ── session loop ──────────────────────────────────────────────────

    def run_session(self) -> None:
        if is_halted():
            raise RuntimeError("kill switch active (models/v3/halt.json) — review before running")
        self.recorder.auth_ping()
        if is_blackout(self.day):
            logger.info("%s is a blackout day — recorder runs, no signals", self.day)
            self.recorder.run_session()
            return
        # PLAN_v4 §7 — day-level regime no-trade (hostile morning). Sits beside the
        # blackout check but is NOT is_blackout: the recorder still runs and the day
        # remains a labeled candidate day; we simply emit no signals.
        if bool(self.cfg.get("regime", {}).get("active", False)):
            from src.intraday.regime import day_regime
            rv = day_regime(self.day)
            if rv.no_trade:
                logger.info("%s regime no-trade (score=%.2f) — recorder runs, no signals",
                            self.day, rv.risk_score)
                journal_write("regime_no_trade", {"risk_score": rv.risk_score,
                                                  "bias": rv.bias}, day=self.day)
                self.recorder.run_session()
                return

        self.recorder.capture_preopen()
        # poll through the pre-screen window so today's first 15-min bar exists live
        _wait_until(time.fromisoformat(self.cfg["data"]["session_open"]))
        screen_t = time.fromisoformat(self.cfg["screen"]["time"])
        while now_ist().time() < screen_t:
            self.recorder.poll_quotes_once()
            time_mod.sleep(self.cfg["data"]["live_poll_seconds"])

        live = {s: self.recorder.history_with_today(s, self.cfg["screen"]["lookback_days"] * 2)
                for s in self.recorder.symbols}
        live = {k: v for k, v in live.items() if not v.empty}
        self.screened = screen_day(self.day, list(live.keys()), live_bars=live)
        self.recorder.set_screened(self.screened.symbol.tolist())
        journal_write("screen", {"top": self.screened.symbol.tolist(),
                               "scores": self.screened.set_index("symbol")["score"].round(4).to_dict()},
                      day=self.day)

        w_start = time.fromisoformat(self.cfg["geometry"]["entry_window"][0])
        w_end = time.fromisoformat(self.cfg["geometry"]["entry_window"][1])
        close_t = time.fromisoformat(self.cfg["data"]["session_close"])
        poll_s = self.cfg["data"]["live_poll_seconds"]
        # Explicit 15-min decision boundaries in the entry window (09:30, 09:45, …).
        # Evaluate each one exactly once, on the first poll AT OR AFTER it. A
        # minute-of-hour dedup would collide (10:30 has the same minute as 09:30)
        # and silently skip boundaries; this also survives an eval that runs >60s.
        boundaries, t = [], datetime.combine(self.day, w_start)
        w_end_dt = datetime.combine(self.day, w_end)
        while t < w_end_dt:
            boundaries.append(t.time())
            t += timedelta(minutes=15)
        evaluated: set = set()
        while now_ist().time() < close_t:
            now = now_ist()
            self.manage_positions()
            due = [b for b in boundaries if b <= now.time() and b not in evaluated]
            if due:
                self.evaluate()
                evaluated.update(due)
            time_mod.sleep(poll_s)

        # Hard square-off — PLAN_v3 §3/§5 "always flat, no exceptions". Every open
        # position MUST close here; if the quote is dead we close at the last known
        # mark rather than leaving the position to silently carry overnight risk.
        for sym in list(self.positions):
            q = self.recorder.poll_quotes_once()
            row = q[q.symbol == sym]
            if not row.empty and pd.notna(row["ltp"].iloc[0]):
                self._close(sym, float(row["ltp"].iloc[0]), "time")
            else:
                mark = self.positions[sym].get("last_ltp", self.positions[sym]["entry"])
                logger.warning("square-off %s on STALE quote — closing at last mark %.2f", sym, mark)
                self._close(sym, mark, "time")
        if self.positions:  # invariant: never carry a position past square-off
            raise RuntimeError(f"square-off failed to flatten: {list(self.positions)}")
        self.recorder.flush_bars()
        self.recorder.reconcile_all()
        self._post_market()

    def _post_market(self) -> None:
        labels = [c["label"] for c in self.closed]
        self.gate.update(labels)
        self.gate.save()
        journal_write("aci_update", {"tau": self.gate.state.tau, "fired_labels": labels}, day=self.day)

        out = self.log_dir / f"paper_{self.day.isoformat()}.json"
        out.write_text(json.dumps(self.closed, indent=2, default=str))

        # kill switches (B-7 — first live caller) over the full paper history
        all_trades = self._all_paper_trades()
        bt_wr = self._backtest_win_rate()
        breaches = check_kill_switches(all_trades, bt_wr)
        if breaches:
            logger.error("kill switches breached: %s", breaches)

        # drift check (B-7 switch c)
        try:
            from src.intraday.drift import run_check
            run_check(self.day)
        except Exception as e:  # noqa: BLE001 — drift infra optional in local runs
            logger.warning("drift check skipped: %s", e)

        seal_day(self.day)
        n = len(self.closed)
        wr = sum(labels) / n if n else float("nan")
        logger.info("session done: %d trades, win rate %.2f → %s", n, wr, out.name)

    def _all_paper_trades(self) -> pd.DataFrame:
        rows = []
        for f in sorted(self.log_dir.glob("paper_*.json")):
            try:
                d = date.fromisoformat(f.stem.replace("paper_", ""))
            except ValueError:
                continue
            for rec in json.loads(f.read_text()):
                rows.append({"date": d, "label": int(rec["label"]),
                             "pnl_pct_net": float(rec["pnl_pct_net"])})
        return pd.DataFrame(rows)

    def _backtest_win_rate(self) -> float:
        reg = ROOT / self.cfg["paths"]["run_registry"]
        if reg.exists():
            df = pd.read_csv(reg)
            passed = df[df["all_pass"] == True]  # noqa: E712
            if not passed.empty:
                return float(passed["win_rate"].iloc[-1])
        return self.cfg["gates"]["min_win_rate"]


def run(capital_inr: float = 1_000_000) -> None:
    PaperRunner(capital_inr).run_session()
