"""Paper-trading runner (PLAN_v3 Section 14 / Gate 4).

Runs the FULL live loop with simulated orders at real quoted prices:
recorder → 09:30 screen → features per 15-min close → blend → ACI gate →
simulated fills → barrier tracking → 14:45 square-off → post-market ACI update.

Every signal is logged with its full feature vector and quoted-vs-assumed fill —
this log is the train/serve-skew detector. Pass criterion: ≥22 sessions, live
win rate within 3 points of backtest, positive expectancy, slippage in model.

Run: python main.py --mode paper --capital 1000000
"""

from __future__ import annotations

import json
import logging
import time as time_mod
from datetime import date, datetime, time

import pandas as pd

from src.intraday import ROOT, load_config
from src.intraday.blend import Blender
from src.intraday.conformal_gate import ACIGate
from src.intraday.costs import round_trip_cost_pct
from src.intraday.features import features_for_day
from src.intraday.flow_model import FlowModel
from src.intraday.price_model import PriceModel
from src.intraday.recorder import SessionRecorder, _wait_until
from src.intraday.risk import RiskManager, is_blackout, is_halted, position_size
from src.intraday.screener import screen_day

logger = logging.getLogger(__name__)


class PaperRunner:
    def __init__(self, capital_inr: float):
        self.cfg = load_config()
        state = ROOT / self.cfg["paths"]["state"]
        self.price_model = PriceModel.load(state / "price_model.joblib")
        self.flow_model = FlowModel.load(state / "flow_model.joblib")
        self.blender = Blender.load(state / "blender.joblib")
        self.gate = ACIGate.load()
        self.risk = RiskManager(capital_inr)
        self.capital = capital_inr
        self.recorder = SessionRecorder()
        self.positions: dict[str, dict] = {}
        self.closed: list[dict] = []
        self.log_dir = ROOT / self.cfg["paths"]["signals_log"]
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── live evaluation at each 15-min close ──────────────────────────

    def evaluate(self, screened: pd.DataFrame, day: date) -> None:
        geo = self.cfg["geometry"]
        for _, s in screened.iterrows():
            sym = s["symbol"]
            if sym in self.positions:
                continue
            ok, reason = self.risk.can_open(sym, day)
            if not ok:
                continue
            feats = features_for_day(sym, day, int(s["direction"]))
            if feats.empty:
                continue
            row = feats.iloc[[-1]]                                   # latest closed decision bar
            p_p = self.price_model.predict_proba(row)
            p_f = self.flow_model.predict_proba(row)
            p_cal = self.blender.calibrated(p_p, p_f)
            if not bool(self.gate.fire(p_cal, p_p, p_f)[0]):
                continue

            quote = self.recorder.poll_quotes_once()
            q = quote[quote.symbol == sym]
            if q.empty or pd.isna(q["ask"].iloc[0]):
                continue
            direction = int(s["direction"])
            entry = float(q["ask"].iloc[0] if direction == 1 else q["bid"].iloc[0])
            atr = float(row["atr_pct"].iloc[0]) * entry
            qty = position_size(self.capital, entry, entry - direction * geo["stop_atr"] * atr)
            if qty == 0:
                continue
            self.positions[sym] = {
                "symbol": sym, "direction": direction, "entry": entry, "qty": qty,
                "target": entry + direction * geo["target_atr"] * atr,
                "stop": entry - direction * geo["stop_atr"] * atr,
                "entry_ts": datetime.now(),
                "t_bar": datetime.now() + pd.Timedelta(hours=geo["time_barrier_hours"]),
                "p_cal": float(p_cal[0]),
                "features": row.drop(columns=["symbol", "ts", "date"]).iloc[0].to_dict(),
            }
            self.risk.on_open(sym)
            logger.info("PAPER OPEN %s dir=%+d entry=%.2f qty=%d p=%.3f", sym, direction, entry, qty, p_cal[0])

    def manage_positions(self) -> None:
        if not self.positions:
            return
        quotes = self.recorder.poll_quotes_once()
        squareoff = time.fromisoformat(self.cfg["geometry"]["squareoff"])
        for sym in list(self.positions):
            pos = self.positions[sym]
            q = quotes[quotes.symbol == sym]
            if q.empty:
                continue
            ltp = float(q["ltp"].iloc[0])
            now = datetime.now()
            d = pos["direction"]
            exit_reason = None
            if (d == 1 and ltp >= pos["target"]) or (d == -1 and ltp <= pos["target"]):
                exit_reason = "target"
            elif (d == 1 and ltp <= pos["stop"]) or (d == -1 and ltp >= pos["stop"]):
                exit_reason = "stop"
            elif now >= pos["t_bar"] or now.time() >= squareoff:
                exit_reason = "time"
            if exit_reason:
                self._close(sym, ltp, exit_reason)

    def _close(self, sym: str, exit_price: float, reason: str) -> None:
        pos = self.positions.pop(sym)
        gross = pos["direction"] * (exit_price - pos["entry"]) / pos["entry"]
        cost = round_trip_cost_pct(pos["entry"], pos["qty"])
        net = gross - cost
        pnl_inr = net * pos["entry"] * pos["qty"]
        self.risk.on_close(sym, pnl_inr)
        rec = {**{k: v for k, v in pos.items() if k != "features"},
               "exit": exit_price, "barrier": reason, "exit_ts": datetime.now(),
               "pnl_pct_gross": gross, "cost_pct": cost, "pnl_pct_net": net,
               "label": int(net > 0) if reason == "time" else int(reason == "target"),
               "features": pos["features"]}
        self.closed.append(rec)
        logger.info("PAPER CLOSE %s %s net=%.3f%%", sym, reason, net * 100)

    # ── session loop ──────────────────────────────────────────────────

    def run_session(self) -> None:
        day = date.today()
        if is_halted():
            raise RuntimeError("kill switch active (models/v3/halt.json) — review before running")
        if is_blackout(day):
            logger.info("%s is a blackout day — no signals; recorder still runs", day)
            self.recorder.run_session()
            return

        self.recorder.capture_preopen()
        _wait_until(time(9, 30, 5))
        screened = screen_day(day)
        self.recorder.set_screened(screened.symbol.tolist())

        w_end = time.fromisoformat(self.cfg["geometry"]["entry_window"][1])
        close_t = time.fromisoformat(self.cfg["data"]["session_close"])
        while datetime.now().time() < close_t:
            now = datetime.now()
            self.manage_positions()
            # evaluate on each 15-min boundary within the entry window
            if now.time() < w_end and now.minute % 15 == 0:
                self.evaluate(screened, day)
            time_mod.sleep(30)
            if not self.positions and datetime.now().time() >= w_end:
                break

        for sym in list(self.positions):                      # safety square-off
            q = self.recorder.poll_quotes_once()
            row = q[q.symbol == sym]
            if not row.empty:
                self._close(sym, float(row["ltp"].iloc[0]), "time")
        self.recorder.flush_bars()
        self._post_market(day)

    def _post_market(self, day: date) -> None:
        self.gate.update([c["label"] for c in self.closed])
        self.gate.save()
        out = self.log_dir / f"paper_{day.isoformat()}.json"
        out.write_text(json.dumps(self.closed, indent=2, default=str))
        n = len(self.closed)
        wr = sum(c["label"] for c in self.closed) / n if n else float("nan")
        logger.info("session done: %d trades, win rate %.2f → %s", n, wr, out.name)


def run(capital_inr: float = 1_000_000) -> None:
    PaperRunner(capital_inr).run_session()
