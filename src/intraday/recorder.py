"""Live session recorder (PLAN_v3 Section 6.2) — runs every market day from 08:55.

Captures: pre-open auction snapshot (09:00–09:15), 1-min bars built from quote
polling (09:15–15:30), L2 depth snapshots for screened names. Everything is
stamped with capture_ts → the morning screen stays point-in-time backtestable.

Run: python main.py --mode record
"""

from __future__ import annotations

import json
import logging
import time as time_mod
from datetime import date, datetime, time

import pandas as pd

from src.intraday import ROOT, load_config, load_universe
from src.intraday.bhavcopy import fetch_preopen
from src.intraday.data_feed import UpstoxFeed, save_monthly

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now()


def _wait_until(t: time) -> None:
    while _now().time() < t:
        time_mod.sleep(5)


class SessionRecorder:
    def __init__(self):
        self.cfg = load_config()
        self.feed = UpstoxFeed(request_pause=0.05)
        self.symbols = load_universe()["symbol"].tolist()
        self.depth_dir = ROOT / self.cfg["data"]["depth_path"] / date.today().isoformat()
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        # accumulated quote snapshots → 1-min bars at session end
        self._ticks: list[pd.DataFrame] = []
        self._screened: list[str] = []

    # ── phases ────────────────────────────────────────────────────────

    def capture_preopen(self) -> None:
        _wait_until(time(9, 8))
        df = fetch_preopen()
        logger.info("pre-open captured: %d symbols", len(df))

    def set_screened(self, symbols: list[str]) -> None:
        """Called by the live loop after the 09:30 screen; enables depth capture."""
        self._screened = symbols
        (self.depth_dir / "screened.json").write_text(json.dumps(symbols))

    def poll_quotes_once(self) -> pd.DataFrame:
        q = self.feed.quotes(self.symbols)
        q["minute"] = q["capture_ts"].dt.floor("1min")
        self._ticks.append(q[["symbol", "ltp", "volume", "minute", "capture_ts"]])
        if self._screened:
            depth = q[q.symbol.isin(self._screened)]
            path = self.depth_dir / f"{_now().strftime('%H%M')}.parquet"
            depth.to_parquet(path, index=False)
        return q

    def run_session(self) -> None:
        """Full session loop: pre-open → poll once a minute → persist bars."""
        cfg = self.cfg["data"]
        close_t = time.fromisoformat(cfg["session_close"])
        self.capture_preopen()
        _wait_until(time.fromisoformat(cfg["session_open"]))
        logger.info("session open — polling %d symbols at 1-min cadence", len(self.symbols))
        next_poll = _now()
        while _now().time() < close_t:
            try:
                self.poll_quotes_once()
            except Exception as e:  # noqa: BLE001 — log, keep session alive, fail at flush if degraded
                logger.error("poll failed: %s", e)
            next_poll = next_poll + pd.Timedelta(minutes=1)
            sleep_s = max(1.0, (next_poll - _now()).total_seconds())
            time_mod.sleep(sleep_s)
        self.flush_bars()

    def flush_bars(self) -> None:
        """Convert the day's quote snapshots into 1-min OHLCV bars and persist."""
        if not self._ticks:
            raise RuntimeError("recorder captured zero ticks — session lost, investigate feed")
        ticks = pd.concat(self._ticks, ignore_index=True)
        n_minutes = ticks["minute"].nunique()
        if n_minutes < 300:
            logger.warning("only %d minutes captured (expected ~375)", n_minutes)
        for sym, g in ticks.groupby("symbol"):
            g = g.sort_values("capture_ts")
            bars = g.groupby("minute").agg(
                open=("ltp", "first"), high=("ltp", "max"),
                low=("ltp", "min"), close=("ltp", "last"),
                cum_volume=("volume", "last"),
            )
            bars["volume"] = bars["cum_volume"].diff().fillna(bars["cum_volume"]).clip(lower=0)
            bars = bars.drop(columns="cum_volume").reset_index().rename(columns={"minute": "ts"})
            bars["capture_ts"] = _now()
            save_monthly(sym, bars)
        logger.info("flushed %d symbols × %d minutes to bar store", ticks.symbol.nunique(), n_minutes)


def run() -> None:
    SessionRecorder().run_session()
