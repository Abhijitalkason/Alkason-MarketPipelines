"""Live session recorder (PLAN_v3 §6.2; v3_supplementry §10.1).

Captures: pre-open auction snapshot, 1-min bars built from quote polling, L2
depth snapshots for screened names. Everything stamped with capture_ts (IST).

Key v3 fixes:
  - bars_today(symbol): builds 1-min OHLCV from the in-memory tick buffer ON
    DEMAND so the 09:30 screen and intraday features see today's bars live
    (B-2 — the train/serve-skew fix);
  - incremental crash-safe flush: completed minutes appended to a session
    spill dir every 15 min; on restart the buffer rehydrates (A.3);
  - the MONTHLY store is touched once, at session close, and never with a
    partial day (H-9): flush_bars refuses before session_close unless force;
  - post-close reconcile replaces provisional bars with official candles, so
    poll-built bars never enter training (load_1min default excludes them);
  - poll failures counted; >10% failed polls ⇒ degraded session raises.

All clocks via now_ist() (M-7).
"""

from __future__ import annotations

import json
import logging
import time as time_mod
from datetime import date, time
from pathlib import Path

import pandas as pd

from src.intraday import ROOT, assert_clock_sane, load_config, load_universe, now_ist, today_ist
from src.intraday.bhavcopy import fetch_preopen
from src.intraday.data_feed import bars_dir, make_feed, save_monthly

logger = logging.getLogger(__name__)


def _wait_until(t: time) -> None:
    while now_ist().time() < t:
        time_mod.sleep(5)


class SessionRecorder:
    def __init__(self, day: date | None = None):
        self.cfg = load_config()
        self.day = day or today_ist()
        self.feed = make_feed(request_pause=0.05)
        self.symbols = load_universe(as_of=self.day)["symbol"].tolist()
        depth_base = Path(self.cfg["data"]["depth_path"])
        depth_base = depth_base if depth_base.is_absolute() else ROOT / depth_base
        self.depth_dir = depth_base / self.day.isoformat()
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self._ticks: list[pd.DataFrame] = []
        self._screened: list[str] = []
        self._poll_ok = 0
        self._poll_fail = 0
        self._spill = bars_dir(".session") / self.day.isoformat()
        self._spill.mkdir(parents=True, exist_ok=True)
        self._last_spill_minute: pd.Timestamp | None = None
        self._rehydrate()

    # ── crash safety ──────────────────────────────────────────────────

    def _rehydrate(self) -> None:
        spilled = sorted(self._spill.glob("*.parquet"))
        if spilled:
            self._ticks = [pd.read_parquet(p) for p in spilled]
            logger.info("recorder rehydrated %d tick chunks from %s", len(spilled), self._spill)

    def _spill_completed_minutes(self) -> None:
        """Append the last 15 min of completed ticks to the spill dir (A.3)."""
        if not self._ticks:
            return
        cur = pd.concat(self._ticks, ignore_index=True)
        chunk = self._spill / f"{now_ist().strftime('%H%M')}.parquet"
        cur.to_parquet(chunk, index=False)

    # ── phases ────────────────────────────────────────────────────────

    def auth_ping(self) -> None:
        assert_clock_sane()
        self.feed.auth_ping()  # raises before market open on a stale token

    def capture_preopen(self) -> pd.DataFrame:
        _wait_until(time(9, 8))
        df = fetch_preopen()
        logger.info("pre-open captured: %d symbols", len(df))
        return df

    def set_screened(self, symbols: list[str]) -> None:
        self._screened = symbols
        (self.depth_dir / "screened.json").write_text(json.dumps(symbols))

    def poll_quotes_once(self) -> pd.DataFrame:
        q = self.feed.quotes(self.symbols)
        q["minute"] = pd.to_datetime(q["capture_ts"]).dt.floor("1min")
        self._ticks.append(q[["symbol", "ltp", "volume", "minute", "capture_ts", "bid", "ask"]])
        self._poll_ok += 1
        if self._screened:
            depth = q[q.symbol.isin(self._screened)]
            depth.to_parquet(self.depth_dir / f"{now_ist().strftime('%H%M')}.parquet", index=False)
        return q

    def bars_today(self, symbol: str) -> pd.DataFrame:
        """Provisional 1-min OHLCV for `symbol` so far today, from the in-memory
        tick buffer (B-2). Same schema/index as load_1min output, provisional."""
        if not self._ticks:
            return pd.DataFrame()
        ticks = pd.concat(self._ticks, ignore_index=True)
        g = ticks[ticks.symbol == symbol].sort_values("capture_ts")
        if g.empty:
            return pd.DataFrame()
        bars = g.groupby("minute").agg(
            open=("ltp", "first"), high=("ltp", "max"),
            low=("ltp", "min"), close=("ltp", "last"),
            cum_volume=("volume", "last"),
        )
        bars["volume"] = bars["cum_volume"].diff().fillna(bars["cum_volume"]).clip(lower=0)
        bars = bars.drop(columns="cum_volume")
        bars.index.name = "ts"
        bars["capture_ts"] = now_ist()
        bars["provisional"] = True
        bars["close_raw"] = bars["close"]
        bars["adj_factor"] = 1.0
        return bars

    def history_with_today(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Stored official history + today's provisional bars, ts-indexed — the
        frame the live screen and features consume (train == serve)."""
        from datetime import timedelta
        from src.intraday.bars import load_1min

        try:
            hist = load_1min(symbol, self.day - timedelta(days=lookback_days), self.day)
        except Exception:  # noqa: BLE001 — first day for a symbol may have no store yet
            hist = pd.DataFrame()
        today = self.bars_today(symbol)
        frames = [f for f in (hist, today) if not f.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()

    # ── session loop (standalone recorder mode) ───────────────────────

    def run_session(self) -> None:
        cfg = self.cfg["data"]
        close_t = time.fromisoformat(cfg["session_close"])
        poll_s = cfg["live_poll_seconds"]
        self.auth_ping()
        self.capture_preopen()
        _wait_until(time.fromisoformat(cfg["session_open"]))
        logger.info("session open — polling %d symbols", len(self.symbols))
        while now_ist().time() < close_t:
            try:
                self.poll_quotes_once()
            except Exception as e:  # noqa: BLE001 — counted; degraded session raises at flush
                self._poll_fail += 1
                logger.error("poll failed: %s", e)
            now = now_ist()
            if self._last_spill_minute is None or (now - self._last_spill_minute).total_seconds() >= 900:
                self._spill_completed_minutes()
                self._last_spill_minute = now
            time_mod.sleep(poll_s)
        self.flush_bars()
        self.reconcile_all()

    def flush_bars(self, force: bool = False) -> None:
        """Convert the day's ticks into provisional 1-min bars and persist to the
        monthly store. Refuses before session_close unless force (H-9 — a partial
        day must never merge into the store)."""
        close_t = time.fromisoformat(self.cfg["data"]["session_close"])
        if not force and now_ist().time() < close_t:
            raise RuntimeError("flush_bars before session close would merge a partial day (use force=True in tests only)")
        if not self._ticks:
            raise RuntimeError("recorder captured zero ticks — session lost, investigate feed")
        ticks = pd.concat(self._ticks, ignore_index=True)
        n_minutes = ticks["minute"].nunique()
        total_polls = self._poll_ok + self._poll_fail
        degraded = total_polls > 0 and self._poll_fail / total_polls > 0.10
        for sym, g in ticks.groupby("symbol"):
            g = g.sort_values("capture_ts")
            bars = g.groupby("minute").agg(
                open=("ltp", "first"), high=("ltp", "max"),
                low=("ltp", "min"), close=("ltp", "last"),
                cum_volume=("volume", "last"),
            )
            bars["volume"] = bars["cum_volume"].diff().fillna(bars["cum_volume"]).clip(lower=0)
            bars = bars.drop(columns="cum_volume").reset_index().rename(columns={"minute": "ts"})
            save_monthly(sym, bars, provisional=True)
        logger.info("flushed %d symbols × %d minutes (provisional)", ticks.symbol.nunique(), n_minutes)
        if degraded:
            raise RuntimeError(
                f"session degraded: {self._poll_fail}/{total_polls} polls failed (>10%) — "
                f"bars persisted but flagged; investigate before trusting the day"
            )

    def reconcile_all(self) -> None:
        """Post-close: replace each recorded symbol's provisional bars with
        official candles + bhavcopy cross-check (the live skew detector).

        One symbol's failure must not abort the rest, but failures are COUNTED
        and a wholesale failure RAISES — silent reconcile loss would leave
        provisional bars in the store with no alarm (no-silent-degradation)."""
        from src.intraday.data_feed import reconcile_day
        recorded = self._screened or self.symbols
        failed = []
        for sym in recorded:
            try:
                reconcile_day(sym, self.day, self.feed)
            except Exception as e:  # noqa: BLE001 — counted; wholesale failure raises below
                logger.error("reconcile %s failed: %s", sym, e)
                failed.append(sym)
        if recorded and len(failed) > len(recorded) * 0.5:
            raise RuntimeError(
                f"reconcile failed for {len(failed)}/{len(recorded)} recorded symbols "
                f"— provisional bars remain unverified; investigate the feed before next session"
            )


def run() -> None:
    SessionRecorder().run_session()
