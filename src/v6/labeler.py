"""PLAN_v6 §4.2 — daily triple-barrier labeler on adjusted OHLC paths.

Entry = next day's open after the signal close (day D decides, D+1 opens the
trade). Barriers: target = entry + a·ATR_abs, stop = entry − b·ATR_abs, where
ATR_abs = atr_pct(D) × close(D) — strictly trailing. Time barrier: exit at
the close of trading day D+H.

Resolution order within each held day (conservative, matches v4 semantics):
  1. the OPEN gaps through a barrier → filled AT THE OPEN (gap-through stops
     fill below the stop level — the fill is the open, not the stop);
  2. intraday both barriers inside the day's range → counted as a STOP
     (conservative dual-touch);
  3. intraday stop; 4. intraday target — filled at the barrier level.
On the entry day itself the open IS the entry, so step 1 is skipped.

win := target exit, or time exit with positive return. Short mirror labels
(drift diagnostic only, F3/F6) flip barriers and sign.

Vectorized with (n_signals × H) windows per symbol per cell — the 48-cell
grid over ~78k universe signal rows resolves in seconds, not hours.

**F26 (review 2, 2026-07-10):** the signal-day range was `n - H - 1`, one
short of `n - H` — the last valid signal day per (symbol, cell) was silently
dropped. Fixed; verified against the hand-computed path tests, which only
assert on `iloc[0]` and are unaffected. Also adds `ca_in_hold` (F19): whether
a corporate-action ex-date (see src.v6.panel, F25) falls in
[entry_date, exit_date] for that trade — the disclosure PLAN_v6 §4.1
promised but the original implementation never wired up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_A = (0.5, 0.75, 1.0, 1.5)   # target multiple
GRID_B = (2, 3, 4, 5)            # stop multiple
GRID_H = (2, 3, 5)               # hold horizon, trading days

EXIT_TARGET, EXIT_STOP, EXIT_TIME = "target", "stop", "time"


@dataclass(frozen=True)
class Cell:
    a: float
    b: float
    h: int

    @property
    def key(self) -> str:
        return f"a{self.a}_b{self.b}_h{self.h}"


def all_cells() -> list[Cell]:
    return [Cell(a, b, h) for a in GRID_A for b in GRID_B for h in GRID_H]


def label_symbol(sym_df: pd.DataFrame, cell: Cell, side: str = "long",
                 ca_ex_dates: np.ndarray | None = None) -> pd.DataFrame:
    """Resolve every eligible signal day for one symbol under one cell.

    sym_df: one symbol's panel rows sorted by date (adjusted OHLC + atr_pct).
    ca_ex_dates: this symbol's corporate-action ex-dates (F19), used to flag
    holds that straddle one; None/empty disables the check (ca_in_hold=False).
    Returns one row per signal with exit type, return, and gap diagnostics.
    """
    o = sym_df["open"].to_numpy(float)
    h = sym_df["high"].to_numpy(float)
    l = sym_df["low"].to_numpy(float)  # noqa: E741
    c = sym_df["close"].to_numpy(float)
    atr_abs = (sym_df["atr_pct"] * sym_df["close"]).to_numpy(float)
    dates = sym_df["date"].to_numpy()
    n = len(sym_df)
    H = cell.h

    sig = np.arange(n - H)                           # signal day index D (F26: was n-H-1, dropped the last valid signal)
    sig = sig[np.isfinite(atr_abs[sig]) & (atr_abs[sig] > 0)]
    if len(sig) == 0:
        return pd.DataFrame()

    entry = o[sig + 1]
    if side == "long":
        target = entry + cell.a * atr_abs[sig]
        stop = entry - cell.b * atr_abs[sig]
    else:                                            # short mirror (diagnostic)
        target = entry - cell.a * atr_abs[sig]
        stop = entry + cell.b * atr_abs[sig]

    m = len(sig)
    # (m, H) windows of the held days D+1 .. D+H
    idx = sig[:, None] + 1 + np.arange(H)[None, :]
    O, Hi, Lo = o[idx], h[idx], l[idx]

    if side == "long":
        gap_stop = O <= stop[:, None]
        gap_tgt = O >= target[:, None]
        hit_stop = Lo <= stop[:, None]
        hit_tgt = Hi >= target[:, None]
    else:
        gap_stop = O >= stop[:, None]
        gap_tgt = O <= target[:, None]
        hit_stop = Hi >= stop[:, None]
        hit_tgt = Lo <= target[:, None]
    gap_stop[:, 0] = False                           # entry day: open IS the entry
    gap_tgt[:, 0] = False

    # per-day outcome code: 0 none, 1 stop, 2 target — priority: gap-stop,
    # gap-target, then intraday with dual-touch => stop
    day_code = np.zeros((m, H), dtype=np.int8)
    day_fill = np.zeros((m, H))
    for k in range(H):
        gs, gt = gap_stop[:, k], gap_tgt[:, k] & ~gap_stop[:, k]
        intraday = ~gs & ~gt
        st = intraday & hit_stop[:, k]               # dual-touch collapses here
        tg = intraday & ~hit_stop[:, k] & hit_tgt[:, k]
        day_code[gs | st, k] = 1
        day_code[gt | tg, k] = 2
        day_fill[gs | gt, k] = O[gs | gt, k]
        day_fill[st, k] = stop[st]
        day_fill[tg, k] = target[tg]

    hit_any = day_code != 0
    first = np.argmax(hit_any, axis=1)               # first hit day offset
    resolved = hit_any[np.arange(m), first]
    code = np.where(resolved, day_code[np.arange(m), first], 0)
    fill = np.where(resolved, day_fill[np.arange(m), first], c[idx[:, -1]])
    exit_off = np.where(resolved, first, H - 1)

    raw_ret = fill / entry - 1.0
    ret = raw_ret if side == "long" else -raw_ret
    exit_type = np.where(code == 2, EXIT_TARGET, np.where(code == 1, EXIT_STOP, EXIT_TIME))
    win = (code == 2) | ((code == 0) & (ret > 0))

    # gap-through severity: stop exits filled beyond the stop level
    stop_dist = np.abs(entry - stop)
    if side == "long":
        beyond = np.where(code == 1, np.maximum(0.0, stop - fill), 0.0)
    else:
        beyond = np.where(code == 1, np.maximum(0.0, fill - stop), 0.0)
    dual = hit_stop & hit_tgt
    dual_first = dual[np.arange(m), first] & resolved & ~gap_stop[np.arange(m), first] \
        & ~gap_tgt[np.arange(m), first]

    exit_idx = idx[np.arange(m), exit_off]

    # F19/F25: does any corporate-action ex-date fall in [entry_date, exit_date]?
    if ca_ex_dates is not None and len(ca_ex_dates):
        ex_flag = np.isin(dates, ca_ex_dates)
        cum_ex = np.cumsum(ex_flag)
        ca_in_hold = (cum_ex[exit_idx] - cum_ex[sig]) > 0
    else:
        ca_in_hold = np.zeros(m, dtype=bool)

    return pd.DataFrame({
        "symbol": sym_df["symbol"].iloc[0],
        "signal_date": dates[sig],
        "entry_date": dates[sig + 1],
        "exit_date": dates[exit_idx],
        "entry": entry,
        "atr_pct": atr_abs[sig] / c[sig],
        "exit_type": exit_type,
        "win": win,
        "ret": ret,
        "exit_day_offset": exit_off + 1,
        "gap_through": beyond > 0,
        "loss_beyond_stop_frac": np.where(stop_dist > 0, beyond / stop_dist, 0.0),
        "dual_touch": dual_first,
        "exit_locked": sym_df["locked_day"].to_numpy()[exit_idx],
        # calendar days between the exit day and the prior trading day:
        # > 1 means the overnight gap spanned a weekend/holiday (F17)
        "multi_night_gap": _prev_gap_days(dates)[exit_idx] > 1,
        "ca_in_hold": ca_in_hold,
    })


def _prev_gap_days(dates: np.ndarray) -> np.ndarray:
    days = dates.astype("datetime64[D]").astype(int)
    return np.diff(days, prepend=days[0])


def label_universe(panel: pd.DataFrame, in_universe: pd.Series, cell: Cell,
                   side: str = "long",
                   ca_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label every in-universe signal row for one cell across all symbols.

    Membership gates the SIGNAL day only — the resolution path always runs on
    the symbol's full contiguous series, so a name rotating out of the
    universe mid-hold still resolves on real prices (delistings simply
    truncate the series and are counted upstream as tail events).

    ca_events: the combined (symbol, ex_date) table from src.v6.panel
    (F19/F25) — if given, each label row is flagged `ca_in_hold`.
    """
    sig_keys = set(
        panel.loc[in_universe, "symbol"] + "|"
        + panel.loc[in_universe, "date"].dt.strftime("%Y-%m-%d")
    )
    eligible_syms = panel.loc[in_universe, "symbol"].unique()
    ca_by_symbol = ({sym: g["ex_date"].to_numpy()
                     for sym, g in ca_events.groupby("symbol", sort=False)}
                    if ca_events is not None else {})
    frames = []
    for sym, sym_df in panel[panel["symbol"].isin(eligible_syms)].groupby("symbol", sort=False):
        res = label_symbol(sym_df.reset_index(drop=True), cell, side,
                           ca_ex_dates=ca_by_symbol.get(sym))
        if len(res):
            key = sym + "|" + pd.to_datetime(res["signal_date"]).dt.strftime("%Y-%m-%d")
            res = res[key.isin(sig_keys)]
        if len(res):
            frames.append(res)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["cell"] = cell.key
    return out
