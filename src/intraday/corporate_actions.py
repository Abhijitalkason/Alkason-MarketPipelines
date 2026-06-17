"""Corporate-action adjustment factors (v3_supplementry §3.5, H-2).

Storage is NEVER adjusted in place — v1's adjustment-seam bug is impossible by
construction. Factors are applied on read by bars.load_1min(adjusted=True) as
explicit columns: adj_factor, adjusted o/h/l/c, close_raw retained.

Factor convention: factor = post-event price / pre-event price for the action
(1:2 split → 0.5; 1:1 bonus → 0.5). Prices strictly before the ex-date are
multiplied by the cumulative product of all factors with ex_date > ts;
volume is divided by the same factor.

Table file: data/reference/corporate_actions.csv
    symbol,ex_date,purpose,factor
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)


class CorporateActionError(RuntimeError):
    pass


def _table_path() -> Path:
    return ROOT / load_config()["data"]["reference_path"] / "corporate_actions.csv"


# ── table construction (from NSE corporate-action archive) ────────────

_SPLIT_RE = re.compile(r"FRM\s*RS?\.?\s*(\d+(?:\.\d+)?)\s*(?:/-)?\s*TO\s*RS?\.?\s*(\d+(?:\.\d+)?)", re.I)
_BONUS_RE = re.compile(r"BONUS\s+(\d+)\s*:\s*(\d+)", re.I)


def parse_purpose(purpose: str) -> float | None:
    """Price factor from an NSE 'purpose' string; None if not price-affecting
    (dividends are intraday-irrelevant at our horizon and are skipped)."""
    m = _SPLIT_RE.search(purpose)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if old_fv > 0 and new_fv > 0 and new_fv < old_fv:
            return new_fv / old_fv
    m = _BONUS_RE.search(purpose)
    if m:
        bonus, base = float(m.group(1)), float(m.group(2))
        if base > 0:
            return base / (base + bonus)
    return None


def build_table(start: date, end: date) -> pd.DataFrame:
    """Download NSE corporate actions for [start, end] and persist the factor
    table. Network failures raise (no silent stale table)."""
    from src.intraday.bhavcopy import _fetch, _session  # shared NSE session plumbing

    s = _session()
    url = (
        "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
        f"&from_date={start.strftime('%d-%m-%Y')}&to_date={end.strftime('%d-%m-%Y')}"
    )
    raw = _fetch(s, url)
    df = pd.read_json(io.BytesIO(raw))
    if df.empty:
        raise CorporateActionError(f"NSE corporate-action API returned 0 rows for {start}..{end}")
    rows = []
    for _, r in df.iterrows():
        factor = parse_purpose(str(r.get("subject", "")))
        if factor is not None:
            rows.append({
                "symbol": str(r["symbol"]).strip(),
                "ex_date": pd.to_datetime(r["exDate"], dayfirst=True).date(),
                "purpose": r["subject"],
                "factor": factor,
            })
    out = pd.DataFrame(rows, columns=["symbol", "ex_date", "purpose", "factor"])
    path = _table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, comment="#", parse_dates=["ex_date"])
        old["ex_date"] = old["ex_date"].dt.date
        out = pd.concat([old, out], ignore_index=True).drop_duplicates(
            subset=["symbol", "ex_date"], keep="last"
        )
    out.sort_values(["symbol", "ex_date"]).to_csv(path, index=False)
    load_factors.cache_clear()
    logger.info("corporate-action table: %d price-affecting rows → %s", len(out), path.name)
    return out


@lru_cache(maxsize=256)
def load_factors(symbol: str) -> tuple[tuple[date, float], ...]:
    """(ex_date, factor) pairs for a symbol, ascending. Empty table is valid
    (no actions); a MISSING table file is also valid until first build."""
    path = _table_path()
    if not path.exists():
        return ()
    df = pd.read_csv(path, comment="#", parse_dates=["ex_date"])
    df = df[df["symbol"].str.strip() == symbol].sort_values("ex_date")
    return tuple((d.date(), float(f)) for d, f in zip(df["ex_date"], df["factor"]))


def apply(df_1min: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Adjusted view of a ts-indexed 1-min frame. Adds adj_factor + close_raw;
    o/h/l/c become adjusted, volume divided by factor. Input is not mutated."""
    factors = load_factors(symbol)
    out = df_1min.copy()
    out["close_raw"] = out["close"]
    out["adj_factor"] = 1.0
    if not factors:
        return out
    for ex_date, factor in factors:
        mask = out.index.normalize() < pd.Timestamp(ex_date)
        out.loc[mask, "adj_factor"] *= factor
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * out["adj_factor"]
    out["volume"] = (out["volume"] / out["adj_factor"]).round().astype("int64")
    return out
