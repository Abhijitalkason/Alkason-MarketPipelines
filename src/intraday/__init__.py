"""PLAN v3 intraday selective signal system — core utilities.

One config (config_v3.yaml), one clock (IST), one point-in-time universe
loader. Everything in src/intraday/ imports these from here.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config_v3.yaml"

# Load .env once at package import so UPSTOX_ACCESS_TOKEN (and other secrets) are
# available to every entry point — backfill, record, paper, serve — not just the
# FastAPI app. Silent no-op if python-dotenv is absent or .env is missing.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_config() -> dict:
    # Tests monkeypatch src.intraday.CONFIG_PATH and call load_config.cache_clear()
    import src.intraday as _self

    with open(_self.CONFIG_PATH) as f:
        return yaml.safe_load(f)


def data_path(key: str) -> Path:
    """Resolve a config path (data.* or paths.*) against ROOT.
    Absolute paths in config (used by tests) pass through unchanged."""
    cfg = load_config()
    raw = cfg["data"].get(key) if key in cfg["data"] else cfg["paths"][key]
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def now_ist() -> datetime:
    """The one clock. ALL session logic uses this (M-7). Returns tz-naive IST
    to match the tz-naive IST timestamps in the bar store."""
    return datetime.now(IST).replace(tzinfo=None)


def today_ist() -> date:
    return now_ist().date()


def assert_clock_sane() -> None:
    """Startup check (recorder/paper/serve): host clock vs IST offset awareness."""
    offset = abs((datetime.now() - now_ist()).total_seconds())
    if offset > 60:
        logger.error(
            "host clock differs from IST by %.0f s — session timing logic assumes IST; "
            "set the host timezone or expect mis-timed screens", offset
        )


def load_universe(as_of: date) -> pd.DataFrame:
    """Point-in-time universe — `as_of` is REQUIRED (no default), so
    survivorship bias cannot sneak back in (H-1).

    Membership file: symbol,name,sector,from_date,to_date (to_date empty =
    current member). Returns members where from_date <= as_of < to_date.
    """
    cfg = load_config()
    p = Path(cfg["universe"]["membership_file"])
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — build it with scripts/build_membership.py (PIT universe is mandatory)"
        )
    df = pd.read_csv(p, comment="#")
    df["symbol"] = df["symbol"].str.strip()
    as_of_ts = pd.Timestamp(as_of)
    frm = pd.to_datetime(df["from_date"])
    to = pd.to_datetime(df["to_date"], errors="coerce")
    active = (frm <= as_of_ts) & (to.isna() | (to > as_of_ts))
    out = df.loc[active, ["symbol", "name", "sector"]].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"universe membership empty as of {as_of} — check {p.name}")
    return out


def point_in_time(df: pd.DataFrame, day: date, decision_ts: pd.Timestamp,
                  date_col: str = "date") -> pd.DataFrame:
    """The one PIT filter (H-8), used by screener and features on every
    capture-stamped input. A row is usable at `decision_ts` on `day` iff:
      - its content date is strictly before `day` (EOD files of prior days
        are safe regardless of backfill capture time), OR
      - it was captured at or before `decision_ts` (same-day live data).
    Rows without capture_ts raise — un-stamped data is not PIT-replayable."""
    if "capture_ts" not in df.columns:
        raise ValueError("point_in_time: frame lacks capture_ts — un-stamped data is not replayable")
    captured_before = pd.to_datetime(df["capture_ts"]) <= pd.Timestamp(decision_ts)
    if date_col in df.columns:
        content_prior = pd.to_datetime(df[date_col]).dt.date < day
        return df[content_prior | captured_before]
    return df[captured_before]
