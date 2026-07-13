"""Daily/Swing/Intraday stock PREDICTION & LISTING system — core utilities.

Companion to src/intraday/ (the intraday TRADING system). This package predicts
and LISTS a top-N watchlist per horizon; it does not execute orders. It reuses
the intraday data layer and modelling patterns wholesale.

One daily config (config_daily.yaml), its own cache, but the SAME clock and the
SAME point-in-time universe loader as intraday — those are config-light and must
not fork. Everything in src/daily/ imports these from here.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

# Reuse the intraday foundations verbatim — one clock, one PIT contract. These are
# config-light and must not fork. (load_universe is re-implemented below against the
# DAILY config so the two systems share the membership FILE but not the config.)
from src.intraday import (  # noqa: F401  (re-export)
    IST,
    ROOT,
    assert_clock_sane,
    now_ist,
    point_in_time,
    today_ist,
)

DAILY_CONFIG_PATH = ROOT / "config" / "config_daily.yaml"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_daily_config() -> dict:
    # Tests monkeypatch src.daily.DAILY_CONFIG_PATH and call
    # load_daily_config.cache_clear() — mirror src.intraday.load_config.
    import src.daily as _self

    with open(_self.DAILY_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def daily_path(key: str) -> Path:
    """Resolve a config path (data.* or paths.*) against ROOT. Absolute paths in
    config (used by tests) pass through unchanged."""
    cfg = load_daily_config()
    raw = cfg["data"].get(key) if key in cfg["data"] else cfg["paths"][key]
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def horizon_config(name: str) -> dict:
    """The config block for one horizon (next_day | swing_2_10d | intraday_open)."""
    horizons = load_daily_config()["horizons"]
    if name not in horizons:
        raise KeyError(f"unknown horizon {name!r}; known: {list(horizons)}")
    return horizons[name]


def enabled_horizons() -> list[str]:
    """Horizons with enabled=true, in config order."""
    return [h for h, c in load_daily_config()["horizons"].items() if c.get("enabled")]


def load_universe(as_of: date) -> pd.DataFrame:
    """Point-in-time universe from the DAILY config's membership file (same file as
    intraday in production, but read via the daily config so the two systems stay
    independent). `as_of` is REQUIRED — survivorship bias cannot sneak back in.

    Members where from_date <= as_of < to_date (empty to_date = current member).
    """
    raw = load_daily_config()["universe"]["membership_file"]
    p = Path(raw)
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — PIT universe membership is mandatory")
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
