"""PLAN v3 intraday selective signal system (see PLAN_v3.md)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config_v3.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_universe() -> pd.DataFrame:
    """Universe list: columns symbol, name, sector."""
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["universe"]["file"], comment="#")
    df["symbol"] = df["symbol"].str.strip()
    return df
