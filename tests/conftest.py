"""Shared pytest fixtures — deterministic synthetic data, no network, no live
feed (v3_supplementry §18). Builds a throwaway repo layout under tmp_path,
points config_v3.yaml's data/paths at it, and clears the config cache.
"""

from __future__ import annotations

import textwrap
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = yaml.safe_load((ROOT / "config" / "config_v3.yaml").read_text())


def _trading_days(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def make_session_bars(symbol: str, day: date, open_px: float, trend: float = 0.0,
                      vol: float = 0.001, base_volume: int = 8000, seed: int = 0) -> pd.DataFrame:
    """One trading session of synthetic 1-min bars (09:15–15:29)."""
    rng = np.random.RandomState(seed)
    minutes = []
    t = datetime.combine(day, time(9, 15))
    end = datetime.combine(day, time(15, 30))
    px = open_px
    while t < end:
        minutes.append(t)
        t += timedelta(minutes=1)
    n = len(minutes)
    rets = rng.normal(trend, vol, n)
    closes = open_px * np.cumprod(1 + rets)
    highs = closes * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lows = closes * (1 - np.abs(rng.normal(0, vol / 2, n)))
    opens = np.concatenate([[open_px], closes[:-1]])
    vols = (base_volume * (1 + np.abs(rng.normal(0, 0.3, n)))).astype(int)
    return pd.DataFrame({
        "ts": minutes, "open": opens, "high": np.maximum.reduce([opens, highs, closes]),
        "low": np.minimum.reduce([opens, lows, closes]), "close": closes,
        "volume": vols, "capture_ts": [datetime.combine(day, time(15, 40))] * n,
        "provisional": [False] * n,
    })


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway repo: synthetic bars for a few symbols over ~26 months, a PIT
    universe, holidays, events, and a config pointed at tmp_path."""
    import src.intraday as si

    data = tmp_path / "data"
    ref = data / "reference"
    ref.mkdir(parents=True)
    for sub in ("bars", "bhavcopy", "preopen", "depth"):
        (data / sub).mkdir()
    models = tmp_path / "models" / "v3"
    models.mkdir(parents=True)
    reports = tmp_path / "reports" / "v3"
    for sub in ("", "signals", "journal", "alerts", "gates", "drift"):
        (reports / sub).mkdir(parents=True, exist_ok=True)

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    sectors = ["Tech", "Bank", "Tech", "Energy"]
    ref.joinpath("universe_membership.csv").write_text(
        "symbol,name,sector,from_date,to_date\n"
        + "\n".join(f"{s},{s} Ltd,{sec},2019-01-01," for s, sec in zip(symbols, sectors)) + "\n"
    )
    ref.joinpath("nse_holidays.csv").write_text("date,description\n2026-01-26,Republic Day\n")
    ref.joinpath("events.csv").write_text("date,type,symbol\n2099-01-01,budget,\n")
    ref.joinpath("corporate_actions.csv").write_text("symbol,ex_date,purpose,factor\n")

    start, end = date(2024, 1, 1), date(2026, 3, 31)
    days = _trading_days(start, end)
    for i, sym in enumerate(symbols):
        frames = []
        base = 1000 + i * 250
        for j, d in enumerate(days):
            if d.weekday() == 3:   # leave Thursdays (expiry blackout) sparse but present
                pass
            trend = 0.00005 * (1 if (j + i) % 2 == 0 else -1)
            frames.append(make_session_bars(sym, d, base * (1 + 0.0002 * j), trend=trend,
                                            seed=(i * 1000 + j)))
        df = pd.concat(frames, ignore_index=True)
        sd = data / "bars" / sym
        sd.mkdir(parents=True)
        for period, chunk in df.groupby(df["ts"].dt.to_period("M")):
            chunk.to_parquet(sd / f"{period}.parquet", index=False)

    cfg = yaml.safe_load(textwrap.dedent(yaml.safe_dump(BASE_CONFIG)))
    cfg["data"]["bars_path"] = str(data / "bars")
    cfg["data"]["bhavcopy_path"] = str(data / "bhavcopy")
    cfg["data"]["preopen_path"] = str(data / "preopen")
    cfg["data"]["depth_path"] = str(data / "depth")
    cfg["data"]["reference_path"] = str(ref)
    cfg["universe"]["membership_file"] = str(ref / "universe_membership.csv")
    cfg["universe"]["min_median_1min_volume"] = 1000
    cfg["training"]["tune"] = False
    cfg["training"]["min_folds"] = 1
    cfg["training"]["train_min_months"] = 12
    cfg["gate"]["flow_model_active"] = False
    cfg["paths"] = {k: str((tmp_path / v)) for k, v in BASE_CONFIG["paths"].items()}
    cfg["paths"]["state"] = str(models)

    cfg_path = tmp_path / "config_v3.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    monkeypatch.setattr(si, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(si, "ROOT", tmp_path)
    si.load_config.cache_clear()
    yield {"root": tmp_path, "symbols": symbols, "days": days, "data": data, "cfg": cfg}
    si.load_config.cache_clear()


@pytest.fixture
def frozen_day(repo):
    """A representative non-blackout (non-Thursday) trading day with ≥20 prior."""
    for d in repo["days"][40:]:
        if d.weekday() != 3:
            return d
    return repo["days"][40]
