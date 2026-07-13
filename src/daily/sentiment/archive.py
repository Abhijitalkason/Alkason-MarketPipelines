"""Append-only storage for the forward sentiment archive.

raw/<date>.parquet    : [symbol, headline, source, url, release_ts, capture_ts]
scored.parquet        : raw columns + [sentiment, score_model, scored_ts]

release_ts (first-seen IST) is the PIT key — everything downstream filters on it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.daily import daily_path, now_ist

RAW_COLUMNS = ["symbol", "headline", "source", "url", "release_ts", "capture_ts"]
SCORED_COLUMNS = RAW_COLUMNS + ["sentiment", "score_model", "scored_ts"]


def _raw_dir() -> Path:
    d = daily_path("sentiment_path") / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scored_path() -> Path:
    d = daily_path("sentiment_path")
    d.mkdir(parents=True, exist_ok=True)
    return d / "scored.parquet"


def append_raw(headlines: list[dict]) -> int:
    """Append headlines (each a dict with at least symbol+headline) for today,
    stamping release_ts/capture_ts = now if absent."""
    if not headlines:
        return 0
    now = now_ist()
    rows = []
    for h in headlines:
        rows.append({c: h.get(c) for c in RAW_COLUMNS} | {
            "release_ts": h.get("release_ts", now), "capture_ts": now})
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    path = _raw_dir() / f"{now.date().isoformat()}.parquet"
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "headline"], keep="first")
    df.to_parquet(path, index=False)
    return len(df)                                   # rows after dedup (true stored count)


def read_raw(since: date | None = None) -> pd.DataFrame:
    frames = []
    for f in sorted(_raw_dir().glob("*.parquet")):
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if since and d < since:
            continue
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)


def append_scored(df: pd.DataFrame) -> int:
    path = _scored_path()
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "headline"], keep="last")
    df.to_parquet(path, index=False)
    return len(df)


def read_scored() -> pd.DataFrame:
    p = _scored_path()
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=SCORED_COLUMNS)
