"""PIT sentiment features (Phase 5). Reads the scored archive filtered to
release_ts <= decision_ts, so it can only ever see news public by the decision —
the no-lookahead property test applies here exactly as to every other block.

Returns the same dict shape features._sentiment_row expects (values + _present).
Until the archive has accrued, every row is naturally _present=0.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.daily.sentiment.archive import read_scored


def sentiment_features(symbol: str, day: date, decision_ts: pd.Timestamp, out: dict) -> dict:
    scored = read_scored()
    if scored.empty or "release_ts" not in scored.columns:
        return out
    df = scored[(scored["symbol"] == symbol) &
                (pd.to_datetime(scored["release_ts"]) <= decision_ts)].copy()
    if df.empty:
        return out
    df["release_ts"] = pd.to_datetime(df["release_ts"])
    recent = df[df["release_ts"] >= decision_ts - pd.Timedelta(days=5)]
    if not recent.empty:
        out["sent_mean_5d"] = float(recent["sentiment"].mean()); out["sent_mean_5d_present"] = 1.0
        out["sent_coverage"] = float(len(recent)); out["sent_coverage_present"] = 1.0
    if len(df) >= 10:
        base = float(df["sentiment"].mean())
        last = float(df.sort_values("release_ts")["sentiment"].tail(3).mean())
        out["sent_surprise"] = last - base; out["sent_surprise_present"] = 1.0
    return out
