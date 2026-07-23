"""Per-pick price chart for the daily list (the user's 'share a graph').

Trailing daily closes + 20/50-DMA, the predicted direction marker, and the
calibrated probability in the title. PNG → reports/daily/charts/<date>/. Pure
rendering; no model logic. matplotlib is optional (Agg) — absence degrades to a
skipped chart, never an error.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.daily import daily_path
from src.daily.panel import load_symbol_daily

logger = logging.getLogger(__name__)


def render_pick(symbol: str, day: date, direction: str, prob: float, horizon: str,
                lookback: int = 120, df_panel: pd.DataFrame | None = None) -> str | None:
    """Render one pick's chart; returns the PNG path (or None if rendering skipped)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        logger.warning("matplotlib unavailable, skipping chart: %s", e)
        return None

    df = load_symbol_daily(symbol, end=day, adjusted=True, df_panel=df_panel)
    df = df[df.index.date <= day].tail(lookback)
    if df.empty:
        return None
    close = df["close"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, close, color="black", lw=1.2, label="close")
    if len(close) >= 20:
        ax.plot(df.index, close.rolling(20).mean(), color="tab:blue", lw=0.9, label="20-DMA")
    if len(close) >= 50:
        ax.plot(df.index, close.rolling(50).mean(), color="tab:orange", lw=0.9, label="50-DMA")
    arrow = "▲" if direction.upper() == "LONG" else "▼"
    color = "tab:green" if direction.upper() == "LONG" else "tab:red"
    ax.scatter([df.index[-1]], [close.iloc[-1]], color=color, s=60, zorder=5)
    ax.set_title(f"{symbol} — {horizon} {arrow} {direction}  P(up)={prob:.2f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()

    out_dir = daily_path("charts") / day.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{horizon}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return str(path)
