"""Backfill the global/macro regime channel (PLAN_v4 §5).

Prior-close instruments (US/Asia/USDINR/crude/NSEI/India VIX) are content-dated
strictly before the trading day that consumes them, so backfilling them is
lookahead-free. Morning snapshots (Asia partial, GIFT gap) are NOT backfilled —
they accrue forward via `python main.py --mode regime-capture`.

Usage:
    python scripts/backfill_regime.py [--years N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intraday import load_config, today_ist  # noqa: E402
from src.intraday.regime_data import backfill  # noqa: E402


def main() -> None:
    from datetime import date
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    args = p.parse_args()

    end = date.fromisoformat(args.end) if args.end else today_ist()
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        years = args.years or load_config()["data"]["backfill_years"]
        start = end - timedelta(days=int(years * 365.25))
    n = backfill(start, end)
    print(f"regime backfill {start} → {end}: {n} rows written")


if __name__ == "__main__":
    main()
