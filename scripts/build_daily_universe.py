"""Generate data/reference/universe_top100.csv — the PIT membership file the
daily system's load_universe() reads — from the v6 monthly liquidity universe.

Bridges src/v6 (full-market panel + monthly top-N by trailing 6-month median
turnover, survivorship-free) into the src/daily membership-file contract:
one row per (symbol, month) with [symbol, name, sector, from_date, to_date],
where from_date = month start and to_date = next month start. Rows of the
LATEST month get an empty to_date (current members) so a decision day in the
current month resolves before the next monthly regeneration.

Usage:
    python scripts/build_daily_universe.py [--no-rebuild-panel]

--no-rebuild-panel reuses the cached v6 panel (data/processed/v6_panel.parquet);
by default the panel is rebuilt so the newest bhavcopy days are included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intraday import ROOT  # noqa: E402
from src.v6.panel import load_panel  # noqa: E402
from src.v6.universe import monthly_pit_universe  # noqa: E402

OUT = ROOT / "data" / "reference" / "universe_top100.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-rebuild-panel", action="store_true",
                   help="reuse the cached v6 panel instead of rebuilding from bhavcopy")
    args = p.parse_args()

    panel = load_panel(use_cache=args.no_rebuild_panel)
    uni = monthly_pit_universe(panel)

    last_month = uni["month"].max()
    rows = []
    for _, r in uni.iterrows():
        m_start = r["month"].to_timestamp().date()
        m_next = (r["month"] + 1).to_timestamp().date()
        rows.append({
            "symbol": r["symbol"],
            "name": r["symbol"],
            "sector": "UNKNOWN",
            "from_date": m_start.isoformat(),
            # empty to_date = current member (covers days past month-end until
            # the next regeneration run)
            "to_date": "" if r["month"] == last_month else m_next.isoformat(),
        })
    out = pd.DataFrame(rows, columns=["symbol", "name", "sector", "from_date", "to_date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# PIT top-100 liquidity universe (v6 monthly_pit_universe).\n"
        "# One row per (symbol, month): from_date <= as_of < to_date; empty to_date = current.\n"
        "# Regenerate monthly: python scripts/build_daily_universe.py\n"
    )
    with open(OUT, "w", newline="") as f:
        f.write(header)
        out.to_csv(f, index=False)
    print(f"universe_top100.csv: {len(out)} rows, "
          f"{out['symbol'].nunique()} unique symbols, "
          f"months {uni['month'].min()}..{last_month} -> {OUT}")


if __name__ == "__main__":
    main()
