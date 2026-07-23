"""PLAN_v6 §4.6 — GDELT historical-coverage probe (sampled design, F14).

Decides whether news enters the Gate-3 backtest or stays forward-only.
Sampled: 8 symbols × 12 random weeks per quarter, 2023Q3→2026Q2, at
1 request / 5 s (GDELT-compliant), plus a market-level lane ("Nifty OR
Sensex" etc.) sampled 2 weeks/quarter. Coverage is reported SEPARATELY for
market-level and symbol-level (the 30% symbol-day threshold applies to
symbol-level only).

Deterministic sampling (seed 42) and resumable: results append to a JSONL;
already-done (query, week) pairs are skipped on restart.

Run (background, ~1.7 h): python -m scripts.v6_gdelt_probe
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.intraday import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("v6.gdelt")

API = "https://api.gdeltproject.org/api/v2/doc/doc"
SLEEP_S = 5
SEED = 42
N_SYMBOLS = 8
WEEKS_PER_QUARTER = 12
OUT = ROOT / "reports" / "v6" / "gdelt_probe.jsonl"
SUMMARY = ROOT / "reports" / "v6" / "gdelt_probe_summary.json"

MARKET_QUERY = '("Nifty 50" OR Sensex OR "Indian stock market")'


def _quarters() -> list[pd.Period]:
    return list(pd.period_range("2023Q3", "2026Q2", freq="Q"))


def _sample_weeks(q: pd.Period, k: int, rng: random.Random) -> list[pd.Timestamp]:
    mondays = pd.date_range(q.start_time, q.end_time, freq="W-MON")
    return sorted(rng.sample(list(mondays), min(k, len(mondays))))


def _sample_symbols() -> list[tuple[str, str]]:
    """(symbol, company name) — universe symbols that have a name on disk."""
    uni = pd.read_parquet(ROOT / "data" / "processed" / "v6_universe.parquet")
    names = pd.read_csv(ROOT / "data" / "reference" / "universe.csv", comment="#")
    named = names[names["symbol"].isin(set(uni["symbol"]))]
    rng = random.Random(SEED)
    picks = rng.sample(list(named.itertuples(index=False)), N_SYMBOLS)
    return [(p.symbol, p.name) for p in picks]


def _fetch_count(query: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    params = {
        "query": f'{query} sourcelang:eng',
        "mode": "artlist", "format": "json", "maxrecords": 250,
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    arts = r.json().get("articles", [])
    days = {a["seendate"][:8] for a in arts if "seendate" in a}
    return {"n_articles": len(arts), "n_days_with_articles": len(days)}


def main() -> None:
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            rec = json.loads(line)
            done.add((rec["lane"], rec["key"], rec["week"]))
    rng = random.Random(SEED)
    symbols = _sample_symbols()
    logger.info("sampled symbols: %s", [s for s, _ in symbols])

    tasks = []
    for q in _quarters():
        for wk in _sample_weeks(q, WEEKS_PER_QUARTER, rng):
            for sym, name in symbols:
                tasks.append(("symbol", sym, f'"{name}"', wk))
        for wk in _sample_weeks(q, 2, rng):
            tasks.append(("market", "MARKET", MARKET_QUERY, wk))

    logger.info("%d probe tasks (%d already done)", len(tasks), len(done))
    with OUT.open("a") as f:
        for lane, key, query, wk in tasks:
            wk_s = wk.strftime("%Y-%m-%d")
            if (lane, key, wk_s) in done:
                continue
            try:
                res = _fetch_count(query, wk, wk + pd.Timedelta(days=6))
            except Exception as e:  # transient failures must not kill the probe
                logger.warning("%s %s %s failed: %s", lane, key, wk_s, e)
                res = {"n_articles": -1, "n_days_with_articles": -1}
            f.write(json.dumps({"lane": lane, "key": key, "week": wk_s, **res}) + "\n")
            f.flush()
            time.sleep(SLEEP_S)
    summarize()


def summarize() -> None:
    recs = [json.loads(x) for x in OUT.read_text().splitlines()]
    df = pd.DataFrame([r for r in recs if r["n_articles"] >= 0])
    sym = df[df["lane"] == "symbol"]
    mkt = df[df["lane"] == "market"]
    summary = {
        "generated": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "symbol_level": {
            "weeks_probed": int(len(sym)),
            "weeks_with_any_article": round(float((sym["n_articles"] > 0).mean()), 4),
            "symbol_day_coverage": round(float(sym["n_days_with_articles"].sum()
                                               / (len(sym) * 7)), 4),
            "per_symbol_week_coverage": {
                k: round(float((g["n_articles"] > 0).mean()), 3)
                for k, g in sym.groupby("key")},
        },
        "market_level": {
            "weeks_probed": int(len(mkt)),
            "weeks_with_any_article": round(float((mkt["n_articles"] > 0).mean()), 4),
            "day_coverage": round(float(mkt["n_days_with_articles"].sum())
                                  / max(len(mkt) * 7, 1), 4),
        },
        "failed_requests": int((pd.DataFrame(recs)["n_articles"] < 0).sum()),
        "verdict_rule": "symbol-day coverage >= 0.30 => news enters Gate-3 backtest; "
                        "else forward-only",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    logger.info("summary written: %s", SUMMARY)


if __name__ == "__main__":
    main()
