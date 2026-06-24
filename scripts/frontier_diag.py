"""Diagnostic ONLY (not part of the pipeline): does the model have ANY edge?

Reuses the real walk-forward machinery (build_dataset → make_folds → fit_fold)
to collect out-of-sample calibrated predictions, then:
  - AUC(label, p_cal): rank quality. ~0.50 = no edge; >0.55 = some signal.
  - frontier: at a sweep of fire thresholds, the coverage / win-rate / expectancy
    that WOULD result — bypassing the live gate that raises on 0 trades.
Flat 0.10% round-trip cost is used for a quick net-expectancy estimate.
Caches the assembled dataset to data/processed/v3_dataset.parquet for reuse.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import numpy as np
import pandas as pd

from src.intraday import ROOT
from src.intraday.backtester import build_dataset, fit_fold, make_folds, trading_sessions

FLAT_COST_PCT = 0.10  # round-trip estimate (PLAN_v3 §12.3: 0.08–0.11%)

cache = ROOT / "data" / "processed" / "v3_dataset.parquet"
if cache.exists():
    print(f"loading cached dataset {cache}")
    data = pd.read_parquet(cache)
    data["date"] = pd.to_datetime(data["date"]).dt.date
else:
    days = trading_sessions(date(2023, 6, 24), date(2026, 6, 23))
    print(f"assembling dataset over {len(days)} trading days …")
    data = build_dataset(days)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    cache.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache, index=False)
    print(f"cached dataset → {cache}  ({len(data)} rows)")

folds = make_folds(sorted(data["date"].unique()))
print(f"{len(folds)} walk-forward folds; base win rate {data['label'].mean():.3f}\n")

oof = []
for k, (train_days, test_days) in enumerate(folds, 1):
    train_df = data[data.date.isin(train_days)]
    pm, fm, blender, record = fit_fold(train_df, tune=False)
    test_df = data[data.date.isin(test_days)].copy()
    if test_df.empty:
        continue
    p_p = pm.predict_proba(test_df)
    p_f = fm.predict_proba(test_df) if fm is not None else None
    p_cal = blender.calibrated(p_p, p_f)
    test_df["p_cal"] = p_cal
    oof.append(test_df[["date", "symbol", "p_cal", "label", "pnl_pct"]])
    print(f"fold {k}: {len(test_df)} OOF rows")

oof = pd.concat(oof, ignore_index=True)
n = len(oof)
y = oof["label"].to_numpy()
p = oof["p_cal"].to_numpy()

# AUC via Mann–Whitney (no sklearn dep needed)
order = np.argsort(p)
ranks = np.empty(n, float)
ranks[order] = np.arange(1, n + 1)
n_pos, n_neg = y.sum(), (1 - y).sum()
auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else float("nan")

print(f"\n=== OOF: {n} rows, base win rate {y.mean():.3f}, AUC = {auc:.4f} ===")
print("(AUC 0.50 = no edge; 0.55 = weak; 0.60+ = usable)\n")

print(f"{'thresh':>7} {'coverage':>9} {'n':>6} {'win_rate':>9} {'gross_exp%':>11} {'net_exp%':>9}")
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92]:
    sel = oof[oof["p_cal"] >= tau]
    if sel.empty:
        print(f"{tau:>7.2f} {0.0:>9.3f} {0:>6} {'—':>9} {'—':>11} {'—':>9}")
        continue
    wr = sel["label"].mean()
    gross = sel["pnl_pct"].mean()
    print(f"{tau:>7.2f} {len(sel)/n:>9.3f} {len(sel):>6} {wr:>9.3f} "
          f"{gross:>11.4f} {gross - FLAT_COST_PCT:>9.4f}")
print(f"\nfire threshold in config (gate.fire_threshold_init) = 0.92")
