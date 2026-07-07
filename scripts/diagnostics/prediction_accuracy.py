"""Direct prediction-accuracy diagnostic (buy/sell skill), out-of-sample.

RETIRED AS A DECISION METRIC (PLAN_v4 §9). Full-coverage accuracy at p≥0.5 over
ALL OOF rows is NOT the system's contract — a selective signal system is designed
to look near-random on the rows it would never trade, so ~0.50 here is EXPECTED,
not failure (this exact number was the misread that motivated PLAN_v4). The
contract metric is fired-signal win rate + post-cost expectancy from
src/intraday/backtester.py (run `python main.py --mode backtest`). Kept for
diagnosis only (AUC / rank skill / precision-at-top-k), under scripts/diagnostics/
so it can never be mistaken for an acceptance gate.

Reuses the cached dataset + the real walk-forward machinery, collects OOF
calibrated predictions, and reports the metrics that actually measure
prediction quality (not geometry-inflated win rate):
  - accuracy of the win/loss call vs the majority-class baseline
  - confusion matrix, precision, recall
  - AUC (rank skill)
  - precision (win rate) at the live fire threshold and at top-k% confidence
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.intraday import ROOT  # noqa: E402
from src.intraday.backtester import build_dataset, fit_fold, make_folds, trading_sessions  # noqa: E402

cache = ROOT / "data" / "processed" / "v3_dataset.parquet"
if cache.exists():
    data = pd.read_parquet(cache)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    print(f"loaded cached dataset ({len(data)} rows)")
else:
    data = build_dataset(trading_sessions(date(2023, 6, 24), date(2026, 6, 23)))
    data["date"] = pd.to_datetime(data["date"]).dt.date

folds = make_folds(sorted(data["date"].unique()))
oof = []
for k, (tr, te) in enumerate(folds, 1):
    pm, fm, blender, rec = fit_fold(data[data.date.isin(tr)], tune=False)
    td = data[data.date.isin(te)].copy()
    if td.empty:
        continue
    pp = pm.predict_proba(td)
    pf = fm.predict_proba(td) if fm is not None else None
    td["p_cal"] = blender.calibrated(pp, pf)
    oof.append(td[["p_cal", "label", "pnl_pct"]])
    print(f"fold {k}: {len(td)} OOF rows")

oof = pd.concat(oof, ignore_index=True)
y = oof["label"].to_numpy().astype(int)
p = oof["p_cal"].to_numpy()
n = len(y)
base = y.mean()                       # share of wins (majority class)
pred = (p >= 0.5).astype(int)         # model's win/loss call

# AUC (Mann–Whitney)
order = np.argsort(p); ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
npos, nneg = y.sum(), n - y.sum()
auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
acc = (tp + tn) / n
majority_acc = max(base, 1 - base)    # trivial "always predict the common class"
prec = tp / (tp + fp) if tp + fp else float("nan")
rec = tp / (tp + fn) if tp + fn else float("nan")

print(f"\n=== PREDICTION ACCURACY (out-of-sample, n={n}) ===")
print(f"base win rate (majority class) : {base:.3f}")
print(f"AUC (rank skill, 0.50=random)  : {auc:.4f}")
print(f"\nmodel win/loss call @ p>=0.50:")
print(f"  accuracy                     : {acc:.3f}")
print(f"  majority-baseline accuracy   : {majority_acc:.3f}   <- must beat this to show skill")
print(f"  lift over baseline           : {acc - majority_acc:+.3f}")
print(f"  precision (P win | predict win): {prec:.3f}")
print(f"  recall                       : {rec:.3f}")
print(f"  confusion matrix  TP={tp} FP={fp}  FN={fn} TN={tn}")

print(f"\nprecision (win rate) when the model is MOST confident:")
for q in (0.10, 0.05, 0.02, 0.01):
    k_ = max(1, int(n * q))
    top = oof.nlargest(k_, "p_cal")
    print(f"  top {q:>4.0%} confident (n={k_:>4}): win rate {top['label'].mean():.3f}")
print(f"\n(majority baseline {majority_acc:.3f}; random AUC 0.50 — interpret skill against these)")
