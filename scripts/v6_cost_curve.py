"""PLAN_v6 §4.4 — c(size) curve + minimum-viable-size floor (F21: computed and
frozen BEFORE any geometry cell is evaluated).

Writes reports/v6/phase0_cost_curve.md (human review) and
reports/v6/cost_curve.json (consumed by the §4.3 geometry grid for the
empirical-δ columns).

Run: python -m scripts.v6_cost_curve
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.intraday import ROOT, load_config
from src.v6 import costs

SIZES_INR = [1_000, 10_000, 25_000, 50_000, 100_000, 1_000_000, 10_000_000]

OUT_DIR = ROOT / "reports" / "v6"


def build_curve() -> dict:
    cfg = load_config()["v6"]
    ref_size = cfg["position_size_inr"]
    rows = []
    for s in SIZES_INR:
        base = costs.round_trip_costs(s)
        rows.append({
            "size_inr": s,
            "cost_inr": round(base.total_inr, 2),
            "cost_pct": round(100 * base.total_inr / s, 4),
            "cost_pct_ex_slippage": round(100 * (base.total_inr - base.slippage_inr) / s, 4),
            "cost_pct_2x_stress": round(100 * costs.round_trip_cost_pct(s, stress=True), 4),
            "flat_fee_share_pct": round(100 * costs.flat_fee_inr() / base.total_inr, 1),
        })
    floor = costs.minimum_viable_size_inr()
    return {
        "generated": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "broker": "Upstox",
        "source": "upstox.com/brokerage-charges [VERIFIED 2026-07-08]",
        "reference_size_inr": ref_size,
        "curve": rows,
        "minimum_viable_size_inr": round(floor),
        "floor_rule": ("smallest size where flat fees (brokerage+DP+GST) <= "
                       "ex-slippage variable cost; frozen per F21, "
                       "slippage-independent by construction"),
        "cost_pct_at_reference": round(100 * costs.round_trip_cost_pct(ref_size), 4),
        "cost_pct_at_reference_2x": round(100 * costs.round_trip_cost_pct(ref_size, stress=True), 4),
        "slippage_status": "PROVISIONAL 5bps/side [MEASURE — unknown #15 open-print study]",
    }


def write_report(data: dict) -> Path:
    md = [
        "# Phase 0 §4.4 — Delivery cost curve c(size) — broker: Upstox",
        "",
        f"Generated: {data['generated']} · constants: {data['source']}",
        "",
        "| Size (₹) | Cost (₹) | c(size) % | ex-slippage % | 2× stress % | flat-fee share |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["curve"]:
        md.append(f"| {r['size_inr']:,} | {r['cost_inr']:,} | {r['cost_pct']} "
                  f"| {r['cost_pct_ex_slippage']} | {r['cost_pct_2x_stress']} "
                  f"| {r['flat_fee_share_pct']}% |")
    md += [
        "",
        f"**Minimum viable size (FROZEN, F21): ₹{data['minimum_viable_size_inr']:,}**",
        f"— rule: {data['floor_rule']}. Sizes below it are plumbing-test-only and",
        "excluded from GO/KILL (§4.4).",
        "",
        f"**Reference size for gate arithmetic: ₹{data['reference_size_inr']:,}** —",
        f"c = **{data['cost_pct_at_reference']}%** base, "
        f"**{data['cost_pct_at_reference_2x']}%** at 2× slippage stress.",
        "These are the `c` inputs to the §4.3 empirical-δ columns.",
        "",
        f"Slippage: {data['slippage_status']} — the floor rule excludes slippage",
        "so the measured constant cannot move the frozen floor.",
        "",
        "Per-symbol maximum size (pre-open auction participation, F18): pending",
        "auction-volume data `[MEASURE — unknown #13]`; not needed for the floor.",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cost_curve.json").write_text(json.dumps(data, indent=2))
    report = OUT_DIR / "phase0_cost_curve.md"
    report.write_text("\n".join(md) + "\n")
    return report


if __name__ == "__main__":
    path = write_report(build_curve())
    print(f"wrote {path}")
