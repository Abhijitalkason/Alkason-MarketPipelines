# Model Card — intraday-v3-signal

**Intended use:** 2–3h NSE large-cap selective trading signals; entries
09:30–11:00 IST, square-off 14:45, always flat overnight.

## Training data
- Span: 2024-12-22 → 2026-06-22
- Rows: 20878
- flow_real_share: 0.600 → flow channel **OFF (price-only)**

## Geometry & gate
- target/stop ATR: 0.4/3.2
- time barrier: 3h, square-off 14:45
- τ₀: 0.92, target win rate 0.89
- agreement floor: 0.75

## Metrics
- calibration-tail Brier: 0.2326
- blend weight (price): 1.00

## Exclusions
- per-symbol coverage exclusions: none

## Known limitations
- flow channel OFF until flow_real_share ≥ 0.6
- pre-open / L2 archive depth grows forward only
- index_alignment uses the universe-median proxy (no index bars yet)

## Provenance
- git SHA: 7351bf618ed026b6d2ba907d8cc92bcfcfc24ddb-dirty
- config hash: 4a86d7a24ef2
- DVC rev: 7351bf618ed026b6d2ba907d8cc92bcfcfc24ddb
