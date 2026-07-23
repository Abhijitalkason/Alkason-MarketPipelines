"""On-demand single-stock deep analysis (feature request 2026-07-16).

Give it one NSE symbol → it assembles every data source this repo can reach
(price history, technicals, delivery/F&O flow, fundamentals, news with links,
market regime, the daily model's calibrated view) into a TRANSPARENT evidence
scorecard and a BUY / HOLD / SELL verdict with end-to-end reasoning.

Honesty contract (same as the rest of the repo): every component reports its
value, weight, score and a one-line reason; missing data is disclosed via
`present` flags, never faked; and the verdict carries the standing disclaimer
that this is rule-based evidence aggregation — the measured record
(reports/daily/run_registry.csv) shows no proven predictive edge, so treat the
output as a research aid, not investment advice.
"""
