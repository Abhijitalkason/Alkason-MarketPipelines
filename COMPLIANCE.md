# COMPLIANCE — SEBI algo posture (v3)

This system is a research/paper-trading pipeline. **No live order may be placed
before the broker-side algo approval below exists.** Gate 5 (capital) cannot be
entered until every item here is satisfied.

## Broker algo approval
- SEBI's algorithmic-trading framework requires broker/exchange approval and a
  unique algo ID for any automated order flow. Live order placement is gated on
  this approval; until then stops are software-simulated in paper only.

## Exchange order tagging
- Every live order must carry the exchange-assigned algo ID in the order tag,
  per the broker's algo-order spec. The order-placement layer (added only after
  Gate 4) must populate this field; orders without it must be rejected locally.

## Audit-log retention → the journal
- `reports/v3/journal/<date>.jsonl` is the hash-chained audit substrate: screen,
  gate evaluations, fills, exits, ACI updates, halts, promotions. It is
  append-only and chain-verifiable (`journal_verify`). Retain per the broker's
  audit-retention requirement (typically ≥5 years).

## Leverage cap
- MIS intraday leverage is capped at **3× notional** in config (`risk.max_leverage`),
  below the broker's ~5× allowance. Position sizing enforces it.

## Stop discipline
- Live stops must be **broker-side SL-M** orders, never software-only. This is a
  Gate-5 prerequisite; the order layer placing SL-M orders is implemented only
  after Gate 4 passes.

## Gate-5 checklist (capital deployment)
- [ ] Broker algo approval + algo ID obtained
- [ ] Order layer tags every order with the algo ID; untagged orders rejected
- [ ] Broker-side SL-M stops placed for every position
- [ ] Leverage cap (3×) enforced and verified
- [ ] Kill switches demonstrated (forced-breach drill journaled)
- [ ] Rollback rehearsed and journaled
- [ ] Smallest viable size, 1 month, kill switches never breached, expectancy positive
- [ ] Journal retention configured per broker requirement
