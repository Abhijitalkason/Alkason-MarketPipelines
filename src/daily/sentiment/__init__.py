"""Track B — forward-built, timestamped news/sentiment archive.

Fully isolated from the factor-based v1 (Milestone 1/2 import nothing from here).
Historical news backfill is forbidden — published-historical text is lookahead
(v1 died of exactly this), so this archive can only accrue FORWARD: every headline
is stored with release_ts = first-seen IST. It folds into the model (Phase 5) only
once enough has accumulated to retest under the staged gate.
"""
