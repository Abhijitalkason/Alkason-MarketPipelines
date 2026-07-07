#!/bin/sh
# Pre-open capture: global/macro regime snapshot + news headlines (PLAN_v4 §5/§8).
# Cron-friendly: one line in crontab runs both; each capture degrades to a clean
# no-op if its dependency (yfinance / feedparser+transformers) or network is
# absent, so a partial failure never blocks the other channel.
# See RUNBOOK.md "Pre-open regime & news capture" for the crontab entries.
set -u
cd "$(dirname "$0")/.."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] preopen_capture start"
python main.py --mode regime-capture || echo "regime-capture failed (continuing)"
python main.py --mode news-capture   || echo "news-capture failed (continuing)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] preopen_capture done"
