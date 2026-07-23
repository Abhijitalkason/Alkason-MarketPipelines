#!/bin/zsh
# Daily top-pick runner — run every trading evening after NSE publishes the
# bhavcopy (~18:30 IST). No broker token needed: EOD public data only.
#
# Since 2026-07-17 this is a thin wrapper around the OBSERVABLE pipeline mode:
#   python main.py --mode daily-pipeline
# which does fetch → global → panel → picks → notify, streams step-by-step
# status to reports/daily/pipeline_status.json (Pipeline panel in the UI at
# http://localhost:8001/ui, or GET /pipeline/status), and automatically falls
# back to the latest session on disk when today's data isn't published yet.
#
# Cron (install with `crontab -e`; tag makes it easy to find/remove):
#   30 19 * * 1-5  cd <repo> && ./scripts/daily_pick.sh >> reports/daily/pick_cron.log 2>&1  # alkason-daily-pick

set -uo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python main.py --mode daily-pipeline
