# RUNBOOK — Intraday Selective Signal System (v4)

Daily operations for the NSE intraday selective-signal system. All times IST.
v4 additions: the pre-open regime/news captures below (PLAN_v4 §5/§8).

## Pre-open regime & news capture (daily, 08:00–09:10)

The regime channel's forward-only features (same-morning Asia snapshot, GIFT gap,
news sentiment) only exist on days the capture ran — schedule it, don't rely on
memory. One helper runs everything:

```bash
scripts/preopen_capture.sh          # regime-capture + news-capture, logged
```

Crontab (`crontab -e`, host in IST; adjust the path):

```cron
# pre-open news sweep (overnight headlines) + regime snapshot
0  8 * * 1-5  cd /path/to/AI-MLOps-Solution && ./scripts/preopen_capture.sh >> reports/v3/capture_cron.log 2>&1
# second pull just before the screen window (late headlines + fresh Asia levels)
10 9 * * 1-5  cd /path/to/AI-MLOps-Solution && ./scripts/preopen_capture.sh >> reports/v3/capture_cron.log 2>&1
```

News scoring needs `pip install feedparser transformers torch` (FinBERT downloads
on first use). Without them the capture degrades to a clean no-op — features stay
`present=0`, which the missingness policy handles by design. Backfill of the
global/macro channel (one-time / after gaps): `python main.py --mode regime-backfill`.

## Daily token refresh (before 08:55)
1. The Upstox access token expires daily. Complete the OAuth login at
   developer.upstox.com and put the new token in `.env` as `UPSTOX_ACCESS_TOKEN`.
2. The recorder/paper runner issue an **auth ping at 08:55** (`feed.auth_ping()`).
   A stale token raises *before* market open — fix it then, never at the first poll.

## Session start / stop
- **Record only:** `python main.py --mode record`
- **Paper trading (Gate 4):** `python main.py --mode paper --capital 1000000`
- Both run the full live loop; the paper runner additionally screens, gates,
  fills, manages positions, squares off at 14:45, and runs post-market checks.
- Stop: Ctrl-C. State is journaled and the session state file persists.

## Restart / resume mid-session
- Positions, closed trades, risk `DayState`, and the recorder tick buffer all
  rehydrate from `models/v3/session_state_<date>.json` and the spill dir
  `data/bars/.session/<date>/`. Re-run the same command; it resumes the day.

## Halt review and clearance
- A breach writes `models/v3/halt.json` and journals a `halt` event. The next
  session **refuses to start** until cleared.
- Review the breach (`models/v3/halt.json` → `breaches`), investigate, then:
  `python -c "from src.intraday.risk import clear_halt; clear_halt()"`.
- Kinds: `daily_loss_limit` (intra-day), `rolling_20d_expectancy_negative`,
  `live_win_rate_10pts_below_backtest`, `drift` (2 consecutive breaches).

## Promotion / rollback (approval gate)
- Train: `python main.py --mode train-v3` → registers a **Staging** version.
- Promote: `python main.py --mode promote --version N --confirm` — requires a
  passing Gate-3 report and ≥22 paper sessions on disk, else it refuses.
- Rollback: `python main.py --mode rollback --confirm` — re-stages the prior
  Production version. Both are journaled.

## Drift-alert response
- Post-market drift runs automatically (paper runner) or via
  `python main.py --mode drift`.
- A breach sets `models/v3/recalibrate_next_session.flag`, writes an alert under
  `reports/v3/alerts/`, and on 2 consecutive breaches halts.
- Response: run `python main.py --mode recalibrate` (refits isotonic on the
  trailing 60 days of matured signals), review the drift HTML under
  `reports/v3/drift/`, then clear the halt if satisfied.

## Holiday-calendar maintenance (yearly)
- `data/reference/nse_holidays.csv` must list every NSE trading holiday for the
  backfill year. Without it `--mode bhavcopy` cannot distinguish a holiday from
  a data hole and refuses to run. Update from the annual NSE holiday list.

## Weekly cadence
- Retrain check: `python main.py --mode train-v3` weekly post-market.
- Recalibration: `python main.py --mode recalibrate` weekly.
- Gates: `python main.py --mode gates --start S --end E` to refresh evidence.
