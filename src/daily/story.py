"""Pipeline STORY builder (feature request 2026-07-17) — turns the run's trace
and status into a numbered, human-readable narrative:

    Step 1  Stock selection  — source, and WHY those stocks
    Step 2  NSE EOD fetch    — which API, what it returned (or the exact error)
    ...
    Step N  Ranking          — WHY the top pick won (derived from real numbers)
    Conclusion               — verdict + what it means + what's next

Each step: {n, title, source, api, why, result, status, details[]}. Composed
server-side so the logic is unit-testable; rendered as the default tab of
/ui/trace. The "why the top pick" narrative is honest: when every calibrated
P(win) ties (the measured no-edge signature), the story says so explicitly and
explains the momentum tie-break with the actual 20-day returns.
"""

from __future__ import annotations

from pathlib import Path

from src.daily.status import read_status
from src.daily.trace import read_trace


def _step(n, title, *, source=None, api=None, why=None, result=None,
          status="ok", details=None):
    return {"n": n, "title": title, "source": source, "api": api, "why": why,
            "result": result, "status": status, "details": details or []}


def build_story(status_dir: Path | None = None) -> dict:
    st = read_status(status_dir)
    ev = read_trace(status_dir=status_dir)
    if not st and not ev:
        return {"state": "never_run", "steps": [], "conclusion": None}

    by: dict[str, list[dict]] = {}
    for e in ev:
        by.setdefault(e["step"], []).append(e)

    steps, n = [], 0

    # 1 — stock selection
    n += 1
    u = (by.get("universe") or [{}])[0]
    steps.append(_step(
        n, "Stock selection — which stocks enter the run",
        source=u.get("source", "data/reference/universe_top100.csv"),
        why=u.get("rule", "PIT top-100 liquidity universe"),
        result=(f"{u.get('n', '?')} stocks selected as of {u.get('as_of', '?')}"
                if u else "not traced in this run"),
        status=u.get("status", "skip")))

    # 2 — NSE EOD fetch
    n += 1
    b = (by.get("fetch_bhavcopy") or [{}])[0]
    steps.append(_step(
        n, "Fetch NSE end-of-day data (bhavcopy)",
        api=b.get("source", "nsearchives.nseindia.com"),
        result=(b.get("detail") if b.get("status") == "ok"
                else f"not available — {b.get('error', 'unknown')[:220]}"),
        status=b.get("status", "skip"),
        details=[f"requested session: {b.get('item', '?')}"]))

    # 3 — global/macro fetch
    n += 1
    g = by.get("fetch_global", [])
    ok_g = [e for e in g if e["status"] == "ok"]
    steps.append(_step(
        n, "Fetch global/macro context series",
        api="Yahoo Finance (yfinance)",
        why="the model's regime features: US/Asia tone, USD/INR, crude, India VIX",
        result=f"{len(ok_g)}/{len(g)} series refreshed" if g else "not traced",
        status="ok" if ok_g else "skip",
        details=[f"{e['item']}: {e.get('source', '?')} → {e.get('rows', 0)} rows"
                 for e in g]))

    # 4 — panel rebuild
    n += 1
    p = by.get("panel", [])
    ok_p = [e for e in p if e["status"] == "ok"]
    steps.append(_step(
        n, "Rebuild per-stock daily price panel",
        source="local bhavcopy store → data/daily/<SYMBOL>.parquet",
        result=f"{len(ok_p)} symbols rebuilt "
               f"({sum(e.get('rows', 0) for e in ok_p):,} rows total)" if p else "not traced",
        status="ok" if ok_p else "skip"))

    # 5 — indicators + model scoring
    n += 1
    s = by.get("score", [])
    ok_s = [e for e in s if e["status"] == "ok"]
    sk_s = [e for e in s if e["status"] == "skip"]
    probs = sorted({e["p_win"] for e in ok_s if "p_win" in e})
    n_ind = len((ok_s[0].get("indicators") or {})) if ok_s else 0
    score_details = [f"skipped {e['item']}: {e.get('reason', '?')}" for e in sk_s[:10]]
    if probs:
        score_details.insert(0, f"calibrated P(win) range: {probs[0]:.4f} → {probs[-1]:.4f} "
                                f"({len(probs)} distinct value(s) across {len(ok_s)} stocks)")
    steps.append(_step(
        n, "Compute indicators & score every stock",
        source=f"~40-feature schema per stock ({n_ind} key indicators traced: "
               "returns 1–20d, RSI, ATR, volume ratio, delivery z, OI, FII, global…)",
        api="LightGBM model 'swing_1_5d' + isotonic calibration (local, no network)",
        result=f"{len(ok_s)} stocks scored, {len(sk_s)} skipped",
        status="ok" if ok_s else "skip",
        details=score_details))

    # 6 — ranking: WHY the top pick won, from real numbers
    n += 1
    r = sorted(by.get("rank", []), key=lambda e: e.get("rank", 99))
    why_rank = None
    if r and ok_s:
        top = r[0]
        tied = len(probs) == 1
        if tied:
            r20 = {e["item"]: (e.get("indicators") or {}).get("ret_20d")
                   for e in ok_s}
            top_r20 = r20.get(top["item"])
            why_rank = (f"every stock received the SAME calibrated P(win)={probs[0]:.4f} — "
                        f"the model cannot distinguish them (its measured no-edge record). "
                        f"Ranking therefore fell to the transparent momentum tie-break: "
                        f"{top['item']} has the strongest trailing 20-day return"
                        + (f" ({top_r20:+.1%})" if top_r20 is not None else "") + ".")
        else:
            why_rank = (f"{top['item']} had the highest calibrated P(win)={top['prob']:.4f} "
                        f"of the scored universe.")
    steps.append(_step(
        n, "Rank & select the top pick",
        why=why_rank or "no picks produced",
        source="rank = calibrated P(win) desc, ties broken by 20-day then 5-day momentum",
        result=(f"top pick: {r[0]['item']} (P={r[0]['prob']:.4f}, "
                f"target {r[0].get('target')}, stop {r[0].get('stop')})" if r else "none"),
        status="ok" if r else "skip",
        details=[f"#{e['rank']} {e['item']}  P={e['prob']:.4f}  drivers: {e.get('why', '')}"
                 for e in r[:5]]))

    conclusion = None
    if st and st.get("result"):
        conclusion = st["result"].get("summary") or {
            "headline": f"top pick {st['result'].get('top_pick')}"}
    return {"state": (st or {}).get("derived_state", "unknown"),
            "generated": (st or {}).get("updated"),
            "steps": steps, "conclusion": conclusion}
