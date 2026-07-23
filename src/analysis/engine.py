"""Single-stock deep-analysis engine — see package docstring for the contract.

analyze(symbol) → dict with sections:
  identity · price_stats · technicals · flow · fundamentals · news · regime ·
  model_view · scorecard · verdict · reasoning · references · disclaimer

Data sources (all fail-soft with present=False, never fabricated):
  price   — data/daily/<SYM>.parquet (universe, current) else the v6 full-market
            panel cache topped up with bhavcopy eq_* files newer than the cache
  flow    — delivery %, futures OI / basis (bhavcopy-derived columns)
  fund    — yfinance <SYM>.NS (name/sector/PE/market cap/52w/analyst fields)
  news    — GDELT DOC API, 30-day window, bucketed today/week/month; sentiment
            via FinBERT when installed, else a small disclosed keyword lexicon
  regime  — latest data/regime/global day-file (India VIX, global tone)
  model   — the swing_1_5d bundle's calibrated P(win) when the symbol is in the
            daily universe (with the honest measured-edge note)
"""

from __future__ import annotations

import glob
import json
import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday import ROOT, today_ist

logger = logging.getLogger(__name__)

DAILY_DIR = ROOT / "data" / "daily"
V6_PANEL = ROOT / "data" / "processed" / "v6_panel.parquet"
BHAV_DIR = ROOT / "data" / "bhavcopy"
REGIME_DIR = ROOT / "data" / "regime" / "global"
OUT_DIR = ROOT / "reports" / "analysis"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# ── verdict thresholds (transparent constants, printed in the output) ──
BUY_MIN, SELL_MAX = 0.25, -0.25

# small disclosed lexicon — the FALLBACK when FinBERT (transformers) is absent
_POS = {"profit", "surge", "surges", "wins", "win", "order", "orders", "upgrade",
        "upgraded", "beats", "beat", "record", "growth", "expansion", "approval",
        "approves", "rally", "gains", "jump", "jumps", "strong", "buyback",
        "dividend", "contract", "awarded", "acquisition", "partnership", "soars"}
_NEG = {"loss", "losses", "falls", "fall", "probe", "fraud", "downgrade",
        "downgraded", "misses", "miss", "decline", "declines", "layoff",
        "layoffs", "penalty", "default", "resign", "resigns", "plunge",
        "plunges", "weak", "slump", "cuts", "investigation", "lawsuit",
        "recall", "strike", "drops", "crash", "scam", "raid"}


# ── data loading ──────────────────────────────────────────────────────

def load_history(symbol: str) -> tuple[pd.DataFrame, str]:
    """Daily OHLCV(+flow) history for any NSE symbol. Returns (df, source_note).
    Universe symbols come from data/daily (current, CA-adjusted at build);
    everything else from the v6 full-market panel topped up with bhavcopy
    files newer than the cache (top-up rows are as-traded — disclosed)."""
    f = DAILY_DIR / f"{symbol}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True), "daily panel (universe, adjusted)"

    if not V6_PANEL.exists():
        raise FileNotFoundError("v6 panel cache missing — run scripts.v6_phase0_grid or daily-panel")
    pan = _v6_symbol_slice(symbol)
    if pan.empty:
        raise ValueError(f"{symbol!r} not found in the NSE bhavcopy panel — check the symbol")
    top = _bhav_topup(symbol, pan["date"].max())
    if len(top):
        pan = pd.concat([pan, top], ignore_index=True).drop_duplicates("date")
    return (pan.sort_values("date").reset_index(drop=True),
            "v6 full-market panel + bhavcopy top-up (recent rows as-traded)")


@lru_cache(maxsize=64)
def _v6_symbol_slice(symbol: str) -> pd.DataFrame:
    pan = pd.read_parquet(V6_PANEL, columns=[
        "symbol", "date", "open", "high", "low", "close", "volume",
        "turnover_lacs", "deliv_per"])
    out = pan[pan["symbol"] == symbol].drop(columns=["symbol"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def _bhav_topup(symbol: str, after: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(BHAV_DIR / "eq_*.parquet"))):
        d = pd.Timestamp(Path(f).stem[3:])
        if d <= after:
            continue
        day = pd.read_parquet(f, columns=["symbol", "date", "open_price", "high_price",
                                          "low_price", "close_price", "ttl_trd_qnty",
                                          "turnover_lacs", "deliv_per"])
        r = day[day["symbol"] == symbol]
        if len(r):
            rows.append(r.rename(columns={
                "open_price": "open", "high_price": "high", "low_price": "low",
                "close_price": "close", "ttl_trd_qnty": "volume"}))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).drop(columns=["symbol"])
    out["date"] = pd.to_datetime(out["date"])
    out["deliv_per"] = pd.to_numeric(out["deliv_per"], errors="coerce")
    return out


# ── technicals ────────────────────────────────────────────────────────

def technicals(df: pd.DataFrame) -> dict:
    c = df["close"].astype(float)
    out: dict = {"present": True, "as_of": str(df["date"].iloc[-1].date()), "close": round(float(c.iloc[-1]), 2)}
    for label, n in [("ret_1d", 1), ("ret_1w", 5), ("ret_1m", 21), ("ret_3m", 63),
                     ("ret_6m", 126), ("ret_1y", 252)]:
        out[label] = round(float(c.iloc[-1] / c.iloc[-1 - n] - 1), 4) if len(c) > n else None
    for n in (20, 50, 200):
        out[f"sma{n}"] = round(float(c.tail(n).mean()), 2) if len(c) >= n else None
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    out["rsi14"] = round(float((100 - 100 / (1 + rs)).iloc[-1]), 1) if len(c) > 15 else None
    if len(df) > 15:
        pc = c.shift(1)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
        out["atr_pct"] = round(float(tr.rolling(14).mean().iloc[-1] / c.iloc[-1]), 4)
    else:
        out["atr_pct"] = None
    if len(c) >= 252:
        yr = c.tail(252)
        rng = float(yr.max() - yr.min())
        out["pos_52w"] = round(float((c.iloc[-1] - yr.min()) / rng), 3) if rng > 0 else 0.5
        out["high_52w"], out["low_52w"] = round(float(yr.max()), 2), round(float(yr.min()), 2)
        peak = yr.cummax()
        out["max_drawdown_1y"] = round(float(((yr - peak) / peak).min()), 4)
    else:
        out["pos_52w"] = out["high_52w"] = out["low_52w"] = out["max_drawdown_1y"] = None
    out["ann_vol"] = round(float(c.pct_change().tail(60).std() * np.sqrt(252)), 4) if len(c) > 61 else None
    if "volume" in df.columns and len(df) > 21:
        v = df["volume"].astype(float)
        out["vol_ratio_20"] = round(float(v.iloc[-1] / max(v.tail(21).head(20).mean(), 1)), 2)
    else:
        out["vol_ratio_20"] = None
    return out


def flow_stats(df: pd.DataFrame) -> dict:
    out = {"present": False}
    if "deliv_per" in df.columns:
        d = pd.to_numeric(df["deliv_per"], errors="coerce").dropna()
        if len(d) >= 65:
            base = d.tail(65).head(60)
            z = (d.tail(5).mean() - base.mean()) / (base.std() or 1.0)
            out.update(present=True, deliv_last5_mean=round(float(d.tail(5).mean()), 1),
                       deliv_60d_mean=round(float(base.mean()), 1), deliv_z=round(float(z), 2))
    if "oi_change" in df.columns:
        oi = pd.to_numeric(df["oi_change"], errors="coerce").dropna()
        if len(oi) >= 5:
            out["oi_change_5d"] = int(oi.tail(5).sum())
            out["basis_pct_last"] = (round(float(df["basis_pct"].iloc[-1]), 4)
                                     if "basis_pct" in df.columns and pd.notna(df["basis_pct"].iloc[-1]) else None)
    return out


# ── fundamentals (yfinance) ───────────────────────────────────────────

def fundamentals(symbol: str) -> dict:
    try:
        import yfinance as yf
        info = yf.Ticker(f"{symbol}.NS").info or {}
        if not info.get("longName") and not info.get("shortName"):
            return {"present": False, "note": "yfinance returned no data for this symbol"}
        pick = {"name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"), "industry": info.get("industry"),
                "market_cap_inr": info.get("marketCap"),
                "pe_trailing": info.get("trailingPE"), "pe_forward": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"), "beta": info.get("beta"),
                "analyst_recommendation": info.get("recommendationKey"),
                "analyst_target_mean": info.get("targetMeanPrice"),
                "analyst_count": info.get("numberOfAnalystOpinions")}
        return {"present": True, **{k: v for k, v in pick.items() if v is not None}}
    except Exception as e:  # noqa: BLE001 — network/parse failures degrade honestly
        return {"present": False, "note": f"fundamentals unavailable ({type(e).__name__})"}


# ── news (GDELT) ──────────────────────────────────────────────────────

def _lexicon_score(title: str) -> float:
    words = set(re.findall(r"[a-z']+", title.lower()))
    pos, neg = len(words & _POS), len(words & _NEG)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


_NEWS_CACHE_DIR = OUT_DIR / "news_cache"
_last_gdelt_call = 0.0


def _gdelt_fetch(query: str) -> list | None:
    """One GDELT call with GLOBAL 5.5s spacing (their limit is 1 req/5s per IP)
    + compliant retries. Returns None only when genuinely unreachable."""
    global _last_gdelt_call
    import time

    import requests
    for backoff in (6, 15, 30):   # GDELT throttle windows can outlast a single 5s wait
        wait = 5.5 - (time.time() - _last_gdelt_call)
        if wait > 0:
            time.sleep(wait)
        _last_gdelt_call = time.time()
        try:
            r = requests.get(GDELT_DOC_API, timeout=25, params={
                "query": f"{query} sourcelang:english", "mode": "artlist",
                "format": "json", "maxrecords": 60, "timespan": "30d", "sort": "datedesc"})
            return r.json().get("articles", [])
        except ValueError:      # non-JSON body = throttled — back off and retry
            time.sleep(backoff)
        except Exception:  # noqa: BLE001 — network down etc.
            return None
    return None


def fetch_news(symbol: str, company: str | None) -> dict:
    """30-day GDELT sweep bucketed into today / week / month, each headline
    scored (FinBERT if installed, else the disclosed lexicon) and linked.
    Cached per (symbol, day) on disk so repeated UI clicks never re-hit GDELT."""
    cache = _NEWS_CACHE_DIR / f"{symbol}_{today_ist().isoformat()}.json"
    if cache.exists():
        return {**json.loads(cache.read_text()), "note": "served from today's cache"}

    query = f'"{company}"' if company else f'"{symbol}" NSE'
    arts = _gdelt_fetch(query)
    if arts is None:
        # last resort: any earlier cached day for this symbol, clearly labelled
        stale = sorted(_NEWS_CACHE_DIR.glob(f"{symbol}_*.json"))
        if stale:
            old = json.loads(stale[-1].read_text())
            return {**old, "note": f"GDELT throttled — showing cached news from "
                                   f"{stale[-1].stem.split('_', 1)[1]}"}
        return {"present": False, "note": "news unavailable (GDELT throttled — retry in ~30s)",
                "articles": []}

    scorer, engine_name = _lexicon_score, "keyword-lexicon (FinBERT not installed)"
    try:  # optional upgrade — only if transformers+torch exist
        from src.intraday.news_regime import _finbert_score  # type: ignore
        scorer, engine_name = _finbert_score, "FinBERT"
    except Exception:  # noqa: BLE001
        pass

    now, seen, items = pd.Timestamp(today_ist()), set(), []
    for a in arts:
        title = (a.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        try:
            ts = pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce")
        except Exception:  # noqa: BLE001
            ts = pd.NaT
        age = (now - ts.normalize()).days if pd.notna(ts) else None
        items.append({"title": title, "url": a.get("url"), "domain": a.get("domain"),
                      "date": str(ts.date()) if pd.notna(ts) else None,
                      "bucket": ("today" if age is not None and age <= 1 else
                                 "week" if age is not None and age <= 7 else "month"),
                      "sentiment": round(float(scorer(title)), 3)})
    scored = [i["sentiment"] for i in items if i["sentiment"] != 0.0]
    out = {"present": bool(items), "engine": engine_name, "n_articles": len(items),
           "n_today": sum(1 for i in items if i["bucket"] == "today"),
           "n_week": sum(1 for i in items if i["bucket"] in ("today", "week")),
           "mean_sentiment": round(float(np.mean(scored)), 3) if scored else 0.0,
           "articles": items[:25]}
    if items:   # persist today's successful sweep → repeat clicks never re-hit GDELT
        _NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


# ── regime + model view ───────────────────────────────────────────────

def market_regime() -> dict:
    files = sorted(glob.glob(str(REGIME_DIR / "*.parquet")))
    if not files:
        return {"present": False}
    df = pd.read_parquet(files[-1])
    vals = {r["instrument"]: float(r["value"]) for _, r in df.iterrows()
            if r.get("field") in ("prior_close", "close", "value")}
    vix = vals.get("india_vix")
    return {"present": True, "as_of": Path(files[-1]).stem, "india_vix": vix,
            "note": ("calm (<14)" if vix and vix < 14 else
                     "elevated (>20)" if vix and vix > 20 else "normal") if vix else "n/a"}


def model_view(symbol: str) -> dict:
    """Calibrated P(win) from the swing model — anchored to the symbol's LAST
    AVAILABLE close (the panel lags the calendar by a session: asking for
    'today' before today's bhavcopy exists raised DatasetThinningError)."""
    f = DAILY_DIR / f"{symbol}.parquet"
    if not f.exists():
        return {"present": False, "note": "symbol not in the daily model universe (top-100)"}
    try:
        from src.daily.features import build_matrix, check_schema
        from src.daily.trainer import load_bundle
        pm, blender, _ = load_bundle("swing_1_5d")
        as_of = pd.read_parquet(f, columns=["date"])["date"].max().date()
        plan = pd.DataFrame({"symbol": [symbol], "date": [as_of]})
        m = build_matrix(plan)
        if m.empty:
            return {"present": False, "note": f"no feature row as of {as_of}"}
        check_schema(m)
        p = float(blender.calibrated(pm.predict_proba(m))[0])
        return {"present": True, "p_win_calibrated": round(p, 4), "as_of": str(as_of),
                "note": ("measured record: the model has shown no reliable edge "
                         "(AUC≈0.50 in walk-forward) — treat as context, not signal")}
    except Exception as e:  # noqa: BLE001
        return {"present": False, "note": f"model view unavailable ({type(e).__name__})"}


# ── scorecard + verdict (pure — unit-tested) ──────────────────────────

def build_scorecard(tech: dict, flow: dict, fund: dict, news: dict,
                    regime: dict, model: dict) -> list[dict]:
    """Each component: name, weight, score ∈ [-1,1], present, value, reason."""
    rows = []

    def add(name, weight, present, score, value, reason):
        rows.append({"component": name, "weight": weight, "present": bool(present),
                     "score": round(float(np.clip(score, -1, 1)), 3) if present else 0.0,
                     "value": value, "reason": reason})

    # ── horizon note: weights tuned for a 1–5 TRADING-DAY swing (user contract
    # 2026-07-16). Short-term momentum/volume/flow/news carry the verdict; slow
    # signals (SMA trend, 52w position) are demoted to context; P/E is dropped
    # (meaningless over 5 days).
    r1d, r1w = tech.get("ret_1d"), tech.get("ret_1w")
    if r1d is not None and r1w is not None:
        mom = float(np.tanh(((r1w + 2 * r1d) / 2) / 0.03))
        add("short_momentum", 0.20, True, mom, f"1d {r1d:+.1%}, 5d {r1w:+.1%}",
            "positive 1–5 session momentum" if mom > 0 else "negative 1–5 session momentum")
    else:
        add("short_momentum", 0.20, False, 0, None, "insufficient history")

    vr = tech.get("vol_ratio_20")
    if vr is not None and r1d is not None:
        if vr >= 1.5:
            score, why = ((0.6, f"volume {vr:.1f}× its 20d norm on an up day — fresh interest")
                          if r1d > 0 else
                          (-0.6, f"volume {vr:.1f}× its 20d norm on a down day — distribution"))
        elif vr <= 0.5:
            score, why = -0.1, f"volume drying up ({vr:.1f}× norm) — little participation"
        else:
            score, why = 0.0, f"volume near its norm ({vr:.1f}×)"
        add("volume_surge", 0.10, True, score, f"{vr:.2f}×", why)
    else:
        add("volume_surge", 0.10, False, 0, None, "volume data unavailable")

    rsi = tech.get("rsi14")
    if rsi is not None:
        score, why = ((-0.6, "overbought (>75) — stretched for a fresh 1–5d entry") if rsi > 75 else
                      (-0.2, "high RSI (65–75)") if rsi > 65 else
                      (0.3, "healthy RSI (45–65)") if rsi >= 45 else
                      (0.0, "soft RSI (30–45)") if rsi >= 30 else
                      (0.2, "oversold (<30) — bounce-watch, needs stabilization"))
        add("rsi", 0.12, True, score, rsi, why)
    else:
        add("rsi", 0.12, False, 0, None, "insufficient history")

    dz = flow.get("deliv_z")
    if flow.get("present") and dz is not None and r1w is not None:
        if dz > 1:
            score, why = ((0.7, "delivery % surging with a rising week — accumulation")
                          if r1w > 0 else (-0.5, "delivery % surging into a falling week — conviction selling"))
        elif dz < -1:
            score, why = -0.2, "delivery % fading — weak conviction"
        else:
            score, why = 0.0, "delivery % near its norm"
        add("delivery_flow", 0.14, True, score, f"deliv z={dz:+.2f} (5d vs 60d)", why)
    else:
        add("delivery_flow", 0.14, False, 0, None, "delivery data unavailable")

    if news.get("present"):
        ms, nw = news["mean_sentiment"], news["n_week"]
        score = float(np.clip(ms * min(1.0, nw / 5.0) * 2, -1, 1))
        add("news_sentiment", 0.14, True, score,
            f"mean {ms:+.2f} over {news['n_articles']} articles ({nw} this week)",
            f"{news['engine']}: {'positive' if ms > 0.05 else 'negative' if ms < -0.05 else 'neutral'} tone")
    else:
        add("news_sentiment", 0.14, False, 0, None, news.get("note", "no coverage found"))

    c, s20, s50, s200 = tech.get("close"), tech.get("sma20"), tech.get("sma50"), tech.get("sma200")
    if all(v is not None for v in (c, s20, s50, s200)):
        checks = [c > s20, c > s50, c > s200, s50 > s200]
        score = (sum(checks) / 4) * 2 - 1
        add("trend_context", 0.12, True, score, f"close {c} vs SMA20/50/200 {s20}/{s50}/{s200}",
            f"{sum(checks)}/4 uptrend checks — context: a 1–5d long swims easier with the trend")
    else:
        add("trend_context", 0.12, False, 0, None, "insufficient history for SMA200")

    if regime.get("present") and regime.get("india_vix"):
        vix = regime["india_vix"]
        score = 0.3 if vix < 14 else (-0.5 if vix > 20 else 0.0)
        add("market_regime", 0.08, True, score, f"India VIX {vix:.1f}",
            f"volatility {regime['note']} — gap risk over a 1–5d hold")
    else:
        add("market_regime", 0.08, False, 0, None, "regime data unavailable")

    p52 = tech.get("pos_52w")
    if p52 is not None:
        score = 0.4 if p52 > 0.8 else (-0.4 if p52 < 0.2 else 0.0)
        add("52w_position", 0.05, True, score, p52,
            "near 52w high — strength context" if p52 > 0.8 else
            "near 52w low — weakness context" if p52 < 0.2 else "mid-range")
    else:
        add("52w_position", 0.05, False, 0, None, "insufficient history")

    if model.get("present"):
        p = model["p_win_calibrated"]
        add("model_view", 0.05, True, float(np.clip((p - 0.5) * 4, -1, 1)),
            f"calibrated P(win) {p:.3f}", model["note"])
    else:
        add("model_view", 0.05, False, 0, None, model.get("note", "unavailable"))
    return rows


def trade_plan(action: str, tech: dict, capital_inr: float = 100_000) -> dict | None:
    """If the user acts on the verdict — a 1–5 TRADING-DAY swing plan (user
    contract 2026-07-16), pure and unit-tested. Geometry mirrors the daily
    system's swing_1_5d horizon (config_daily.yaml): target/stop = ±target_atr/
    stop_atr × ATR(14), TIME EXIT at the close of the 5th session — a swing
    trade that hasn't worked in 5 days is over, not 'given more time'.

    SELL plans are exit/avoid guidance — retail cannot hold overnight shorts
    in the NSE cash segment."""
    c, atr_pct = tech.get("close"), tech.get("atr_pct")
    if action not in ("BUY", "SELL") or not c or not atr_pct:
        return None
    try:  # one source of truth with the daily system's swing block
        from src.daily import horizon_config
        hc = horizon_config("swing_1_5d")
        t_atr, s_atr, hold_days = hc["target_atr"], hc["stop_atr"], hc["time_barrier_days"]
    except Exception:  # noqa: BLE001 — config unavailable (tests) → same defaults
        t_atr, s_atr, hold_days = 1.0, 1.0, 5
    atr_abs = atr_pct * c
    risk = round(s_atr * atr_abs, 2)
    sign = 1 if action == "BUY" else -1
    stop = round(c - sign * risk, 2)
    t1 = round(c + sign * t_atr * atr_abs, 2)
    t2 = round(c + sign * 2 * t_atr * atr_abs, 2)
    return {
        "horizon": "swing 1–5 trading days",
        "direction": "LONG" if action == "BUY" else "EXIT/AVOID (no overnight cash-segment short)",
        "entry_rule": "next session open — re-anchor these levels to your actual fill",
        "ref_close": round(c, 2), "atr_pct": round(atr_pct, 4),
        "risk_per_share": risk, "risk_basis": f"R = {s_atr:g}×ATR(14)",
        "stop_price": stop,
        "stop_rule": f"hard stop at entry−{s_atr:g}×ATR — no averaging down, no 'one more day'",
        "target_1": t1, "target_1_rr": f"+{t_atr:g}×ATR (1:1) — book at least half",
        "target_2": t2, "target_2_rr": f"+{2 * t_atr:g}×ATR (2:1) — stretch; rarely hit inside 5 days",
        "max_hold_days": int(hold_days),
        "time_exit_rule": f"EXIT at the close of session {hold_days} regardless — "
                          "measured NSE data shows most 1–5d trades end as time exits, "
                          "so the time stop is the rule, not the exception",
        "suggested_holding": f"1–{hold_days} trading days; the scorecard weights 1–5 session "
                             "momentum/volume/flow — it says nothing beyond next week",
        "capital_ref_inr": capital_inr,
        "qty_ref": max(1, int(capital_inr / c)),
        "qty_if_risking_1pct": max(1, int(0.01 * capital_inr / risk)) if risk > 0 else None,
        "sizing_note": "qty_if_risking_1pct risks ~1% of reference capital if the stop is hit "
                       "— the safer default; qty_ref simply spends the full reference amount",
        "round_trip_cost_pct": 0.39,
        "cost_note": f"NSE delivery round trip ≈0.39% at ₹1L; T1 is ±{t_atr * atr_pct:.1%} away "
                     "— the cost eats a meaningful slice of a 1×ATR move; factor it in",
        "geometry_note": "symmetric ±1×ATR ⇒ ~50/50 no-edge baseline by construction — the "
                         "plan's value is DISCIPLINE (pre-committed exits), not prediction",
    }


def verdict_from_scorecard(rows: list[dict]) -> dict:
    present = [r for r in rows if r["present"]]
    w_all = sum(r["weight"] for r in rows)
    w_present = sum(r["weight"] for r in present)
    composite = (sum(r["weight"] * r["score"] for r in present) / w_present) if w_present else 0.0
    completeness = w_present / w_all if w_all else 0.0
    action = "BUY" if composite >= BUY_MIN else ("SELL" if composite <= SELL_MAX else "HOLD")
    strength = abs(composite)
    label = ("HIGH" if strength > 0.5 and completeness > 0.8 else
             "MEDIUM" if strength > 0.3 and completeness > 0.6 else "LOW")
    return {"action": action, "composite_score": round(composite, 3),
            "thresholds": {"buy_min": BUY_MIN, "sell_max": SELL_MAX},
            "confidence": label, "data_completeness": round(completeness, 2)}


# ── chart ─────────────────────────────────────────────────────────────

def render_chart(symbol: str, df: pd.DataFrame, plan: dict | None = None) -> str | None:
    """60 daily sessions (~3 months of context for a 1–5 day swing) + SMA20 and,
    when a trade plan exists, its target/stop levels drawn on the chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = df.tail(60)
        c = d["close"].astype(float)
        fig, (ax, axv) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(d["date"], c, lw=1.4, label="close")
        sma20 = df["close"].astype(float).rolling(20).mean().tail(60)
        if sma20.notna().any():
            ax.plot(d["date"], sma20, lw=1, label="SMA20")
        if plan:
            ax.axhline(plan["target_1"], color="green", ls="--", lw=1,
                       label=f"target {plan['target_1']}")
            ax.axhline(plan["stop_price"], color="red", ls="--", lw=1,
                       label=f"stop {plan['stop_price']}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{symbol} — last {len(d)} sessions (1–5d swing frame)")
        if "volume" in d.columns:
            axv.bar(d["date"], d["volume"].astype(float), width=1.0)
            axv.set_ylabel("vol", fontsize=8)
        fig.autofmt_xdate()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{symbol}_{today_ist().isoformat()}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return str(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("chart render %s: %s", symbol, e)
        return None


# ── orchestration ─────────────────────────────────────────────────────

DISCLAIMER = ("Research aid, not investment advice. The verdict is transparent "
              "rule-based evidence aggregation; this repo's measured record "
              "(reports/daily/run_registry.csv) shows NO proven predictive edge "
              "for any model at these horizons. Verify every input via the "
              "reference links before acting; markets carry risk of loss.")


def analyze(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    hist, source_note = load_history(symbol)
    if len(hist) < 30:
        raise ValueError(f"{symbol}: only {len(hist)} sessions on disk — too thin to analyze")

    tech = technicals(hist)
    flow = flow_stats(hist)
    fund = fundamentals(symbol)
    news = fetch_news(symbol, fund.get("name"))
    regime = market_regime()
    model = model_view(symbol)

    card = build_scorecard(tech, flow, fund, news, regime, model)
    verdict = verdict_from_scorecard(card)
    plan = trade_plan(verdict["action"], tech)

    contributors = sorted((r for r in card if r["present"]),
                          key=lambda r: abs(r["weight"] * r["score"]), reverse=True)
    reasoning = [f"{r['component']}: {r['reason']} (score {r['score']:+.2f} × weight {r['weight']:.2f})"
                 for r in contributors]
    reasoning.insert(0, (
        f"Composite evidence score {verdict['composite_score']:+.3f} against thresholds "
        f"[SELL ≤ {SELL_MAX} < HOLD < {BUY_MIN} ≤ BUY] with "
        f"{verdict['data_completeness']:.0%} of evidence sources available → "
        f"{verdict['action']} ({verdict['confidence']} confidence)."))

    refs = [
        {"label": "NSE quote (official)", "url": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"},
        {"label": "Screener.in fundamentals", "url": f"https://www.screener.in/company/{symbol}/"},
        {"label": "Yahoo Finance", "url": f"https://finance.yahoo.com/quote/{symbol}.NS"},
        {"label": "TradingView chart", "url": f"https://www.tradingview.com/symbols/NSE-{symbol}/"},
        {"label": "Google News search", "url": f"https://news.google.com/search?q={symbol}%20NSE%20stock"},
    ] + [{"label": f"news: {a['title'][:70]}", "url": a["url"]}
         for a in news.get("articles", [])[:8] if a.get("url")]

    return {
        "symbol": symbol, "generated": datetime.now().isoformat(timespec="seconds"),
        "identity": {"name": fund.get("name", symbol), "sector": fund.get("sector"),
                     "industry": fund.get("industry"), "history_source": source_note,
                     "sessions_on_disk": len(hist)},
        "verdict": verdict, "trade_plan": plan, "scorecard": card, "reasoning": reasoning,
        "technicals": tech, "flow": flow, "fundamentals": fund,
        "news": news, "regime": regime, "model_view": model,
        "chart_path": render_chart(symbol, hist, plan),
        "references": refs, "disclaimer": DISCLAIMER,
    }
