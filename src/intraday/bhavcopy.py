"""Official NSE EOD files — equity bhavcopy (delivery %), F&O bhavcopy (OI, basis),
FII/DII flows, pre-open snapshot (PLAN_v3 Section 6, Tier 1).

NSE serves archives only to browser-like sessions: we warm up a session against
nseindia.com and send a real User-Agent. All outputs land in data/bhavcopy/ as
parquet, stamped with capture_ts. Parse failures raise (v1 post-mortem #7).
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime

import pandas as pd
import requests

from src.intraday import ROOT, load_config

logger = logging.getLogger(__name__)

ARCHIVES = "https://nsearchives.nseindia.com"
NSE_API = "https://www.nseindia.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


class BhavcopyError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Warm-up: NSE sets cookies needed by the API endpoints
    s.get("https://www.nseindia.com/", timeout=15)
    return s


def _out_dir():
    d = ROOT / load_config()["data"]["bhavcopy_path"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(s: requests.Session, url: str) -> bytes:
    resp = s.get(url, timeout=30)
    if resp.status_code != 200:
        raise BhavcopyError(f"HTTP {resp.status_code} for {url}")
    return resp.content


# ── Equity bhavcopy + delivery % ──────────────────────────────────────

def fetch_equity_bhavcopy(d: date, s: requests.Session | None = None) -> pd.DataFrame:
    """sec_bhavdata_full — OHLCV + DELIV_PER (delivery %) per symbol for one day."""
    s = s or _session()
    url = f"{ARCHIVES}/products/content/sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv"
    raw = _fetch(s, url)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df = df[df[" SERIES"].str.strip() == "EQ"] if " SERIES" in df.columns else df[df["SERIES"].str.strip() == "EQ"]
    df = df.rename(columns=lambda c: c.strip().lower())
    keep = ["symbol", "date1", "prev_close", "open_price", "high_price", "low_price",
            "close_price", "ttl_trd_qnty", "turnover_lacs", "deliv_qty", "deliv_per"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise BhavcopyError(f"equity bhavcopy {d}: missing columns {missing}")
    out = df[keep].copy()
    out["symbol"] = out["symbol"].str.strip()
    out["deliv_per"] = pd.to_numeric(out["deliv_per"], errors="coerce")
    out["date"] = pd.to_datetime(d)
    out["capture_ts"] = datetime.now()
    path = _out_dir() / f"eq_{d.isoformat()}.parquet"
    out.to_parquet(path, index=False)
    logger.info("equity bhavcopy %s: %d rows → %s", d, len(out), path.name)
    return out


# ── F&O bhavcopy: stock futures OI + basis ────────────────────────────

def fetch_fo_bhavcopy(d: date, s: requests.Session | None = None) -> pd.DataFrame:
    """F&O bhavcopy (UDiFF format) → near-month stock futures: OI, OI change, close."""
    s = s or _session()
    url = f"{ARCHIVES}/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
    raw = _fetch(s, url)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    fut = df[df["FinInstrmTp"].isin(["STF"])].copy()  # stock futures
    if fut.empty:
        raise BhavcopyError(f"F&O bhavcopy {d}: no stock futures rows")
    fut["XpryDt"] = pd.to_datetime(fut["XpryDt"])
    near = fut.sort_values("XpryDt").groupby("TckrSymb", as_index=False).first()
    out = near.rename(columns={
        "TckrSymb": "symbol", "ClsPric": "fut_close",
        "OpnIntrst": "oi", "ChngInOpnIntrst": "oi_change", "UndrlygPric": "spot_close",
    })[["symbol", "fut_close", "spot_close", "oi", "oi_change"]]
    out["basis_pct"] = (out["fut_close"] - out["spot_close"]) / out["spot_close"]
    out["date"] = pd.to_datetime(d)
    out["capture_ts"] = datetime.now()
    path = _out_dir() / f"fo_{d.isoformat()}.parquet"
    out.to_parquet(path, index=False)
    logger.info("F&O bhavcopy %s: %d stock futures → %s", d, len(out), path.name)
    return out


# ── FII / DII flows ───────────────────────────────────────────────────

def fetch_fii_dii(s: requests.Session | None = None) -> pd.DataFrame:
    """Latest FII/DII cash-market net flows (NSE publishes T-day after close)."""
    s = s or _session()
    raw = _fetch(s, f"{NSE_API}/fiidiiTradeReact")
    df = pd.DataFrame(pd.read_json(io.BytesIO(raw)))
    if df.empty:
        raise BhavcopyError("FII/DII endpoint returned empty payload")
    df["capture_ts"] = datetime.now()
    path = _out_dir() / "fii_dii.parquet"
    if path.exists():  # append-only history
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=[c for c in df.columns if c != "capture_ts"]
        )
    df.to_parquet(path, index=False)
    return df


# ── Pre-open auction snapshot (used live by recorder.py) ──────────────

def fetch_preopen(s: requests.Session | None = None) -> pd.DataFrame:
    """Pre-open data (09:00–09:08 order collection): indicative price + buy/sell imbalance."""
    s = s or _session()
    raw = _fetch(s, f"{NSE_API}/market-data-pre-open?key=FO")  # FO = all F&O-listed stocks
    payload = pd.read_json(io.BytesIO(raw))
    rows = []
    now = datetime.now()
    for rec in payload.get("data", []):
        meta = rec.get("metadata", {})
        detail = rec.get("detail", {}).get("preOpenMarket", {})
        rows.append({
            "symbol": meta.get("symbol"),
            "prev_close": meta.get("previousClose"),
            "indicative_open": meta.get("lastPrice"),
            "total_buy_qty": detail.get("totalBuyQuantity"),
            "total_sell_qty": detail.get("totalSellQuantity"),
            "capture_ts": now,
        })
    df = pd.DataFrame(rows).dropna(subset=["symbol"])
    if df.empty:
        raise BhavcopyError("pre-open endpoint returned no rows")
    buy = df["total_buy_qty"].astype(float)
    sell = df["total_sell_qty"].astype(float)
    df["imbalance"] = (buy - sell) / (buy + sell).replace(0, pd.NA)
    out_dir = ROOT / load_config()["data"]["preopen_path"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{date.today().isoformat()}.parquet", index=False)
    return df


# ── range download ────────────────────────────────────────────────────

def backfill_bhavcopies(start: date, end: date) -> None:
    """Download equity + F&O bhavcopies for a date range (skips weekends/holidays)."""
    s = _session()
    failures = []
    for d in pd.bdate_range(start, end).date:
        for fn in (fetch_equity_bhavcopy, fetch_fo_bhavcopy):
            out = _out_dir() / f"{'eq' if fn is fetch_equity_bhavcopy else 'fo'}_{d.isoformat()}.parquet"
            if out.exists():
                continue
            try:
                fn(d, s)
            except BhavcopyError as e:
                # Holidays return 404 — only a failure if equity AND fo both missing on a weekday
                failures.append((d, fn.__name__, str(e)))
    logger.info("bhavcopy backfill done; %d misses (incl. holidays)", len(failures))
