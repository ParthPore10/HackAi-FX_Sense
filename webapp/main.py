from __future__ import annotations

import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict, List
import re
import requests
import feedparser
import os
import threading
import time
import json

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

try:
    from utils.gemini_client import _call_gemini as _gemini_call
except Exception:
    _gemini_call = None


APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT.parent / "data" / "signals_latest.csv"

PAIRS: List[str] = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURJPY=X",
    "EURGBP=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "EURAUD=X",
    "EURNZD=X",
    "EURCHF=X",
    "CADJPY=X",
    "CHFJPY=X",
    "AUDNZD=X",
    "AUDCAD=X",
    "GBPCAD=X",
    "GBPAUD=X",
]
YAHOO_FX_TICKERS: List[str] = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURJPY=X",
    "EURGBP=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "EURAUD=X",
    "EURNZD=X",
    "EURCHF=X",
    "CADJPY=X",
    "CHFJPY=X",
    "AUDNZD=X",
    "AUDCAD=X",
    "GBPCAD=X",
    "GBPAUD=X",
    "USDINR=X",
    "USDCNY=X",
]

YAHOO_CURRENCIES_URL = "https://finance.yahoo.com/markets/currencies/"
AISSTREAM_API_KEY = os.environ.get("ca332f9932b2b365bef2b00162bb8035076e5261", "").strip()

YOUTUBE_LIVE_HANDLES = {
    "bloomberg": "BloombergTV",
    "cnbc": "CNBC",
    "reuters": "Reuters",
    "aljazeera": "AlJazeeraEnglish",
    "france24": "France24_en",
    "skynews": "SkyNews",
}

TREASURY_TICKERS = {
    "^IRX": "3M",
    "^FVX": "5Y",
    "^TNX": "10Y",
    "^TYX": "30Y",
}

COMMODITY_TICKERS = {
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "NG=F": "Nat Gas",
}


FX_GEO = {
    "USD": {"name": "Washington DC", "lat": 38.9072, "lon": -77.0369, "color": "#2ee6a6"},
    "EUR": {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "color": "#1fb7ff"},
    "GBP": {"name": "London", "lat": 51.5074, "lon": -0.1278, "color": "#ffb703"},
    "JPY": {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "color": "#ff4d6d"},
    "AUD": {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "color": "#8ac926"},
    "INR": {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "color": "#f77f00"},
    "CNY": {"name": "Beijing", "lat": 39.9042, "lon": 116.4074, "color": "#7b2cbf"},
}

COUNTRY_NEWS_FEEDS = {
    "USD": [
        ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
    ],
    "EUR": [
        ("ECB", "https://www.ecb.europa.eu/rss/press.html"),
    ],
    "GBP": [
        ("Bank of England", "https://www.bankofengland.co.uk/rss/news"),
    ],
    "JPY": [
        ("Bank of Japan", "https://www.boj.or.jp/en/rss/whatsnew.rdf"),
        ("Japan FSA", "https://www.fsa.go.jp/fsaEnNewsList_rss2.xml"),
    ],
    "AUD": [
        ("RBA", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml"),
    ],
    "CAD": [
        ("Bank of Canada", "https://www.bankofcanada.ca/content_type/press-releases/feed/"),
    ],
    "CHF": [
        ("SNB", "https://www.snb.ch/public/en/rss/pressrel"),
    ],
    "NZD": [
        ("NZ Treasury", "http://treasury.govt.nz/feeds/news"),
    ],
    "INR": [
        ("RBI", "https://rbi.org.in/pressreleases_rss.xml"),
    ],
    "CNY": [
        ("China Daily Biz", "http://www.chinadaily.com.cn/rss/bizchina_rss.xml"),
    ],
}


app = FastAPI(title="FXSense")
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")


@app.on_event("startup")
def _on_startup():
    _start_aisstream()


_cache: Dict[str, dict] = {}


def _cache_get(key: str, ttl: int):
    item = _cache.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > ttl:
        return None
    return item["value"]


def _cache_set(key: str, value, ttl: int):
    _cache[key] = {"ts": time.time(), "ttl": ttl, "value": value}


def _fetch_yahoo_fx_tickers() -> List[str]:
    """
    Best-effort scrape of Yahoo Finance currencies page to discover FX tickers.
    Falls back to static list if blocked or parsing fails.
    """
    cached = _cache_get("yahoo_fx_tickers", ttl=3600)
    if cached:
        return cached
    try:
        headers = {
            "User-Agent": "FXSenseMVP/0.1 (+https://example.com)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(YAHOO_CURRENCIES_URL, headers=headers, timeout=10)
        if resp.status_code != 200:
            _cache_set("yahoo_fx_tickers", YAHOO_FX_TICKERS, ttl=3600)
            return YAHOO_FX_TICKERS
        text = resp.text
        # Extract symbols like EURUSD=X, USDJPY=X, etc.
        symbols = set(re.findall(r'\"symbol\":\"([A-Z]{6}=X)\"', text))
        # Keep a reasonable size
        tickers = sorted(symbols)
        if not tickers:
            tickers = YAHOO_FX_TICKERS
        _cache_set("yahoo_fx_tickers", tickers, ttl=3600)
        return tickers
    except Exception:
        _cache_set("yahoo_fx_tickers", YAHOO_FX_TICKERS, ttl=3600)
        return YAHOO_FX_TICKERS


# -------------------------
# AISStream live tankers
# -------------------------
_ais_lock = threading.Lock()
_ais_tankers: Dict[str, dict] = {}
_ais_started = False


def _is_tanker(msg: dict) -> bool:
    # Temporarily allow all vessels until tanker filter is verified
    return True


def _aisstream_worker() -> None:
    try:
        import websocket  # type: ignore
    except Exception:
        return

    if not AISSTREAM_API_KEY:
        return

    url = "wss://stream.aisstream.io/v0/stream"
    payload = {
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": [[[ -90, -180 ], [ 90, 180 ]]],
    }

    def on_message(ws, message):
        try:
            data = json.loads(message)
            msg = data.get("Message", {})
            pos = msg.get("PositionReport") or msg.get("Position") or msg
            if not isinstance(pos, dict):
                return
            lat = pos.get("Latitude") or pos.get("LAT") or pos.get("lat")
            lon = pos.get("Longitude") or pos.get("LON") or pos.get("lon")
            mmsi = pos.get("MMSI") or msg.get("MMSI") or data.get("MMSI")
            if lat is None or lon is None or mmsi is None:
                return
            if not _is_tanker(msg):
                return
            entry = {
                "mmsi": str(mmsi),
                "lat": float(lat),
                "lon": float(lon),
                "speed": float(pos.get("SOG", 0)) if pos.get("SOG") is not None else 0,
                "heading": pos.get("COG", 0),
                "timestamp": data.get("Timestamp") or msg.get("Timestamp"),
                "name": msg.get("ShipName") or msg.get("NAME") or "",
            }
            with _ais_lock:
                _ais_tankers[str(mmsi)] = entry
                # keep memory bounded
                if len(_ais_tankers) > 2000:
                    # drop oldest by timestamp if possible
                    keys = list(_ais_tankers.keys())[:500]
                    for k in keys:
                        _ais_tankers.pop(k, None)
        except Exception:
            return

    def on_open(ws):
        ws.send(json.dumps(payload))

    while True:
        try:
            ws = websocket.WebSocketApp(url, on_message=on_message, on_open=on_open)
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception:
            time.sleep(5)


def _start_aisstream() -> None:
    global _ais_started
    if _ais_started:
        return
    _ais_started = True
    t = threading.Thread(target=_aisstream_worker, daemon=True)
    t.start()




def _fetch_country_news(limit_per_ccy: int = 5) -> dict:
    cache_key = f"country_news:{limit_per_ccy}"
    cached = _cache_get(cache_key, ttl=300)
    if cached:
        return cached

    out = {}
    for ccy, feeds in COUNTRY_NEWS_FEEDS.items():
        items = []
        for source, url in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit_per_ccy]:
                    items.append(
                        {
                            "source": source,
                            "title": entry.get("title", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", ""),
                        }
                    )
            except Exception:
                continue
        out[ccy] = items[:limit_per_ccy]

    _cache_set(cache_key, out, ttl=300)
    return out


def _fetch_youtube_live_embed(handle: str) -> dict:
    """
    Best-effort resolver for YouTube live video ID from channel handle.
    Returns embed URL or fallback to channel /live.
    """
    cache_key = f"yt_live:{handle}"
    cached = _cache_get(cache_key, ttl=300)
    if cached:
        return cached

    # Reuters often publishes live content under /streams
    if handle.lower() == "reuters":
        url = f"https://www.youtube.com/@{handle}/streams"
    else:
        url = f"https://www.youtube.com/@{handle}/live"
    embed_url = url
    video_id = None
    try:
        headers = {"User-Agent": "FXSenseMVP/0.1"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            # Prefer an actively live stream when available
            live_match = re.search(
                r'"videoId":"([a-zA-Z0-9_-]{11})".*?"isLiveNow":true',
                resp.text,
                re.DOTALL,
            )
            if live_match:
                video_id = live_match.group(1)
            else:
                # Fallback to first videoId found
                match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
                if match:
                    video_id = match.group(1)
            if video_id:
                embed_url = f"https://www.youtube.com/embed/{video_id}"
    except Exception:
        pass

    payload = {
        "handle": handle,
        "video_id": video_id,
        "embed_url": embed_url,
        "live_url": url,
    }
    _cache_set(cache_key, payload, ttl=300)
    return payload


def _download_multi(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        df = yf.download(
            tickers=" ".join(tickers),
            period="5d",
            interval="5m",
            progress=False,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )
    return df


def _build_live_rows(df: pd.DataFrame, tickers: List[str]) -> List[dict]:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = []
    if df is None or df.empty:
        for t in tickers:
            rows.append(
                {
                    "pair": t,
                    "last": None,
                    "change": None,
                    "change_pct": None,
                    "timestamp": ts,
                }
            )
        return rows

    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t not in df.columns.levels[0]:
                rows.append(
                    {
                        "pair": t,
                        "last": None,
                        "change": None,
                        "change_pct": None,
                        "timestamp": ts,
                    }
                )
                continue
            sub = df[t].dropna()
            if sub.empty or "Close" not in sub.columns:
                rows.append(
                    {
                        "pair": t,
                        "last": None,
                        "change": None,
                        "change_pct": None,
                        "timestamp": ts,
                    }
                )
                continue
            last = float(sub["Close"].iloc[-1])
            prev = float(sub["Close"].iloc[-2]) if len(sub) > 1 else last
            change = last - prev
            change_pct = (change / prev * 100.0) if prev else 0.0
            rows.append(
                {
                    "pair": t,
                    "last": round(last, 6),
                    "change": round(change, 6),
                    "change_pct": round(change_pct, 3),
                    "timestamp": ts,
                }
            )
    else:
        # Single-level columns: treat as one ticker and pad others
        t = tickers[0] if tickers else "FX"
        if "Close" in df.columns:
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            change = last - prev
            change_pct = (change / prev * 100.0) if prev else 0.0
            rows.append(
                {
                    "pair": t,
                    "last": round(last, 6),
                    "change": round(change, 6),
                    "change_pct": round(change_pct, 3),
                    "timestamp": ts,
                }
            )
        for t2 in tickers[1:]:
            rows.append(
                {
                    "pair": t2,
                    "last": None,
                    "change": None,
                    "change_pct": None,
                    "timestamp": ts,
                }
            )
    return rows


def _download_history(pair: str) -> pd.DataFrame:
    try:
        df = yf.download(
            pair, period="2y", interval="1d", progress=False, auto_adjust=True
        )
    except Exception:
        df = yf.download(
            pair, period="1y", interval="1d", progress=False, auto_adjust=True
        )
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime", "Close"])
    # If yfinance returns multi-index columns, select the pair slice
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(pair, axis=1, level=0)
        except Exception:
            # Fallback: take the first level if exact pair not found
            first = df.columns.levels[0][0]
            df = df.xs(first, axis=1, level=0)
    df = df.reset_index()
    if "Datetime" not in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df.rename(columns={num_cols[-1]: "Close"}, inplace=True)
    return df[["Datetime", "Close"]]

def _fetch_treasuries() -> dict:
    key = "treasuries"
    cached = _cache_get(key, ttl=600)
    if cached:
        return cached
    rows = []
    for ticker, label in TREASURY_TICKERS.items():
        try:
            df = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=True)
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty or "Close" not in df.columns:
            rows.append({"tenor": label, "last": None, "chg": None, "chg_pct": None})
            continue
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        chg = last - prev
        chg_pct = (chg / prev) * 100 if prev else 0.0
        rows.append(
            {
                "tenor": label,
                "last": round(last, 2),
                "chg": round(chg, 2),
                "chg_pct": round(chg_pct, 2),
            }
        )

    # Simple FX read from 10Y move
    fx_hint = "Mixed / neutral"
    t10 = next((r for r in rows if r["tenor"] == "10Y"), None)
    if t10 and t10["chg"] is not None:
        if t10["chg"] > 0:
            fx_hint = "USD bid · Risk-off tilt"
        elif t10["chg"] < 0:
            fx_hint = "USD softer · Risk-on tilt"
    payload = {"rows": rows, "fx_hint": fx_hint}
    _cache_set(key, payload, ttl=600)
    return payload

def _fetch_commodities() -> dict:
    key = "commodities"
    cached = _cache_get(key, ttl=300)
    if cached:
        return cached
    rows = []
    for ticker, label in COMMODITY_TICKERS.items():
        try:
            df = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=True)
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty or "Close" not in df.columns:
            rows.append({"name": label, "last": None, "chg": None, "chg_pct": None})
            continue
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        chg = last - prev
        chg_pct = (chg / prev) * 100 if prev else 0.0
        rows.append(
            {
                "name": label,
                "last": round(last, 2),
                "chg": round(chg, 2),
                "chg_pct": round(chg_pct, 2),
            }
        )
    payload = {"rows": rows}
    _cache_set(key, payload, ttl=300)
    return payload



def _load_signals() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            pass
    return pd.DataFrame(
        [
            {
                "source": "Sample",
                "headline": "Federal Reserve officials warn inflation remains persistent",
                "topic": "Inflation / Monetary Policy",
                "sentiment": "negative",
                "entities": "Federal Reserve, USD",
                "macro_interpretation": "Hawkish Fed tone",
                "currency_bias": "USD bullish",
                "trade_suggestion": "Short EUR/USD",
                "signal_confidence": 0.78,
            }
        ]
    )

def _fallback_story_items(df: pd.DataFrame, limit: int = 6) -> List[dict]:
    items = []
    if df is None or df.empty:
        return items
    df = df.copy()
    if "events" in df.columns:
        df = df[df["events"].fillna("").astype(str).str.len() > 0]
    if "trade_suggestion" in df.columns:
        df = df[df["trade_suggestion"].fillna("").str.lower() != "no trade"]
    if "signal_confidence" in df.columns:
        df = df.sort_values(by="signal_confidence", ascending=False)
    for _, r in df.head(limit).iterrows():
        bias = str(r.get("currency_bias", ""))
        ccy = next((k for k in FX_GEO.keys() if k in bias), "USD")
        items.append(
            {
                "ccy": ccy,
                "title": r.get("topic", "FX Signal"),
                "summary": r.get("macro_interpretation", "")[:180]
                or r.get("headline", "")[:180],
            }
        )
    return items

def _gemini_story_items(df: pd.DataFrame, limit: int = 6) -> List[dict]:
    if not _gemini_call or df is None or df.empty:
        return []
    rows = []
    for _, r in df.head(10).iterrows():
        rows.append(
            f"- {r.get('headline','')}; topic={r.get('topic','')}; bias={r.get('currency_bias','')}; trade={r.get('trade_suggestion','')}"
        )
    prompt = (
        "You are an FX briefing assistant. Create a story-mode feed of up to "
        f"{limit} beats based on these signals. Do NOT copy headlines verbatim. "
        "Synthesize higher-level insights. If bond yields or U.S. Treasuries "
        "are relevant, include a concise 'Treasury/Bonds buzz' beat and relate "
        "it to USD or risk sentiment.\n"
        "Output each line as:\n"
        "CCY|TITLE|SUMMARY\n"
        "Where CCY is one of: USD, EUR, GBP, JPY, AUD, INR, CNY.\n"
        "Keep TITLE under 6 words, SUMMARY under 20 words.\n"
        "Signals:\n"
        + "\n".join(rows)
    )
    text = _gemini_call(prompt, temperature=0.3) or ""
    items = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            continue
        ccy, title, summary = parts
        if ccy not in FX_GEO:
            continue
        items.append({"ccy": ccy, "title": title, "summary": summary})
    return items[:limit]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "pairs": PAIRS,
            "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


@app.get("/api/pairs")
def api_pairs():
    return JSONResponse({"pairs": PAIRS, "tickers": _fetch_yahoo_fx_tickers()})


@app.get("/api/live")
def api_live(limit: int = Query(20, ge=1, le=50)):
    key = "live"
    cached = _cache_get(key, ttl=50)
    if cached:
        return JSONResponse(cached)
    tickers = _fetch_yahoo_fx_tickers()
    df = _download_multi(tickers)
    rows = _build_live_rows(df, tickers)
    if not rows:
        rows = [
            {
                "pair": t,
                "last": None,
                "change": None,
                "change_pct": None,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            for t in tickers
        ]
    rows = rows[:limit]
    payload = {"last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), "rows": rows}
    _cache_set(key, payload, ttl=50)
    return JSONResponse(payload)


@app.get("/api/history")
def api_history(pair: str = Query("EURUSD=X")):
    key = f"hist:{pair}"
    cached = _cache_get(key, ttl=900)
    if cached:
        return JSONResponse(cached)
    df = _download_history(pair)
    rows = []
    if "Close" in df.columns:
        for _, r in df.iterrows():
            v = r["Close"]
            if pd.notna(v):
                rows.append({"t": str(r["Datetime"]), "v": float(v)})
    if not rows:
        # Fallback synthetic series for demo
        base = 1.0 if "USDJPY" not in pair else 140.0
        today = datetime.utcnow()
        rows = []
        for i in range(365 * 2):
            t = today - timedelta(days=(365 * 2 - i))
            drift = (i / 730.0) * 0.05
            val = base + drift
            rows.append({"t": t.strftime("%Y-%m-%d"), "v": round(val, 4)})
    payload = {"pair": pair, "rows": rows}
    _cache_set(key, payload, ttl=900)
    return JSONResponse(payload)


@app.get("/api/signals")
def api_signals():
    key = "signals"
    cached = _cache_get(key, ttl=120)
    if cached:
        return JSONResponse(cached)
    df = _load_signals()
    # De-dup display: collapse near-identical headlines and repeated trades
    if "headline" in df.columns:
        df["_hl_key"] = (
            df["headline"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    if "trade_suggestion" in df.columns:
        df["_trade_key"] = df["trade_suggestion"].fillna("").astype(str)
    if "topic" in df.columns:
        df["_topic_key"] = df["topic"].fillna("").astype(str).str.lower()

    if "events" in df.columns:
        df["_has_event"] = df["events"].fillna("").astype(str).str.len() > 0
        df = df.sort_values(by=["_has_event", "signal_confidence"], ascending=False)
        df = df.drop(columns=["_has_event"], errors="ignore")
    elif "signal_confidence" in df.columns:
        df = df.sort_values(by="signal_confidence", ascending=False)

    # Drop duplicate headlines
    if "_hl_key" in df.columns:
        df = df.drop_duplicates(subset=["_hl_key"])

    # Drop duplicate trades by topic (keep highest confidence)
    if "_trade_key" in df.columns:
        df = df.drop_duplicates(subset=["_trade_key", "_topic_key"], keep="first")

    df = df.drop(columns=["_hl_key", "_trade_key", "_topic_key"], errors="ignore")
    df = df.head(12)
    rows = df.fillna("").to_dict(orient="records")
    payload = {"rows": rows}
    _cache_set(key, payload, ttl=120)
    return JSONResponse(payload)

@app.get("/api/story")
def api_story(limit: int = Query(6, ge=3, le=10)):
    df = _load_signals()
    # Prefer Gemini, fallback to heuristic
    items = _gemini_story_items(df, limit=limit)
    if not items:
        items = _fallback_story_items(df, limit=limit)
    return JSONResponse({"items": items})

@app.get("/api/treasuries")
def api_treasuries():
    return JSONResponse(_fetch_treasuries())


@app.get("/api/commodities")
def api_commodities():
    return JSONResponse(_fetch_commodities())




@app.get("/api/fx_brief")
def api_fx_brief():
    df = _load_signals()
    if df is None or df.empty:
        return JSONResponse(
            {"title": "FX Brief", "summary": "No signals available yet.", "events": []}
        )
    df = df.head(12).copy()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"- {r.get('headline','')}; topic={r.get('topic','')}; bias={r.get('currency_bias','')}; trade={r.get('trade_suggestion','')}"
        )
    # Heuristic: extract likely scheduled events from headlines
    event_keywords = [
        "cpi", "inflation", "jobs", "nfp", "payroll", "rate decision", "meeting",
        "minutes", "press conference", "gdp", "speech", "testimony", "pmi",
    ]
    events = []
    for _, r in df.iterrows():
        h = str(r.get("headline", "")).strip()
        h_l = h.lower()
        if any(k in h_l for k in event_keywords):
            events.append(h)
        if len(events) >= 3:
            break

    if _gemini_call:
        prompt = (
            "Summarize how the FX market looks today as a short paragraph (6-7 sentences). "
            "Highlight USD, EUR, JPY, GBP, AUD tone if present.\n"
            "Mention the potential reasons/indicators for this market behavior (e.g., risk-off, inflation, rates, growth).\n"
            "Use the signals below as evidence. Avoid copying headlines.\n"
            "If there are notable scheduled events today, add one short sentence "
            "starting with 'Events:' listing up to 3.\n"
            + "\n".join(rows)
        )
        text = _gemini_call(prompt, temperature=0.3)
        if text:
            return JSONResponse({"title": "FX Brief", "summary": text.strip(), "events": events})

    # Fallback summary from net bias counts
    counts = {}
    for v in df["currency_bias"].fillna("").astype(str):
        for ccy in FX_GEO.keys():
            if f"{ccy} bullish" in v:
                counts[ccy] = counts.get(ccy, 0) + 1
            if f"{ccy} bearish" in v:
                counts[ccy] = counts.get(ccy, 0) - 1
    sorted_ccy = sorted(counts.items(), key=lambda x: abs(x[1]), reverse=True)
    top_parts = []
    for ccy, score in sorted_ccy[:3]:
        tone = "bullish" if score > 0 else "bearish" if score < 0 else "mixed"
        top_parts.append(f"{ccy} {tone}")
    tone_line = "Market tone: " + ", ".join(top_parts) if top_parts else "Market tone mixed."

    topic_counts = df["topic"].fillna("").astype(str).str.lower().value_counts()
    top_topic = topic_counts.index[0].title() if len(topic_counts) > 0 else "Macro"
    risk_flag = "risk" in " ".join(df["topic"].fillna("").astype(str)).lower()
    second_line = (
        f"Dominant theme: {top_topic}. "
        + ("Risk-off signals are elevated." if risk_flag else "Risk sentiment is balanced.")
    )

    trades = df["trade_suggestion"].fillna("").astype(str)
    trade_count = (trades.str.lower() != "no trade").sum()
    third_line = f"Actionable signals: {trade_count} today, focused on majors."

    drivers = (
        df["macro_interpretation"]
        .fillna("")
        .astype(str)
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .index[:3]
        .tolist()
    )
    drivers_line = ""
    if drivers:
        drivers_line = "Drivers: " + "; ".join(drivers) + "."

    if events:
        event_line = "Events: " + "; ".join(events[:3]) + "."
    else:
        event_line = "Events: No major releases detected in feeds."

    summary = " ".join([tone_line, second_line, drivers_line, third_line, event_line]).strip()
    return JSONResponse({"title": "FX Brief", "summary": summary, "events": events})


@app.get("/api/map_points")
def api_map_points():
    df = _load_signals()
    points = []
    # Count biases
    counts = {k: 0 for k in FX_GEO.keys()}
    if "currency_bias" in df.columns:
        for v in df["currency_bias"].fillna(""):
            for ccy in counts.keys():
                if ccy in str(v):
                    counts[ccy] += 1

    for ccy, meta in FX_GEO.items():
        points.append(
            {
                "label": f"{ccy} signals: {counts[ccy]}",
                "lat": meta["lat"],
                "lon": meta["lon"],
                "color": meta["color"],
                "size": 6 + counts[ccy] * 2,
            }
        )
    return JSONResponse({"points": points})


@app.get("/api/country_news")
def api_country_news(limit: int = Query(5, ge=1, le=10)):
    data = _fetch_country_news(limit_per_ccy=limit)
    return JSONResponse({"news": data})


@app.get("/api/ais_tankers")
def api_ais_tankers(limit: int = Query(300, ge=50, le=2000)):
    with _ais_lock:
        vals = list(_ais_tankers.values())[:limit]
    return JSONResponse({"tankers": vals})




@app.get("/api/youtube_live")
def api_youtube_live(channel: str = Query(...)):
    if channel.lower() == "bloomberg":
        return JSONResponse(
            {
                "handle": "Bloomberg",
                "video_id": "iEpJwprxDdk",
                "embed_url": "https://www.youtube.com/embed/iEpJwprxDdk",
                "live_url": "https://www.youtube.com/watch?v=iEpJwprxDdk",
            }
        )
    handle = YOUTUBE_LIVE_HANDLES.get(channel.lower())
    if not handle:
        return JSONResponse({"error": "unknown channel"}, status_code=400)
    return JSONResponse(_fetch_youtube_live_embed(handle))


@app.get("/api/alert_banner")
def api_alert_banner():
    """
    Returns a single-line geopolitical headline (best-effort from signals).
    Falls back to "No geopolitical alerts".
    """
    df = _load_signals()
    if "topic" in df.columns:
        geo = df[df["topic"].str.contains("risk|geopolit", case=False, na=False)]
    else:
        geo = df
    if not geo.empty:
        row = geo.head(1).to_dict(orient="records")[0]
        text = row.get("headline") or row.get("gemini_rationale") or ""
        return JSONResponse({"text": text})
    return JSONResponse({"text": "No geopolitical alerts."})
