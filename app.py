"""
FXSense Streamlit app (MVP)
Live FX monitor (auto-refresh every N seconds).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser
import streamlit.components.v1 as components


REFRESH_SECONDS = 60
PAIRS: List[str] = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
YAHOO_FX_TICKERS: List[str] = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "USDCHF=X",
    "USDCAD=X",
    "NZDUSD=X",
    "EURJPY=X",
    "EURGBP=X",
    "EURAUD=X",
    "EURCHF=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "AUDNZD=X",
    "CADJPY=X",
    "CHFJPY=X",
    "GBPCHF=X",
    "NZDJPY=X",
    "AUDCAD=X",
    "EURNZD=X",
]
PROJECT_ROOT = Path(__file__).resolve().parent
CNBC_RSS = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"


def _download_pair(pair: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            pair, period=period, interval=interval, progress=False, auto_adjust=True
        )
    except Exception:
        df = yf.download(pair, period="5d", interval="5m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime", "Close"])
    df = df.reset_index()
    if "Datetime" not in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    df = df[["Datetime", "Close"]].tail(300)
    return df


def _build_fx_table(cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    for pair, df in cache.items():
        if df.empty:
            rows.append(
                {
                    "Pair": pair,
                    "Last": None,
                    "Change": None,
                    "Change %": None,
                    "Timestamp": ts,
                }
            )
            continue
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        change = last - prev
        change_pct = (change / prev * 100.0) if prev else 0.0
        rows.append(
            {
                "Pair": pair,
                "Last": round(last, 6),
                "Change": round(change, 6),
                "Change %": round(change_pct, 3),
                "Timestamp": ts,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=55)
def refresh_cache() -> Tuple[Dict[str, pd.DataFrame], str]:
    cache = {}
    for pair in PAIRS:
        cache[pair] = _download_pair(pair, period="1d", interval="1m")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return cache, ts


@st.cache_data(ttl=55)
def load_live_tickers(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Pair", "Last", "Change", "Change %", "Timestamp"])
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
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = []
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t not in df.columns.levels[0]:
                rows.append(
                    {
                        "Pair": t,
                        "Last": None,
                        "Change": None,
                        "Change %": None,
                        "Timestamp": ts,
                    }
                )
                continue
            sub = df[t].dropna()
            if sub.empty or "Close" not in sub.columns:
                rows.append(
                    {
                        "Pair": t,
                        "Last": None,
                        "Change": None,
                        "Change %": None,
                        "Timestamp": ts,
                    }
                )
                continue
            last = float(sub["Close"].iloc[-1])
            prev = float(sub["Close"].iloc[-2]) if len(sub) > 1 else last
            change = last - prev
            change_pct = (change / prev * 100.0) if prev else 0.0
            rows.append(
                {
                    "Pair": t,
                    "Last": round(last, 6),
                    "Change": round(change, 6),
                    "Change %": round(change_pct, 3),
                    "Timestamp": ts,
                }
            )
    else:
        # Single ticker fallback
        sub = df.dropna()
        if not sub.empty and "Close" in sub.columns:
            last = float(sub["Close"].iloc[-1])
            prev = float(sub["Close"].iloc[-2]) if len(sub) > 1 else last
            change = last - prev
            change_pct = (change / prev * 100.0) if prev else 0.0
            rows.append(
                {
                    "Pair": tickers[0],
                    "Last": round(last, 6),
                    "Change": round(change, 6),
                    "Change %": round(change_pct, 3),
                    "Timestamp": ts,
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def load_history(pair: str) -> pd.DataFrame:
    return _download_pair(pair, period="2y", interval="1d")


@st.cache_data(ttl=300)
def load_cnbc_headlines(limit: int = 15) -> pd.DataFrame:
    feed = feedparser.parse(CNBC_RSS)
    rows = []
    for entry in feed.entries[:limit]:
        rows.append(
            {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            }
        )
    return pd.DataFrame(rows)


def _load_signals() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "signals_latest.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    # Fallback sample
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
            },
            {
                "source": "Sample",
                "headline": "ECB highlights weaker growth outlook for euro area",
                "topic": "Growth",
                "sentiment": "negative",
                "entities": "ECB, EUR",
                "macro_interpretation": "Soft growth outlook",
                "currency_bias": "EUR bearish",
                "trade_suggestion": "Short EUR/USD",
                "signal_confidence": 0.71,
            },
        ]
    )


def _summarize_bias(df: pd.DataFrame) -> Dict[str, int]:
    counts = {"USD": 0, "EUR": 0, "GBP": 0, "JPY": 0, "AUD": 0}
    if "currency_bias" not in df.columns:
        return counts
    for v in df["currency_bias"].fillna(""):
        for ccy in counts.keys():
            if ccy in str(v):
                counts[ccy] += 1
    return counts


def main() -> None:
    st.set_page_config(page_title="FXSense Live FX", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
        .fx-bg {
          background: radial-gradient(1200px 600px at 80% -10%, #1b6bff22 0%, transparent 60%),
                      radial-gradient(900px 500px at 10% 0%, #00d4ff22 0%, transparent 55%),
                      linear-gradient(180deg, #0b0f17 0%, #0c1220 60%, #0a0f1a 100%);
          padding: 1.5rem 1.5rem 2.5rem 1.5rem;
          border-radius: 18px;
          border: 1px solid #1f2a44;
        }
        .fx-title {
          font-size: 2.2rem;
          font-weight: 700;
          letter-spacing: 0.5px;
        }
        .fx-sub {
          color: #9fb0c8;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.9rem;
        }
        .fx-card {
          background: #111828;
          border: 1px solid #1c2942;
          border-radius: 14px;
          padding: 1rem 1.2rem;
        }
        .fx-pill {
          display: inline-block;
          padding: 0.25rem 0.6rem;
          border-radius: 999px;
          background: #15223a;
          border: 1px solid #243455;
          font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="fx-bg">
          <div class="fx-title">FXSense — Live FX Intelligence</div>
          <div class="fx-sub">Auto-refreshing every {REFRESH_SECONDS}s · {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cache, last_update = refresh_cache()
    fx_table = _build_fx_table(cache)

    signals = _load_signals()
    bias_counts = _summarize_bias(signals)

    st.markdown("### Market Pulse")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("USD Mentions", bias_counts["USD"])
    m2.metric("EUR Mentions", bias_counts["EUR"])
    m3.metric("GBP Mentions", bias_counts["GBP"])
    m4.metric("JPY Mentions", bias_counts["JPY"])
    m5.metric("AUD Mentions", bias_counts["AUD"])

    st.markdown("### CNBC Live")
    components.iframe(
        "https://www.cnbc.com/live-tv/",
        height=700,
        scrolling=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Prices")
        st.dataframe(fx_table, use_container_width=True, height=260)
        st.subheader("Yahoo FX Tickers (Live)")
        live_df = load_live_tickers(YAHOO_FX_TICKERS)
        st.dataframe(live_df, use_container_width=True, height=320)
    with col2:
        st.subheader("Controls")
        selected_pair = st.selectbox("Select pair", PAIRS, index=0)
        if st.button("Refresh now"):
            refresh_cache.clear()
            cache, last_update = refresh_cache()

    st.markdown(f"<span class='fx-pill'>Last update: {last_update}</span>", unsafe_allow_html=True)

    st.subheader("Pair Charts (2Y)")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    cols = [c1, c2, c3, c4]
    for col, pair in zip(cols, PAIRS):
        with col:
            df_hist = load_history(pair)
            st.caption(pair)
            if not df_hist.empty:
                st.line_chart(df_hist.set_index("Datetime")["Close"])
            else:
                st.info("No data available.")

    st.markdown("### Top Signals (Latest)")
    if not signals.empty:
        top = signals.sort_values(
            by="signal_confidence", ascending=False
        ).head(8)
        st.dataframe(
            top[
                [
                    "source",
                    "headline",
                    "topic",
                    "sentiment",
                    "macro_interpretation",
                    "currency_bias",
                    "trade_suggestion",
                    "signal_confidence",
                ]
            ],
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No signals available yet. Run the NLP + signal pipeline.")

    # Auto-refresh
    st.markdown(
        f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
