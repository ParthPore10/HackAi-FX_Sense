"""
FXSense signal generation layer (MVP)
Maps NLP insights into currency bias and trade suggestions.

Usage:
  python fxsense/signals/generate_signal.py --input fxsense/data/analyzed_latest.csv
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.gemini_client import refine_signal
except Exception:
    refine_signal = lambda _text: None


HAWKISH_TERMS = {"hawkish", "tighten", "rate hike", "higher rates", "inflation persistent"}
DOVISH_TERMS = {"dovish", "rate cut", "lower rates", "easing", "slowdown"}
RISK_OFF_TERMS = {"risk-off", "geopolitical", "conflict", "war", "sanctions"}
RISK_ON_TERMS = {"risk-on", "optimism", "growth", "expansion", "rally"}


CENTRAL_BANK_MAP = {
    "Federal Reserve": "USD",
    "Fed": "USD",
    "ECB": "EUR",
    "European Central Bank": "EUR",
    "Bank of England": "GBP",
    "BoE": "GBP",
    "Bank of Japan": "JPY",
    "BoJ": "JPY",
    "RBA": "AUD",
    "Reserve Bank of Australia": "AUD",
}


PAIR_MAP = {
    "USD": ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X"],
    "EUR": ["EURUSD=X"],
    "GBP": ["GBPUSD=X"],
    "JPY": ["USDJPY=X"],
    "AUD": ["AUDUSD=X"],
}


def _score_text(text: str, terms: set) -> int:
    text_l = text.lower()
    return sum(1 for t in terms if t in text_l)


def _entity_currency_bias(entities: str) -> List[str]:
    bias = []
    for ent, ccy in CENTRAL_BANK_MAP.items():
        if ent.lower() in entities.lower():
            bias.append(ccy)
    return bias


def _infer_bias(row: pd.Series) -> Tuple[List[str], str]:
    text = f"{row.get('headline', '')}. {row.get('summary', '')}".strip()
    sentiment = str(row.get("sentiment", "")).lower()
    topic = str(row.get("topic", "")).lower()
    entities = str(row.get("entities", ""))

    hawkish = _score_text(text, HAWKISH_TERMS)
    dovish = _score_text(text, DOVISH_TERMS)
    risk_off = _score_text(text, RISK_OFF_TERMS)
    risk_on = _score_text(text, RISK_ON_TERMS)

    ccy_bias = _entity_currency_bias(entities)
    reasons = []
    events = str(row.get("events", "")).lower()

    if hawkish > dovish:
        reasons.append("Hawkish tone")
    elif dovish > hawkish:
        reasons.append("Dovish tone")

    if "rate hike" in events:
        reasons.append("Rate hike")
    if "rate cut" in events:
        reasons.append("Rate cut")
    if "inflation surprise" in events:
        reasons.append("Inflation surprise")
    if "growth miss" in events:
        reasons.append("Growth miss")
    if "risk-off shock" in events:
        reasons.append("Risk-off shock")

    if "inflation" in topic and hawkish >= dovish:
        reasons.append("Inflation pressure")
    if "risk" in topic and risk_off > risk_on:
        reasons.append("Risk-off")

    # Default biases if not enough explicit signals
    if not ccy_bias:
        if "fed" in text.lower():
            ccy_bias.append("USD")
        if "ecb" in text.lower():
            ccy_bias.append("EUR")
        if "boe" in text.lower():
            ccy_bias.append("GBP")
        if "boj" in text.lower():
            ccy_bias.append("JPY")
        if "rba" in text.lower():
            ccy_bias.append("AUD")

    # Resolve final bias
    bias = []
    for ccy in ccy_bias:
        if dovish > hawkish:
            bias.append(f"{ccy} bearish")
        elif hawkish > dovish:
            bias.append(f"{ccy} bullish")
        else:
            # sentiment can tip
            if sentiment == "positive":
                bias.append(f"{ccy} bullish")
            elif sentiment == "negative":
                bias.append(f"{ccy} bearish")
            else:
                bias.append(f"{ccy} neutral")

    if not bias:
        bias = ["Neutral"]
    reason = ", ".join(reasons) if reasons else "General macro tone"
    return bias, reason


def _trade_suggestion(bias: List[str]) -> str:
    # Simple mapping to major pairs
    # If USD bullish -> short EURUSD/GBPUSD/AUDUSD, long USDJPY
    if any("USD bullish" in b for b in bias):
        return "Short EUR/USD, Short GBP/USD, Short AUD/USD, Long USD/JPY"
    if any("USD bearish" in b for b in bias):
        return "Long EUR/USD, Long GBP/USD, Long AUD/USD, Short USD/JPY"
    if any("EUR bullish" in b for b in bias):
        return "Long EUR/USD"
    if any("EUR bearish" in b for b in bias):
        return "Short EUR/USD"
    if any("GBP bullish" in b for b in bias):
        return "Long GBP/USD"
    if any("GBP bearish" in b for b in bias):
        return "Short GBP/USD"
    if any("JPY bullish" in b for b in bias):
        return "Short USD/JPY"
    if any("JPY bearish" in b for b in bias):
        return "Long USD/JPY"
    if any("AUD bullish" in b for b in bias):
        return "Long AUD/USD"
    if any("AUD bearish" in b for b in bias):
        return "Short AUD/USD"
    return "No trade"


def _confidence(row: pd.Series, bias: List[str]) -> float:
    base = float(row.get("sentiment_confidence", 0.5))
    topic = str(row.get("topic", "")).lower()
    text = f"{row.get('headline', '')}. {row.get('summary', '')}".lower()
    events = str(row.get("events", "")).lower()
    boost = 0.0
    if "monetary policy" in topic:
        boost += 0.1
    if any(k in text for k in ["rate", "rates", "inflation", "cpi"]):
        boost += 0.1
    if events:
        boost += 0.15
    if any("bullish" in b or "bearish" in b for b in bias):
        boost += 0.05
    return float(np.clip(base + boost, 0.1, 0.95))


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, row in df.iterrows():
        bias, reason = _infer_bias(row)
        events = str(row.get("events", "")).strip()
        trade = _trade_suggestion(bias) if events else "No trade"
        conf = _confidence(row, bias)
        text = f"{row.get('headline', '')}. {row.get('summary', '')}".strip()
        gemini_rationale = refine_signal(text) if text else None
        out_rows.append(
            {
                **row.to_dict(),
                "macro_interpretation": reason,
                "currency_bias": "; ".join(bias),
                "trade_suggestion": trade,
                "signal_confidence": round(conf, 3),
                "gemini_rationale": gemini_rationale,
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="FXSense signal generator (MVP)")
    parser.add_argument(
        "--input",
        default="fxsense/data/analyzed_latest.csv",
        help="Path to analyzed CSV",
    )
    parser.add_argument(
        "--output",
        default="fxsense/data/signals_latest.csv",
        help="Path to write signals CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = generate_signals(df)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} records to {args.output}")


if __name__ == "__main__":
    main()
