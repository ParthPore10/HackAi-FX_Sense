"""
FXSense signal generation layer (MVP)
Maps NLP insights into currency bias and trade suggestions.

Usage:
  python fxsense/signals/generate_signal.py --input fxsense/data/analyzed_latest.csv
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.gemini_client import refine_signal
except Exception:
    refine_signal = lambda _text: None

try:
    from utils.health_report import compute_health
except Exception:
    def compute_health(_df):
        return {}

def _log_signals_fallback(df: pd.DataFrame, log_path: str) -> None:
    import re
    from datetime import datetime, timezone

    pair_re = re.compile(r"([A-Z]{3})/([A-Z]{3})")
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for _, r in df.iterrows():
        trade = str(r.get("trade_suggestion", "")).strip()
        if not trade or trade.lower() == "no trade":
            continue
        m = pair_re.search(trade)
        pair = f"{m.group(1)}{m.group(2)}=X" if m else ""
        direction = "long" if "long" in trade.lower() else "short" if "short" in trade.lower() else ""
        rows.append(
            {
                "timestamp_utc": now,
                "source": r.get("source", ""),
                "headline": r.get("headline", ""),
                "events": r.get("events", ""),
                "bias": r.get("currency_bias", ""),
                "trade": trade,
                "pair": pair,
                "direction": direction,
                "confidence": r.get("signal_confidence", ""),
            }
        )
    if not rows:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_new = pd.DataFrame(rows)
    if os.path.exists(log_path):
        df_new.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(log_path, index=False)

def _mlflow_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_ENABLE"))


def _mlflow_run(name: str):
    from contextlib import contextmanager

    @contextmanager
    def _noop():
        yield None

    if not _mlflow_enabled():
        return _noop()
    try:
        import mlflow

        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "fxsense"))

        @contextmanager
        def _run():
            with mlflow.start_run(run_name=name):
                yield mlflow

        return _run()
    except Exception:
        return _noop()


def _mlflow_log_dict(mlf, d: dict, prefix: str = "metric"):
    if not mlf:
        return
    for k, v in d.items():
        try:
            mlf.log_metric(f"{prefix}.{k}", float(v))
        except Exception:
            continue


def _extract_pair_direction(trade: str) -> Tuple[str, str]:
    trade_l = trade.lower()
    direction = ""
    if "long" in trade_l:
        direction = "long"
    elif "short" in trade_l:
        direction = "short"
    pair = ""
    for token in ["EUR/USD", "GBP/USD", "AUD/USD", "USD/JPY", "NZD/USD", "USD/CHF"]:
        if token.lower() in trade_l:
            pair = token.replace("/", "") + "=X"
            break
    return pair, direction


def _score_with_return_model(df: pd.DataFrame) -> pd.Series:
    if os.environ.get("MODEL_SCORE") != "1":
        return pd.Series([np.nan] * len(df))
    model_path = os.environ.get("RETURN_LOGIT_MODEL", "data/return_logit_metrics.pkl")
    cols_path = os.environ.get("RETURN_LOGIT_COLS", "data/return_logit_metrics.json")
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        return pd.Series([np.nan] * len(df))
    try:
        import joblib
        with open(cols_path, "r", encoding="utf-8") as f:
            cols = json.load(f).get("columns", [])
        model = joblib.load(model_path)
    except Exception:
        return pd.Series([np.nan] * len(df))

    base = df.copy()
    base["sentiment"] = base.get("sentiment", "").fillna("").astype(str).str.lower()
    base["topic"] = base.get("topic", "").fillna("").astype(str).str.lower()
    base["has_events"] = base.get("events", "").fillna("").astype(str).str.len().gt(0).astype(int)
    base["confidence"] = pd.to_numeric(base.get("signal_confidence", 0.5), errors="coerce").fillna(0.5)

    pairs = []
    directions = []
    for t in base.get("trade_suggestion", "").fillna("").astype(str):
        pair, direction = _extract_pair_direction(t)
        pairs.append(pair)
        directions.append(direction)
    base["pair"] = pairs
    base["direction"] = directions

    feats = base[["confidence", "has_events"]].copy()
    for col in ["sentiment", "topic", "direction", "pair"]:
        feats = feats.join(pd.get_dummies(base[col], prefix=col))
    # align columns
    for c in cols:
        if c not in feats.columns:
            feats[c] = 0
    feats = feats[cols]

    try:
        probs = model.predict_proba(feats)[:, 1]
        return pd.Series(probs)
    except Exception:
        return pd.Series([np.nan] * len(df))


HAWKISH_TERMS = {"hawkish", "tighten", "rate hike", "higher rates", "inflation persistent"}
DOVISH_TERMS = {"dovish", "rate cut", "lower rates", "easing", "slowdown"}
RISK_OFF_TERMS = {"risk-off", "geopolitical", "conflict", "war", "sanctions"}
RISK_ON_TERMS = {"risk-on", "optimism", "growth", "expansion", "rally"}

MAX_TRADES_PER_PAIR = int(os.environ.get("FXSENSE_MAX_TRADES_PER_PAIR", "3"))


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
    "NZD": ["NZDUSD=X"],
    "CHF": ["USDCHF=X"],
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

    risk_gate = (
        "risk-off shock" in events
        or risk_off >= 2
        or ("risk" in topic and risk_off > 0 and sentiment == "negative")
    )

    # Risk-off macro mapping (industry standard)
    if risk_gate:
        ccy_bias.extend(["JPY", "CHF", "AUD", "NZD"])

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

    # Override with risk-off safe-haven mapping
    if risk_gate:
        risk_bias = {
            "JPY": "bullish",
            "CHF": "bullish",
            "AUD": "bearish",
            "NZD": "bearish",
        }
        for ccy, direction in risk_bias.items():
            bias.append(f"{ccy} {direction}")

    if not bias:
        bias = ["Neutral"]
    reason = ", ".join(reasons) if reasons else "General macro tone"
    return bias, reason


def _trade_suggestion(bias_map: Dict[str, str]) -> str:
    # Simple mapping to major pairs
    # If USD bullish -> short EURUSD/GBPUSD/AUDUSD, long USDJPY
    if bias_map.get("USD") == "bullish":
        return "Short EUR/USD, Short GBP/USD, Short AUD/USD, Long USD/JPY"
    if bias_map.get("USD") == "bearish":
        return "Long EUR/USD, Long GBP/USD, Long AUD/USD, Short USD/JPY"
    if bias_map.get("EUR") == "bullish":
        return "Long EUR/USD"
    if bias_map.get("EUR") == "bearish":
        return "Short EUR/USD"
    if bias_map.get("GBP") == "bullish":
        return "Long GBP/USD"
    if bias_map.get("GBP") == "bearish":
        return "Short GBP/USD"
    if bias_map.get("JPY") == "bullish":
        return "Short USD/JPY"
    if bias_map.get("JPY") == "bearish":
        return "Long USD/JPY"
    if bias_map.get("AUD") == "bullish":
        return "Long AUD/USD"
    if bias_map.get("AUD") == "bearish":
        return "Short AUD/USD"
    if bias_map.get("NZD") == "bullish":
        return "Long NZD/USD"
    if bias_map.get("NZD") == "bearish":
        return "Short NZD/USD"
    if bias_map.get("CHF") == "bullish":
        return "Short USD/CHF"
    if bias_map.get("CHF") == "bearish":
        return "Long USD/CHF"
    return "No trade"

def _normalize_bias(bias: List[str]) -> Tuple[List[str], Dict[str, str], str]:
    counts: Dict[str, Dict[str, int]] = {}
    for b in bias:
        parts = b.split()
        if len(parts) < 2:
            continue
        ccy = parts[0].upper()
        direction = parts[1].lower()
        if ccy not in counts:
            counts[ccy] = {"bullish": 0, "bearish": 0, "neutral": 0}
        if direction in counts[ccy]:
            counts[ccy][direction] += 1

    net_map: Dict[str, str] = {}
    net_list: List[str] = []
    for ccy, d in counts.items():
        if d["bullish"] > d["bearish"]:
            net = "bullish"
        elif d["bearish"] > d["bullish"]:
            net = "bearish"
        else:
            net = "neutral"
        net_map[ccy] = net
        net_list.append(f"{ccy} {net}")

    summary = "; ".join(net_list) if net_list else "Neutral"
    return net_list if net_list else ["Neutral"], net_map, summary

def _session_active() -> bool:
    # Use UTC to detect main FX sessions: Asia (0-9), Europe (7-16), US (13-22)
    h = datetime.now(timezone.utc).hour
    return (0 <= h < 9) or (7 <= h < 16) or (13 <= h < 22)


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
    df = df.copy()
    if "headline" in df.columns:
        df["_norm_headline"] = (
            df["headline"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df = df.drop_duplicates(subset=["_norm_headline"])

    out_rows = []
    for _, row in df.iterrows():
        bias, reason = _infer_bias(row)
        norm_bias, bias_map, bias_summary = _normalize_bias(bias)
        events = str(row.get("events", "")).strip()
        conf = _confidence(row, norm_bias)
        trade = _trade_suggestion(bias_map) if events else "No trade"
        # Confidence + session gate
        if conf < 0.6 or not _session_active():
            trade = "No trade"
        text = f"{row.get('headline', '')}. {row.get('summary', '')}".strip()
        gemini_rationale = refine_signal(text) if text else None
        out_rows.append(
            {
                **row.to_dict(),
                "macro_interpretation": reason,
                "currency_bias": bias_summary,
                "trade_suggestion": trade,
                "signal_confidence": round(conf, 3),
                "gemini_rationale": gemini_rationale,
            }
        )

    out = pd.DataFrame(out_rows)

    # Optional model probability scoring
    out["model_prob"] = _score_with_return_model(out)

    # Cap repeated trades so one pair doesn't dominate the board
    mask_trade = out["trade_suggestion"].fillna("").ne("No trade")
    if mask_trade.any():
        out["_rank_trade"] = out.loc[mask_trade].groupby("trade_suggestion")[
            "signal_confidence"
        ].rank(ascending=False, method="first")
        out.loc[
            mask_trade & (out["_rank_trade"] > MAX_TRADES_PER_PAIR),
            "trade_suggestion",
        ] = "No trade"
        out.drop(columns=["_rank_trade"], inplace=True)

        # Single-source dampener: require higher confidence if only one trade instance exists
        trade_counts = (
            out.loc[out["trade_suggestion"].ne("No trade")]
            .groupby("trade_suggestion")["trade_suggestion"]
            .transform("count")
        )
        out.loc[
            (out["trade_suggestion"].ne("No trade"))
            & (trade_counts == 1)
            & (out["signal_confidence"] < 0.8),
            "trade_suggestion",
        ] = "No trade"

    return out


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

    # Optional: log signals for downstream modeling/evaluation
    if os.environ.get("SIGNAL_LOG") == "1":
        log_path = os.environ.get("SIGNAL_LOG_PATH", "data/signal_log.csv")
        _log_signals_fallback(out, log_path)

    with _mlflow_run("signal_generate") as mlf:
        if mlf:
            mlf.log_param("input", args.input)
            mlf.log_param("output", args.output)
            _mlflow_log_dict(
                mlf,
                {
                    "rows": len(out),
                    "trades": (out["trade_suggestion"] != "No trade").sum(),
                    "events_tagged": out["events"].fillna("").astype(str).str.len().gt(0).sum(),
                    "avg_conf": out["signal_confidence"].mean(),
                },
                prefix="signals",
            )
            # Health metrics
            health = compute_health(out)
            _mlflow_log_dict(mlf, health, prefix="health")
    print(f"Wrote {len(out)} records to {args.output}")



if __name__ == "__main__":
    main()