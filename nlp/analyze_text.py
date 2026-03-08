"""
FXSense NLP analysis layer (MVP)
Rule-based topic detection + optional model-based sentiment + optional spaCy NER.

Usage:
  python fxsense/nlp/analyze_text.py --input fxsense/data/scraped_latest.csv
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.gemini_client import summarize
except Exception:
    summarize = lambda _text: None

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



# ---------------------------
# Topic detection (rule-based)
# ---------------------------
TOPIC_RULES: Dict[str, List[str]] = {
    "Monetary Policy": [
        "rate",
        "rates",
        "interest",
        "policy",
        "meeting",
        "fomc",
        "decision",
        "statement",
        "minutes",
        "hike",
        "cut",
        "tighten",
        "easing",
    ],
    "Inflation": [
        "inflation",
        "cpi",
        "prices",
        "price pressures",
        "pce",
        "deflator",
    ],
    "Growth": [
        "gdp",
        "growth",
        "recession",
        "slowdown",
        "soft landing",
        "expansion",
        "activity",
    ],
    "Labor Market": [
        "jobs",
        "employment",
        "unemployment",
        "wages",
        "labor",
        "payrolls",
    ],
    "FX / Markets": [
        "fx",
        "foreign exchange",
        "usd",
        "eur",
        "gbp",
        "jpy",
        "aud",
        "yen",
        "euro",
        "sterling",
        "dollar",
    ],
    "Risk / Geopolitics": [
        "risk-off",
        "geopolitical",
        "war",
        "conflict",
        "sanctions",
        "safe haven",
    ],
}

EVENT_RULES: Dict[str, List[str]] = {
    "Rate Hike": ["rate hike", "raises rates", "raised rates", "increase rates"],
    "Rate Cut": ["rate cut", "cuts rates", "cut rates", "lower rates"],
    "Inflation Surprise": ["cpi above", "inflation hotter", "price pressures"],
    "Inflation Easing": ["inflation cools", "disinflation", "price pressures ease"],
    "Growth Miss": ["gdp miss", "slowdown", "recession risk", "weak demand"],
    "Growth Beat": ["gdp beat", "strong growth", "expansion", "robust demand"],
    "Risk-Off Shock": ["geopolitical", "conflict", "war", "sanctions", "risk-off"],
}


def detect_events(text: str) -> List[str]:
    text_l = text.lower()
    hits = []
    for event, keywords in EVENT_RULES.items():
        if any(k in text_l for k in keywords):
            hits.append(event)
    return hits


def detect_topic(text: str) -> str:
    text_l = text.lower()
    scores = {}
    for topic, keywords in TOPIC_RULES.items():
        score = 0
        for k in keywords:
            if k in text_l:
                score += 1
        scores[topic] = score
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "Other"


# ---------------------------
# Sentiment analysis
# ---------------------------
POSITIVE_WORDS = {
    "improve",
    "strong",
    "accelerate",
    "optimistic",
    "hawkish",
    "tighten",
    "surge",
    "resilient",
    "upside",
    "beat",
    "above forecast",
    "upgrade",
    "expansion",
    "robust",
}
NEGATIVE_WORDS = {
    "weaken",
    "slow",
    "recession",
    "dovish",
    "cut",
    "risk",
    "uncertainty",
    "stress",
    "downside",
    "miss",
    "below forecast",
    "downgrade",
    "contraction",
    "fragile",
}

POS_PHRASES = {
    r"\brate hike\b": 1.5,
    r"\brates? higher\b": 1.2,
    r"\btightening\b": 1.2,
    r"\binflation persistent\b": 1.0,
    r"\bstrong growth\b": 1.1,
}
NEG_PHRASES = {
    r"\brate cut\b": 1.5,
    r"\brates? lower\b": 1.2,
    r"\bloosening\b": 1.0,
    r"\brecession risk\b": 1.2,
    r"\bweak demand\b": 1.0,
}


def _lexicon_sentiment(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_l)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_l)
    # Phrase weighting
    pos += sum(1 for p in POS_PHRASES if re.search(p, text_l))
    neg += sum(1 for p in NEG_PHRASES if re.search(p, text_l))
    score = (pos - neg) / max(1, pos + neg)
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"
    conf = min(0.9, 0.5 + abs(score))
    return label, float(conf)


_SENTIMENT_PIPELINE = None


def _get_sentiment_pipeline() -> Optional[object]:
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE
    try:
        from transformers import pipeline  # type: ignore

        model_name = os.environ.get("SENTIMENT_MODEL", "ProsusAI/finbert")
        _SENTIMENT_PIPELINE = pipeline(
            "text-classification", model=model_name, tokenizer=model_name, return_all_scores=True
        )
        return _SENTIMENT_PIPELINE
    except Exception:
        _SENTIMENT_PIPELINE = None
        return None


def _model_sentiment(text: str) -> Tuple[str, float]:
    try:
        nlp = _get_sentiment_pipeline()
        if not nlp:
            return _lexicon_sentiment(text)

        outputs = nlp(text[:512])[0]
        # outputs is list of dicts when return_all_scores=True
        best = max(outputs, key=lambda x: x.get("score", 0.0))
        label = str(best.get("label", "neutral")).lower()
        score = float(best.get("score", 0.5))

        if "pos" in label:
            return "positive", score
        if "neg" in label:
            return "negative", score
        if "neu" in label:
            return "neutral", score
        # FinBERT labels are often POSITIVE/NEGATIVE/NEUTRAL
        if label in {"positive", "negative", "neutral"}:
            return label, score
        return "neutral", score
    except Exception:
        return _lexicon_sentiment(text)


# ---------------------------
# Entity extraction (NER)
# ---------------------------
ENTITY_FALLBACK = [
    "Federal Reserve",
    "ECB",
    "Bank of England",
    "Bank of Japan",
    "RBA",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "Dollar",
    "Euro",
    "Yen",
    "Sterling",
]


def extract_entities(text: str) -> List[str]:
    # Try spaCy NER if available
    try:
        import spacy  # type: ignore

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        ents = list({ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE"}})
        if ents:
            return ents[:8]
    except Exception:
        pass

    # Fallback: simple keyword matching
    found = []
    for ent in ENTITY_FALLBACK:
        if re.search(rf"\b{re.escape(ent)}\b", text, flags=re.I):
            found.append(ent)
    return found[:8]


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = f"{row.get('headline', '')}. {row.get('summary', '')}".strip()
        topic = detect_topic(text)
        sentiment, sent_conf = _model_sentiment(text)
        entities = extract_entities(text)
        events = detect_events(text)
        gemini_summary = summarize(text) if text else None
        records.append(
            {
                **row.to_dict(),
                "topic": topic,
                "sentiment": sentiment,
                "sentiment_confidence": float(np.round(sent_conf, 3)),
                "entities": ", ".join(entities),
                "events": ", ".join(events),
                "gemini_summary": gemini_summary,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="FXSense NLP analysis (MVP)")
    parser.add_argument(
        "--input",
        default="fxsense/data/scraped_latest.csv",
        help="Path to scraped CSV",
    )
    parser.add_argument(
        "--output",
        default="fxsense/data/analyzed_latest.csv",
        help="Path to write analyzed CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = analyze_dataframe(df)
    out.to_csv(args.output, index=False)

    # Optional MLflow logging
    with _mlflow_run("nlp_analyze") as mlf:
        if mlf:
            mlf.log_param("input", args.input)
            mlf.log_param("output", args.output)
            _mlflow_log_dict(
                mlf,
                {
                    "rows": len(out),
                    "sent_pos": (out["sentiment"] == "positive").sum(),
                    "sent_neg": (out["sentiment"] == "negative").sum(),
                    "sent_neu": (out["sentiment"] == "neutral").sum(),
                    "events_tagged": out["events"].fillna("").astype(str).str.len().gt(0).sum(),
                },
                prefix="nlp",
            )
    print(f"Wrote {len(out)} records to {args.output}")


if __name__ == "__main__":
    main()
