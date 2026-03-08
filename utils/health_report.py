"""
Lightweight robustness/health report for signals.
"""

from __future__ import annotations

import pandas as pd


def compute_health(df: pd.DataFrame) -> dict:
    out = {}
    out["rows"] = len(df)
    out["trade_rate"] = float((df["trade_suggestion"] != "No trade").mean()) if "trade_suggestion" in df.columns else 0.0
    out["event_rate"] = float(df["events"].fillna("").astype(str).str.len().gt(0).mean()) if "events" in df.columns else 0.0
    out["sent_pos"] = int((df["sentiment"] == "positive").sum()) if "sentiment" in df.columns else 0
    out["sent_neg"] = int((df["sentiment"] == "negative").sum()) if "sentiment" in df.columns else 0
    out["sent_neu"] = int((df["sentiment"] == "neutral").sum()) if "sentiment" in df.columns else 0
    out["avg_conf"] = float(df["signal_confidence"].mean()) if "signal_confidence" in df.columns else 0.0
    return out
