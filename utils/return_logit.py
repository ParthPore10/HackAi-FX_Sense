"""
Train a lightweight logistic model to predict 1-day return direction
from signal features.

Usage:
  python utils/return_logit.py --log data/signal_log.csv --out data/return_logit_metrics.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def _parse_ts(x: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return pd.NaT


def _fetch_prices(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.tz_localize(timezone.utc)
    if start.tzinfo is None:
        start = start.tz_localize(timezone.utc)
    end = min(end, now)
    start_date = start.date()
    end_date = (end + timedelta(days=2)).date()
    df = yf.download(
        pair,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime", "Close"])
    df = df.reset_index()
    if "Date" in df.columns and "Datetime" not in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
    return df[["Datetime", "Close"]]


def _label_returns(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp_utc"].apply(_parse_ts)
    df = df.dropna(subset=["timestamp", "pair", "direction"])
    if df.empty:
        return df

    start = df["timestamp"].min()
    end = df["timestamp"].max()

    labeled = []
    for pair, g in df.groupby("pair"):
        prices = _fetch_prices(pair, start, end)
        if prices.empty:
            continue
        prices["Datetime"] = pd.to_datetime(prices["Datetime"], utc=True)
        prices = prices.sort_values("Datetime")

        for _, r in g.iterrows():
            ts = r["timestamp"]
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            # find last close at or before signal
            before = prices[prices["Datetime"] <= ts]
            if before.empty:
                continue
            p0 = float(before.iloc[-1]["Close"])
            # pick next available close at/after horizon
            target = ts + pd.Timedelta(days=horizon_days)
            after = prices[prices["Datetime"] >= target]
            if after.empty:
                continue
            p1 = float(after.iloc[0]["Close"])
            ret = (p1 - p0) / p0 if p0 else 0.0
            direction = r["direction"].lower()
            label = 1 if (ret > 0 and direction == "long") or (ret < 0 and direction == "short") else 0
            row = r.to_dict()
            row.update({"ret_1d": ret, "label": label})
            labeled.append(row)
    return pd.DataFrame(labeled)


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    base = df.copy()
    base["sentiment"] = base.get("sentiment", "").fillna("").astype(str).str.lower()
    base["topic"] = base.get("topic", "").fillna("").astype(str).str.lower()
    base["has_events"] = base.get("events", "").fillna("").astype(str).str.len().gt(0).astype(int)
    base["confidence"] = pd.to_numeric(base.get("confidence", 0.5), errors="coerce").fillna(0.5)
    base["direction"] = base.get("direction", "").fillna("").astype(str).str.lower()

    feats = base[["confidence", "has_events"]].copy()
    for col in ["sentiment", "topic", "direction", "pair"]:
        feats = feats.join(pd.get_dummies(base[col], prefix=col))

    y = base["label"].astype(int)
    return feats, y


def train(log_path: str, out_path: str) -> None:
    df = pd.read_csv(log_path)
    labeled = _label_returns(df, horizon_days=1)
    if labeled.empty:
        metrics = {
            "samples": 0,
            "train_samples": 0,
            "test_samples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "roc_auc": 0.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "note": "No labeled samples. Check log timestamps and price availability.",
        }
        pd.DataFrame([metrics]).to_csv(out_path, index=False)
        print(f"Wrote metrics to {out_path}")
        return

    # Time-based split
    labeled = labeled.sort_values("timestamp")
    split = int(len(labeled) * 0.8)
    train_df = labeled.iloc[:split]
    test_df = labeled.iloc[split:]

    X_train, y_train = _build_features(train_df)
    X_test, y_test = _build_features(test_df)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros(len(y_test))

    metrics = {
        "samples": len(labeled),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    try:
        import joblib
        model_path = str(Path(out_path).with_suffix(".pkl"))
        joblib.dump(model, model_path)
        # Save feature columns for inference alignment
        cols_path = str(Path(out_path).with_suffix(".json"))
        with open(cols_path, "w", encoding="utf-8") as f:
            json.dump({"columns": list(X_train.columns)}, f)
    except Exception:
        pass
    print(f"Wrote metrics to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FXSense return logistic model (1d horizon)")
    parser.add_argument("--log", default="data/signal_log.csv", help="Signal log CSV path")
    parser.add_argument("--out", default="data/return_logit_metrics.csv", help="Output metrics CSV")
    args = parser.parse_args()
    train(args.log, args.out)


if __name__ == "__main__":
    main()
