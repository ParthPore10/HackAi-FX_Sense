"""
Minimal Gemini client (REST) for hackathon use.
Requires GEMINI_API_KEY in environment.
"""

from __future__ import annotations

import os
import requests


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def _call_gemini(prompt: str, temperature: float = 0.2) -> str | None:
    if not GEMINI_API_KEY:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def summarize(text: str) -> str | None:
    prompt = (
        "Summarize the following in 1 sentence, focusing on FX-relevant macro signal:\n"
        f"{text}"
    )
    return _call_gemini(prompt, temperature=0.2)


def refine_signal(text: str) -> str | None:
    prompt = (
        "Given the headline/summary below, provide a 1-sentence FX trade rationale "
        "and suggested bias (bullish/bearish) if applicable. Keep it concise.\n"
        f"{text}"
    )
    return _call_gemini(prompt, temperature=0.3)
