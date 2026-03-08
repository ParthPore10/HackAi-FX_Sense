"""
FXSense scraping layer (MVP)
Lightweight scraping from public RSS feeds and a few HTML pages.

Usage:
  python fxsense/scrapers/scrape_sources.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "FXSenseMVP/0.1 (+https://example.com; contact: hackathon@local)"
)
REQUEST_TIMEOUT = 10


@dataclass
class Source:
    name: str
    kind: str  # "rss" or "html"
    url: str
    # Optional CSS selector for HTML pages
    selector: Optional[str] = None


SOURCES: List[Source] = [
    # RSS feeds
    Source(
        name="Federal Reserve (Press)",
        kind="rss",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
    ),
    Source(
        name="Federal Reserve (Monetary Policy)",
        kind="rss",
        url="https://www.federalreserve.gov/feeds/press_monetary.xml",
    ),
    Source(
        name="Federal Reserve (Speeches)",
        kind="rss",
        url="https://www.federalreserve.gov/feeds/speeches.xml",
    ),
    Source(
        name="Federal Reserve (Speeches & Testimony)",
        kind="rss",
        url="https://www.federalreserve.gov/feeds/speeches_and_testimony.xml",
    ),
    Source(
        name="ECB (Press & Speeches)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/press.html",
    ),
    Source(
        name="ECB (Blog)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/blog.html",
    ),
    Source(
        name="ECB (Statistical Press Releases)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/statpress.html",
    ),
    Source(
        name="ECB (Publications)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/pub.html",
    ),
    Source(
        name="ECB (Working Papers)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/wppub.html",
    ),
    Source(
        name="ECB (Market Operations)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/operations.html",
    ),
    Source(
        name="ECB (Research Bulletin)",
        kind="rss",
        url="https://www.ecb.europa.eu/rss/rbu.html",
    ),
    Source(
        name="BoE (News)",
        kind="rss",
        url="https://www.bankofengland.co.uk/rss/news",
    ),
    Source(
        name="BoJ (Press)",
        kind="rss",
        url="https://www.boj.or.jp/en/rss/whatsnew.rdf",
    ),
    Source(
        name="BIS (All Categories)",
        kind="rss",
        url="https://www.bis.org/doclist/rss_all_categories.rss",
    ),
    Source(
        name="BIS (Press Releases)",
        kind="rss",
        url="https://www.bis.org/doclist/all_pressrels.rss",
    ),
    Source(
        name="BIS (Central Bank Speeches)",
        kind="rss",
        url="https://www.bis.org/doclist/cbspeeches.rss",
    ),
    Source(
        name="BIS (Statistics)",
        kind="rss",
        url="https://www.bis.org/doclist/all_statistics.rss",
    ),
    Source(
        name="BIS (Research Hub)",
        kind="rss",
        url="https://www.bis.org/doclist/reshub_papers.rss",
    ),
    Source(
        name="BIS Data Portal (Release Calendar)",
        kind="rss",
        url="https://data.bis.org/feed.xml",
    ),
    Source(
        name="RBA (Media Releases)",
        kind="rss",
        url="https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
    ),
    Source(
        name="RBA (Speeches)",
        kind="rss",
        url="https://www.rba.gov.au/rss/rss-cb-speeches.xml",
    ),
    Source(
        name="RBA (Statement on Monetary Policy)",
        kind="rss",
        url="https://www.rba.gov.au/rss/rss-cb-smp.xml",
    ),
    Source(
        name="RBA (Financial Stability Review)",
        kind="rss",
        url="https://www.rba.gov.au/rss/rss-cb-fsr.xml",
    ),
    Source(
        name="RBA (Bulletin)",
        kind="rss",
        url="https://www.rba.gov.au/rss/rss-cb-bulletin.xml",
    ),
    Source(
        name="CNBC (Breaking News)",
        kind="rss",
        url="https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    ),
    # A couple of HTML pages as backup (best-effort)
    Source(
        name="IMF (News)",
        kind="html",
        url="https://www.imf.org/en/News",
        selector="h3 a",
    ),
    Source(
        name="BIS (Press)",
        kind="html",
        url="https://www.bis.org/press/",
        selector="h2 a",
    ),
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _parse_date(entry) -> Optional[str]:
    # feedparser may provide published_parsed or updated_parsed
    parsed = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if parsed:
        return dt.datetime(*parsed[:6]).isoformat()
    return None


def _parse_iso(dt_str: Optional[str]) -> Optional[dt.datetime]:
    if not dt_str:
        return None
    try:
       time_now = dt.datetime.fromisoformat(dt_str)
       return time_now if time_now <= dt.datetime.now() else None
    except Exception:
        return None


def scrape_rss(source: Source, limit: int = 10) -> List[dict]:
    feed = feedparser.parse(source.url)
    items = []
    for entry in feed.entries[:limit]:
        items.append(
            {
                "source": source.name,
                "headline": _clean_text(entry.get("title", "")),
                "summary": _clean_text(entry.get("summary", "")),
                "url": entry.get("link", ""),
                "published": _parse_date(entry),
            }
        )
    return items


def scrape_html(source: Source, limit: int = 10) -> List[dict]:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(source.url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    selector = source.selector or "a"
    links = soup.select(selector)[:limit]
    items = []
    for link in links:
        headline = _clean_text(link.get_text())
        url = link.get("href", "")
        if url and url.startswith("/"):
            # naive absolute URL join
            base = re.match(r"^https?://[^/]+", source.url)
            if base:
                url = base.group(0) + url
        if not headline:
            continue
        items.append(
            {
                "source": source.name,
                "headline": headline,
                "summary": "",
                "url": url,
                "published": None,
            }
        )
    return items


def _is_recent(published: Optional[str], days: int) -> bool:
    if not published:
        return False
    parsed = _parse_iso(published)
    if not parsed:
        return False
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    return parsed >= cutoff


def filter_recent(
    records: List[dict], days: int, include_undated: bool = True
) -> List[dict]:
    if days <= 0:
        return records
    filtered = []
    for r in records:
        pub = r.get("published")
        if _is_recent(pub, days):
            filtered.append(r)
        elif include_undated and not pub:
            filtered.append(r)
    return filtered


def dedupe_records(records: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for r in records:
        key = (
            _clean_text(r.get("headline", "")).lower(),
            _clean_text(r.get("url", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def sample_fallback_records() -> List[dict]:
    return [
        {
            "source": "Sample",
            "headline": "Federal Reserve officials warn inflation remains persistent",
            "summary": "Policymakers emphasize vigilance as price pressures ease slowly.",
            "url": "https://example.com/fed-inflation",
            "published": dt.datetime.isoformat(),
        },
        {
            "source": "Sample",
            "headline": "ECB highlights weaker growth outlook for euro area",
            "summary": "Officials note downside risks amid soft demand.",
            "url": "https://example.com/ecb-growth",
            "published": dt.datetime.isoformat(),
        },
    ]


def scrape_xdk_posts(query: str, limit: int = 20) -> List[dict]:
    """
    Fetch recent posts from X using the official XDK.
    Requires X_BEARER_TOKEN in environment. Returns [] if missing/unavailable.
    """
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        return []

    try:
        from xdk import Client  # type: ignore
    except Exception:
        return []

    client = Client(bearer_token=token)
    items: List[dict] = []

    # Prefer official method name per docs; fallback to alt names
    try:
        iterator = client.posts.search_recent(query=query, max_results=min(100, limit))
    except Exception:
        try:
            iterator = client.posts.recent_search(
                query=query, max_results=min(100, limit)
            )
        except Exception:
            return []

    for page in iterator:
        data = getattr(page, "data", None)
        if not data:
            continue
        for post in data:
            text = getattr(post, "text", None) or post.get("text", "")
            created_at = getattr(post, "created_at", None) or post.get(
                "created_at", None
            )
            items.append(
                {
                    "source": f"X ({query})",
                    "headline": _clean_text(text)[:140],
                    "summary": _clean_text(text),
                    "url": "",
                    "published": (
                        created_at.isoformat()
                        if hasattr(created_at, "isoformat")
                        else created_at
                    ),
                }
            )
            if len(items) >= limit:
                return items
    return items


def scrape_all(limit_per_source: int = 10) -> pd.DataFrame:
    records: List[dict] = []
    for source in SOURCES:
        try:
            if source.kind == "rss":
                records.extend(scrape_rss(source, limit=limit_per_source))
            elif source.kind == "html":
                records.extend(scrape_html(source, limit=limit_per_source))
        except Exception:
            # Fail gracefully per source
            continue

    # Optional XDK integration (requires env var + API access)
    x_queries = [
        "Federal Reserve policy",
        "ECB rates",
        "Bank of England rates",
        "Bank of Japan policy",
        "FX market risk-off",
    ]
    for q in x_queries:
        try:
            records.extend(scrape_xdk_posts(q, limit=5))
        except Exception:
            continue

    records = dedupe_records(records)
    if not records:
        records = sample_fallback_records()

    df = pd.DataFrame(records)
    # Normalize columns
    for col in ["source", "headline", "summary", "url", "published"]:
        if col not in df.columns:
            df[col] = None
    return df


def _sort_records(records: List[dict]) -> List[dict]:
    def key_fn(r: dict) -> Tuple[int, dt.datetime]:
        parsed = _parse_iso(r.get("published")) or dt.datetime.min
        # Put dated items first
        return (0 if r.get("published") else 1, parsed)

    return sorted(records, key=key_fn, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FXSense scraper (MVP)")
    parser.add_argument("--limit-per-source", type=int, default=50)
    parser.add_argument("--total-limit", type=int, default=1000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--include-undated",
        action="store_true",
        help="Include items without timestamps when filtering by days",
    )
    parser.add_argument(
        "--watch-minutes",
        type=int,
        default=0,
        help="If set > 0, keep polling sources every N minutes.",
    )
    args = parser.parse_args()

    # Save relative to project root (one level up from this file)
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "scraped_latest.csv"

    def run_once() -> None:
        df = scrape_all(limit_per_source=args.limit_per_source)
        records = df.to_dict(orient="records")
        records = filter_recent(
            records, days=args.days, include_undated=args.include_undated
        )
        records = dedupe_records(records)
        records = _sort_records(records)
        if args.total_limit > 0:
            records = records[: args.total_limit]
        df_out = pd.DataFrame(records)
        df_out.to_csv(out_path, index=False)
        print(f"Saved {len(df_out)} records to {out_path}")

    if args.watch_minutes and args.watch_minutes > 0:
        interval = args.watch_minutes * 60
        print(f"Polling every {args.watch_minutes} minutes...")
        while True:
            run_once()
            time.sleep(interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
