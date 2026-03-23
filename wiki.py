from __future__ import annotations

from typing import Dict, List
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def _get_json(session: requests.Session, url: str, *, params: dict | None = None) -> dict:
    r = session.get(url, params=params, timeout=(5, 20))
    r.raise_for_status()
    return r.json()


def _get_text(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=(5, 20))
    r.raise_for_status()
    return r.text


def _normalize_title(title: str) -> str:
    return title.replace("_", " ").strip()


def _is_valid_citation_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False

    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False

    host = p.netloc.lower()
    if "wikipedia.org" in host or "wikimedia.org" in host or "wikidata.org" in host:
        return False

    return True


def get_popular_articles(limit: int = 25) -> List[Dict[str, str]]:
    limit = max(1, min(limit, 50))

    with requests.Session() as s:
        s.headers.update(
            {
                "User-Agent": "citation-pipeline/0.1 (research project)",
                "Accept": "application/json",
            }
        )

        data = _get_json(
            s,
            WIKIPEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "mostviewed",
                "pvimlimit": limit,
            },
        )

    items = data.get("query", {}).get("mostviewed", [])
    result: List[Dict[str, str]] = []

    for item in items:
        title = item.get("title")
        if not title:
            continue
        if title.startswith(("Special:", "Wikipedia:")) or title == "Main Page":
            continue

        normalized = _normalize_title(title)
        result.append(
            {
                "title": normalized,
                "url": f"https://en.wikipedia.org/wiki/{quote(normalized.replace(' ', '_'))}",
            }
        )

    return result


def extract_citations(article_url: str, max_citations: int = 180) -> List[Dict[str, str]]:
    with requests.Session() as s:
        s.headers.update({"User-Agent": "citation-pipeline/0.1 (research project)"})
        html = _get_text(s, article_url)

    soup = BeautifulSoup(html, "lxml")
    refs = soup.select("ol.references li, cite a[href], span.reference a[href]")

    seen: set[str] = set()
    citations: List[Dict[str, str]] = []

    for ref in refs:
        for a in ref.select("a[href]") if hasattr(ref, "select") else []:
            href = a.get("href")
            if not href:
                continue

            url = urljoin(article_url, href)
            if not _is_valid_citation_url(url):
                continue
            if url in seen:
                continue

            seen.add(url)
            citations.append(
                {
                    "url": url,
                    "anchor_text": a.get_text(" ", strip=True)[:500],
                }
            )

            if len(citations) >= max_citations:
                return citations

    return citations