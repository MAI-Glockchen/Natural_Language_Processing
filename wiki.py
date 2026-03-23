from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import (
    MAX_ARTICLES,
    MAX_CITATIONS_PER_ARTICLE,
    MIN_CITATIONS,
    POPULAR_ARTICLES_FILE,
    USER_AGENT,
)


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


def _is_useful_title(title: str) -> bool:
    if not title:
        return False

    bad_prefixes = ("Special:", "Wikipedia:", "Template:", "Help:", "Category:", "Portal:", "File:")
    if title.startswith(bad_prefixes):
        return False
    if title == "Main Page":
        return False
    return True


def _title_to_url(title: str) -> str:
    normalized = _normalize_title(title)
    return f"https://en.wikipedia.org/wiki/{quote(normalized.replace(' ', '_'))}"


def _load_titles_from_file(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Popular article file not found: {path}. "
            f"Create it with one article title per line."
        )

    out: list[str] = []
    seen: set[str] = set()

    for line in p.read_text(encoding="utf-8").splitlines():
        title = _normalize_title(line)
        if not _is_useful_title(title):
            continue
        if title in seen:
            continue
        seen.add(title)
        out.append(title)

    return out


def get_popular_articles(limit: int = MAX_ARTICLES) -> List[Dict[str, str]]:
    titles = _load_titles_from_file(POPULAR_ARTICLES_FILE)

    result: list[dict[str, str]] = []
    for title in titles[:limit]:
        result.append(
            {
                "title": title,
                "url": _title_to_url(title),
            }
        )
    return result


def extract_citations(article_url: str, max_citations: int = MAX_CITATIONS_PER_ARTICLE) -> List[Dict[str, str]]:
    with requests.Session() as s:
        s.headers.update({"User-Agent": USER_AGENT})
        html = _get_text(s, article_url)

    soup = BeautifulSoup(html, "lxml")
    refs = soup.select("ol.references li, cite a[href], span.reference a[href]")

    seen: set[str] = set()
    citations: List[Dict[str, str]] = []

    for ref in refs:
        links = ref.select("a[href]") if hasattr(ref, "select") else []
        for a in links:
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

    if len(citations) < MIN_CITATIONS:
        return []

    return citations