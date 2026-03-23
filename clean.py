from __future__ import annotations

import re

from bs4 import BeautifulSoup
from readability import Document


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\S+")


def _strip_nul(text: str) -> str:
    return text.replace("\x00", "")


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_html(html: str) -> str:
    html = _strip_nul(html or "")
    if not html.strip():
        return ""

    extracted_html = ""
    try:
        extracted_html = Document(html).summary()
    except Exception:
        extracted_html = ""

    candidate = extracted_html if extracted_html.strip() else html

    try:
        soup = BeautifulSoup(candidate, "lxml")
    except Exception:
        return ""

    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe", "form"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = _normalize_ws(_strip_nul(text))
    return text


def make_passages(text: str, target_words: int = 180, overlap_words: int = 30) -> list[tuple[int, str, int]]:
    words = text.split()
    if not words:
        return []

    step = max(1, target_words - overlap_words)
    out: list[tuple[int, str, int]] = []

    idx = 0
    start = 0
    n = len(words)

    while start < n:
        chunk = words[start:start + target_words]
        if not chunk:
            break
        passage = " ".join(chunk).strip()
        wc = len(chunk)
        if wc >= 40:
            out.append((idx, passage, wc))
            idx += 1
        start += step

    return out