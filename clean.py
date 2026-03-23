from __future__ import annotations

import re

from bs4 import BeautifulSoup
from readability import Document

from config import PASSAGE_MAX_WORDS, PASSAGE_MIN_WORDS, PASSAGE_TARGET_WORDS


_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\S+")


def _strip_nul(text: str) -> str:
    return text.replace("\x00", "")


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


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


def _split_sentences(text: str) -> list[str]:
    text = _normalize_ws(text)
    if not text:
        return []

    parts = _SENTENCE_RE.split(text)
    out = []
    for part in parts:
        s = _normalize_ws(part)
        if s:
            out.append(s)
    return out


def make_passages(
    text: str,
    target_words: int = PASSAGE_TARGET_WORDS,
    max_words: int = PASSAGE_MAX_WORDS,
    min_words: int = PASSAGE_MIN_WORDS,
) -> list[tuple[int, str, int]]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    passages: list[tuple[int, str, int]] = []
    current: list[str] = []
    current_wc = 0
    idx = 0

    for sentence in sentences:
        sent_wc = _word_count(sentence)
        if sent_wc == 0:
            continue

        if current and current_wc >= target_words:
            passage = " ".join(current).strip()
            if current_wc >= min_words:
                passages.append((idx, passage, current_wc))
                idx += 1
            current = [sentence]
            current_wc = sent_wc
            continue

        if current and current_wc + sent_wc > max_words:
            passage = " ".join(current).strip()
            if current_wc >= min_words:
                passages.append((idx, passage, current_wc))
                idx += 1
            current = [sentence]
            current_wc = sent_wc
            continue

        current.append(sentence)
        current_wc += sent_wc

    if current and current_wc >= min_words:
        passage = " ".join(current).strip()
        passages.append((idx, passage, current_wc))

    return passages