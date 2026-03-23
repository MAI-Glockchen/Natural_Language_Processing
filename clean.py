from __future__ import annotations
import re
from bs4 import BeautifulSoup
from readability import Document
from config import CHUNK_WORDS, CHUNK_OVERLAP, MAX_DOC_CHARS
from util import squash

_rx_noise = re.compile(r'\b(cookie|privacy|subscribe|advertis|newsletter|sign up|all rights reserved)\b', re.I)


def clean_html(html: str) -> str:
    try:
        main = Document(html).summary(html_partial=True)
    except Exception:
        main = html
    soup = BeautifulSoup(main[:MAX_DOC_CHARS], 'lxml')
    for t in soup(['script', 'style', 'noscript', 'svg', 'img', 'form', 'nav', 'footer', 'header', 'aside']):
        t.decompose()
    parts = []
    for node in soup.select('h1,h2,h3,h4,p,li,blockquote'):
        txt = squash(node.get_text(' ', strip=True))
        if len(txt) > 40 and not _rx_noise.search(txt):
            parts.append(txt)
    return '\n'.join(parts)


def make_passages(text: str) -> list[tuple[int, str, int]]:
    words = text.split()
    if not words:
        return []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    out = []
    for i, start in enumerate(range(0, len(words), step)):
        chunk = words[start:start + CHUNK_WORDS]
        if len(chunk) < 40:
            break
        out.append((i, ' '.join(chunk), len(chunk)))
    return out
