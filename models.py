from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class WikiArticle:
    title: str
    url: str
    html: str


@dataclass(slots=True)
class Citation:
    wiki_url: str
    source_url: str
    source_host: str
    anchor_text: str
    ordinal: int


@dataclass(slots=True)
class FetchedDoc:
    url: str
    final_url: str
    status_code: int
    content_type: str
    html: str
    fetch_ms: int
