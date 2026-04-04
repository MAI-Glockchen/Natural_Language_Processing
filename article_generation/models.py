from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ArticleTopicRecord:
    article_id: int
    url: str
    title: str
    topic: str
    index_file: str


@dataclass(frozen=True, slots=True)
class PassageRecord:
    faiss_row_id: int
    passage_key: str
    source_document_id: int
    passage_idx: int
    text: str


@dataclass(frozen=True, slots=True)
class ReferenceArticle:
    article_id: int
    wiki_article_id: int
    title: str
    url: str
    reference_text: str


@dataclass(frozen=True, slots=True)
class GeneratedArticleRecord:
    article_id: int
    split: str
    method: str
    prompt_version: str
    model_name: str
    top_k: int
    topic: str
    index_file: str
    generated_title: str
    generated_text: str
    reference_title: str
    reference_text: str
    rouge1_f1: float | None = None
    rouge2_f1: float | None = None
    rougel_f1: float | None = None
    bertscore_f1: float | None = None
    title_similarity: float | None = None
    section_count_generated: int | None = None
    section_count_reference: int | None = None
    section_count_abs_diff: int | None = None
    article_length_ratio: float | None = None
    created_at: datetime | None = None
