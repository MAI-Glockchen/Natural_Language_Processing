from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
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
