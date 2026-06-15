from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from bert_score import score as bert_score  # type: ignore[import-not-found]
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class EvaluationResult:
    rouge1_f1: float
    rouge2_f1: float
    rougel_f1: float
    bertscore_f1: float
    title_similarity: float
    section_count_generated: int
    section_count_reference: int
    section_count_abs_diff: int
    article_length_ratio: float


class EvaluationService:
    def __init__(self, title_embedding_model_name: str, embed_device: str) -> None:
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self._title_encoder = SentenceTransformer(title_embedding_model_name, device=embed_device)

    def evaluate(
        self,
        generated_title: str,
        generated_text: str,
        reference_title: str,
        reference_text: str,
    ) -> EvaluationResult:
        rouge = self._rouge.score(reference_text, generated_text)
        bertscore_f1 = self._bertscore_f1(generated_text, reference_text)
        title_similarity = self._title_similarity(generated_title, reference_title)

        generated_sections = _estimate_section_count(generated_text)
        reference_sections = _estimate_section_count(reference_text)
        generated_words = max(_word_count(generated_text), 1)
        reference_words = max(_word_count(reference_text), 1)

        return EvaluationResult(
            rouge1_f1=float(rouge["rouge1"].fmeasure),
            rouge2_f1=float(rouge["rouge2"].fmeasure),
            rougel_f1=float(rouge["rougeL"].fmeasure),
            bertscore_f1=bertscore_f1,
            title_similarity=title_similarity,
            section_count_generated=generated_sections,
            section_count_reference=reference_sections,
            section_count_abs_diff=abs(generated_sections - reference_sections),
            article_length_ratio=float(generated_words / reference_words),
        )

    def _title_similarity(self, generated_title: str, reference_title: str) -> float:
        vectors = self._title_encoder.encode([generated_title, reference_title], convert_to_numpy=True)
        return float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])

    @staticmethod
    @lru_cache(maxsize=256)
    def _bertscore_f1(candidate: str, reference: str) -> float:
        _, _, f1 = bert_score([candidate], [reference], lang="en", verbose=False)
        return float(f1.mean().item())


def _estimate_section_count(text: str) -> int:
    heading_matches = re.findall(r"(?m)^==+\s*[^=].*?\s*==+\s*$", text)
    if heading_matches:
        return len(heading_matches)

    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if _word_count(part) >= 40]
    return max(len(parts), 1)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
