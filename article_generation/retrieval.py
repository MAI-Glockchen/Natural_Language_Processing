from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-not-found]
import numpy as np
from sentence_transformers import SentenceTransformer

from .db import Database


@dataclass(slots=True)
class ArticleBundle:
    article_id: int
    article_url: str
    article_title: str
    topic: str
    candidates_json: str | None
    index_file: str
    index_backend: str | None
    embedding_dim: int | None
    wiki_article_id: int
    reference_title: str
    reference_text: str
    available_passage_count: int


@dataclass(slots=True)
class RetrievedPassage:
    rank: int
    faiss_row_id: int
    score: float
    source_document_id: int
    idx: int
    text: str
    word_count: int
    passage_key: str


class RetrievalService:
    def __init__(
        self,
        database: Database,
        index_dir: Path,
        embedding_model_name: str,
        embed_device: str,
        normalize_embeddings: bool = True,
    ) -> None:
        self._db = database
        self._index_dir = index_dir
        self._normalize_embeddings = normalize_embeddings
        self._encoder = SentenceTransformer(embedding_model_name, device=embed_device)

    def fetch_article_bundle(self, article_id: int) -> ArticleBundle:
        row = self._db.fetch_one(
            """
            SELECT
                a.article_id,
                a.url AS article_url,
                a.title AS article_title,
                ato.topic,
                ato.candidates_json,
                ato.index_file,
                ato.index_backend,
                ato.embedding_dim,
                w.id AS wiki_article_id,
                w.title AS reference_title,
                w.text AS reference_text,
                COALESCE(fpm_counts.available_passage_count, 0) AS available_passage_count
            FROM articles a
            JOIN article_topic_outputs ato ON ato.article_id = a.article_id
            JOIN wiki_article w ON w.url = a.url
            LEFT JOIN (
                SELECT
                    article_id,
                    COUNT(*)::int AS available_passage_count
                FROM faiss_passage_map
                GROUP BY article_id
            ) AS fpm_counts ON fpm_counts.article_id = a.article_id
            WHERE a.article_id = :article_id
            """,
            {"article_id": article_id},
        )
        if row is None:
            raise ValueError(f"Article {article_id} not found")
        return ArticleBundle(
            article_id=int(row["article_id"]),
            article_url=str(row["article_url"]),
            article_title=str(row["article_title"]),
            topic=str(row["topic"]),
            candidates_json=row["candidates_json"],
            index_file=str(row["index_file"]),
            index_backend=row["index_backend"],
            embedding_dim=row["embedding_dim"],
            wiki_article_id=int(row["wiki_article_id"]),
            reference_title=str(row["reference_title"]),
            reference_text=str(row["reference_text"]),
            available_passage_count=int(row["available_passage_count"]),
        )

    def retrieve_top_k(self, bundle: ArticleBundle, top_k: int) -> list[RetrievedPassage]:
        index_path = self._index_dir / bundle.index_file
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")

        index = faiss.read_index(str(index_path))
        query = self._encode_query(self._build_query_text(bundle))

        search_k = min(max(top_k * 3, top_k), index.ntotal)
        scores, row_ids = index.search(query, search_k)
        faiss_rows = [int(row_id) for row_id in row_ids[0] if int(row_id) >= 0]
        if not faiss_rows:
            return []

        rows = self._db.fetch_all(
            """
            SELECT
                fpm.faiss_row_id,
                fpm.passage_key,
                p.source_document_id,
                p.idx,
                p.text,
                p.word_count
            FROM faiss_passage_map fpm
            JOIN passage p
              ON p.source_document_id = split_part(fpm.passage_key, '_', 1)::bigint
             AND p.idx = split_part(fpm.passage_key, '_', 2)::int
            WHERE fpm.article_id = :article_id
              AND fpm.faiss_row_id = ANY(:faiss_rows)
            """,
            {"article_id": bundle.article_id, "faiss_rows": faiss_rows},
        )
        by_row_id: dict[int, dict[str, Any]] = {int(row["faiss_row_id"]): dict(row) for row in rows}

        ranked: list[RetrievedPassage] = []
        for rank, row_id in enumerate(faiss_rows, start=1):
            row = by_row_id.get(row_id)
            if row is None:
                continue
            ranked.append(
                RetrievedPassage(
                    rank=rank,
                    faiss_row_id=row_id,
                    score=float(scores[0][rank - 1]),
                    source_document_id=int(row["source_document_id"]),
                    idx=int(row["idx"]),
                    text=str(row["text"]),
                    word_count=int(row["word_count"]),
                    passage_key=str(row["passage_key"]),
                )
            )

        return self._select_diverse_passages(ranked, top_k)

    def _select_diverse_passages(
        self,
        passages: list[RetrievedPassage],
        top_k: int,
    ) -> list[RetrievedPassage]:
        selected: list[RetrievedPassage] = []
        per_source_counts: dict[int, int] = {}

        for passage in passages:
            if len(selected) >= top_k:
                break

            source_count = per_source_counts.get(passage.source_document_id, 0)
            if source_count >= 3:
                continue

            norm = _normalize_text(passage.text)
            if any(_too_similar(norm, _normalize_text(existing.text)) for existing in selected):
                continue

            per_source_counts[passage.source_document_id] = source_count + 1
            selected.append(
                RetrievedPassage(
                    rank=len(selected) + 1,
                    faiss_row_id=passage.faiss_row_id,
                    score=passage.score,
                    source_document_id=passage.source_document_id,
                    idx=passage.idx,
                    text=passage.text,
                    word_count=passage.word_count,
                    passage_key=passage.passage_key,
                )
            )

        if len(selected) < min(top_k, len(passages)):
            used = {p.faiss_row_id for p in selected}
            for passage in passages:
                if len(selected) >= top_k:
                    break
                if passage.faiss_row_id in used:
                    continue
                selected.append(
                    RetrievedPassage(
                        rank=len(selected) + 1,
                        faiss_row_id=passage.faiss_row_id,
                        score=passage.score,
                        source_document_id=passage.source_document_id,
                        idx=passage.idx,
                        text=passage.text,
                        word_count=passage.word_count,
                        passage_key=passage.passage_key,
                    )
                )

        return selected

    def _build_query_text(self, bundle: ArticleBundle) -> str:
        parts = [bundle.article_title.strip(), bundle.topic.strip()]
        parts.extend(self._top_candidate_terms(bundle.candidates_json, limit=3))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _top_candidate_terms(candidates_json: str | None, limit: int) -> list[str]:
        if not candidates_json:
            return []

        try:
            raw = json.loads(candidates_json)
        except json.JSONDecodeError:
            return []

        candidates: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, (list, tuple)) or not item:
                continue
            term = str(item[0]).strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(term)
            if len(candidates) >= limit:
                break
        return candidates

    def _encode_query(self, text: str) -> np.ndarray:
        vector = self._encoder.encode(
            [text],
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
        )
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        return vector


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text.strip()


def _too_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))
    return overlap >= 0.8