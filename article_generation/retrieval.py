from __future__ import annotations

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
                w.text AS reference_text
            FROM articles a
            JOIN article_topic_outputs ato ON ato.article_id = a.article_id
            JOIN wiki_article w ON w.url = a.url
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
        )

    def retrieve_top_k(self, bundle: ArticleBundle, top_k: int) -> list[RetrievedPassage]:
        index_path = self._index_dir / bundle.index_file
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")

        index = faiss.read_index(str(index_path))
        query = self._encode_query(bundle.topic)
        k = min(top_k, index.ntotal)
        scores, row_ids = index.search(query, k)
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

        passages: list[RetrievedPassage] = []
        for rank, row_id in enumerate(faiss_rows, start=1):
            row = by_row_id.get(row_id)
            if row is None:
                continue
            passages.append(
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
        return passages

    def _encode_query(self, text: str) -> np.ndarray:
        vector = self._encoder.encode(
            [text],
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
        )
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        return vector
