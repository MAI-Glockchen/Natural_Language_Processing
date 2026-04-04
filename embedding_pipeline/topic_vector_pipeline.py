# -----------------------------
# Topic Inference & Vector Indexing Pipeline
# Processes articles from database, infers topics, builds FAISS indices
# -----------------------------

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

try:
    import faiss  # type: ignore[import-not-found]
except Exception:
    faiss = None

from db.models import Article, CitationPassage, ArticleTopicOutput, FaissPassageMap
from db.session import get_session
from .topic_inference import (
    infer_topic,
    prepare_topic_inference_inputs,
    finalize_topic_inference,
)
from .vector_index import PassageVectorIndex, batch_embed


def _extract_article_runtime_fields(
    article_data: Dict[str, Any],
    min_passages: int,
) -> Optional[Dict[str, Any]]:
    passage_data = article_data.get("passage_data", [])

    if len(passage_data) < min_passages:
        return None

    passages_text = [p["text"] for p in passage_data if p.get("text")]
    if len(passages_text) < min_passages:
        return None

    title_hints = [article_data.get("article_title", "")]
    title_hints.extend(
        p.get("citation_title", "")
        for p in passage_data
        if p.get("citation_title")
    )

    passage_rows = [
        {
            "passage_id": p["passage_id"],
            "text": p["text"],
        }
        for p in passage_data
        if p.get("text")
    ]

    return {
        "passage_data": passage_data,
        "passages_text": passages_text,
        "title_hints": title_hints,
        "passage_rows": passage_rows,
    }


def _prepare_article_payload_worker(
    article_data: Dict[str, Any],
    min_passages: int,
    infer_topic_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    runtime = _extract_article_runtime_fields(article_data, min_passages)
    if runtime is None:
        return None

    infer_kwargs = infer_topic_kwargs or {}
    prepared = prepare_topic_inference_inputs(
        passages=runtime["passages_text"],
        title_hints=runtime["title_hints"],
        **infer_kwargs,
    )

    return {
        "prepared_topic": prepared,
    }


class TopicVectorPipeline:
    """
    Pipeline for inferring topics and building vector indices from database articles.
    """

    def __init__(
        self,
        output_dir: str = "vector_indices",
        batch_size: int = 10,
        min_passages: int = 5,
        use_db: bool = True,
        persist_outputs_to_db: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.min_passages = min_passages
        self.session = get_session() if use_db else None
        self.use_db = use_db
        self.persist_outputs_to_db = persist_outputs_to_db
        self.output_session = (
            self.session
            if self.session is not None
            else (get_session() if persist_outputs_to_db else None)
        )
        self.preprocess_workers = max(1, (os.cpu_count() or 6) - 2)
        self.gpu_article_batch_size = max(
            1, int(os.getenv("TOPIC_PIPELINE_GPU_ARTICLE_BATCH", "10"))
        )

        self.results = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "timestamp": datetime.now().isoformat(),
            "articles": [],
        }

    def _extract_passages_from_article(self, article: Article) -> Dict[str, Any]:
        passage_data = []

        for citation in article.citations:
            passages = citation.passages
            if passages is None:
                passages = (
                    self.session.query(CitationPassage)
                    .filter(CitationPassage.citation_id == citation.citation_id)
                    .all()
                )

            for passage in passages:
                passage_data.append(
                    {
                        "passage_id": f"{citation.citation_id}_{passage.passage_id}",
                        "text": passage.content,
                        "citation_url": citation.url,
                        "citation_title": citation.title,
                    }
                )

        return {
            "article_url": article.url,
            "article_title": article.title,
            "passage_data": passage_data,
        }

    def _get_or_create_article_row(
        self, article_data: Dict[str, Any]
    ) -> Optional[Article]:
        if self.output_session is None:
            return None

        article_url = article_data.get("article_url", "")
        article_title = article_data.get("article_title", "")
        if not article_url:
            return None

        article_row = (
            self.output_session.query(Article)
            .filter(Article.url == article_url)
            .one_or_none()
        )
        if article_row is None:
            article_row = Article(url=article_url, title=article_title)
            self.output_session.add(article_row)
            self.output_session.flush()
        elif article_title and article_row.title != article_title:
            article_row.title = article_title

        return article_row

    def _persist_outputs(
        self,
        article_data: Dict[str, Any],
        metadata: Dict[str, Any],
        passage_data: List[Dict[str, Any]],
    ) -> None:
        if not self.persist_outputs_to_db or self.output_session is None:
            return

        try:
            article_row = self._get_or_create_article_row(article_data)
            if article_row is None:
                return

            topic_row = (
                self.output_session.query(ArticleTopicOutput)
                .filter(ArticleTopicOutput.article_id == article_row.article_id)
                .one_or_none()
            )
            candidates_json = json.dumps(
                [
                    (label, float(score))
                    for label, score in metadata.get("candidates", [])
                ]
            )

            if topic_row is None:
                topic_row = ArticleTopicOutput(
                    article_id=article_row.article_id,
                    topic=metadata.get("topic", "unknown topic"),
                    candidates_json=candidates_json,
                    index_file=metadata.get("index_file", ""),
                    index_backend=metadata.get("index_backend", ""),
                    embedding_dim=metadata.get("embedding_dim", 0),
                )
                self.output_session.add(topic_row)
            else:
                topic_row.topic = metadata.get("topic", "unknown topic")
                topic_row.candidates_json = candidates_json
                topic_row.index_file = metadata.get("index_file", "")
                topic_row.index_backend = metadata.get("index_backend", "")
                topic_row.embedding_dim = metadata.get("embedding_dim", 0)

            (
                self.output_session.query(FaissPassageMap)
                .filter(FaissPassageMap.article_id == article_row.article_id)
                .delete(synchronize_session=False)
            )

            for row_id, p in enumerate(passage_data):
                if not p.get("text"):
                    continue
                self.output_session.add(
                    FaissPassageMap(
                        article_id=article_row.article_id,
                        faiss_row_id=row_id,
                        index_file=metadata.get("index_file", ""),
                        passage_key=str(p.get("passage_id", "")),
                    )
                )

            self.output_session.commit()
        except Exception as e:
            self.output_session.rollback()
            print(f"[WARNING] Failed to persist topic/mapping outputs: {e}")

    def _prepare_article_payload(
        self,
        article_data: Dict[str, Any],
        infer_topic_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        prepared = _prepare_article_payload_worker(
            article_data=article_data,
            min_passages=self.min_passages,
            infer_topic_kwargs=infer_topic_kwargs,
        )
        if prepared is None:
            return None

        runtime = _extract_article_runtime_fields(article_data, self.min_passages)
        if runtime is None:
            return None

        return {
            "article_data": article_data,
            "prepared_topic": prepared["prepared_topic"],
            "runtime": runtime,
        }

    def _finalize_prepared_article(
        self,
        prepared_article: Dict[str, Any],
        candidate_embeddings: np.ndarray | None,
        passage_embeddings: np.ndarray | None,
    ) -> Optional[Dict[str, Any]]:
        try:
            article_data = prepared_article["article_data"]
            runtime = prepared_article["runtime"]

            topic_result = finalize_topic_inference(
                prepared=prepared_article["prepared_topic"],
                passages=runtime["passages_text"],
                title_hints=runtime["title_hints"],
                candidate_embeddings=candidate_embeddings,
                passage_embeddings=passage_embeddings,
                return_passage_embeddings=True,
            )

            index = PassageVectorIndex()
            index.add_many_precomputed(
                passages=runtime["passage_rows"],
                vectors=topic_result["passage_embeddings"],
            )

            index_filename = f"{len(self.results['articles'])}_article.faiss"
            index_path = self.output_dir / index_filename

            if index._faiss_index is not None and faiss is not None:
                faiss.write_index(index._faiss_index, str(index_path))

            embedding_dim = 0
            if len(topic_result["passage_embeddings"]) > 0:
                embedding_dim = int(topic_result["passage_embeddings"].shape[1])

            metadata = {
                "article_url": article_data.get("article_url", "unknown"),
                "article_title": article_data.get("article_title", "unknown"),
                "topic": topic_result["topic"],
                "candidates": topic_result["candidates"],
                "num_passages": len(runtime["passage_rows"]),
                "index_file": index_filename,
                "index_backend": index.get_index_backend(),
                "embedding_dim": embedding_dim,
            }

            self._persist_outputs(
                article_data=article_data,
                metadata=metadata,
                passage_data=runtime["passage_data"],
            )

            self.results["articles"].append(metadata)
            self.results["processed"] += 1
            return metadata
        except Exception as e:
            print(f"[ERROR] Failed finalizing article payload: {e}")
            self.results["failed"] += 1
            return None

    def _process_article_payload(
        self,
        article_data: Dict[str, Any],
        infer_topic_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            prepared_article = self._prepare_article_payload(
                article_data=article_data,
                infer_topic_kwargs=infer_topic_kwargs,
            )
            if prepared_article is None:
                self.results["skipped"] += 1
                return None

            runtime = prepared_article["runtime"]
            topic_result = infer_topic(
                passages=runtime["passages_text"],
                title_hints=runtime["title_hints"],
                return_passage_embeddings=True,
                **(infer_topic_kwargs or {}),
            )

            index = PassageVectorIndex()
            index.add_many_precomputed(
                passages=runtime["passage_rows"],
                vectors=topic_result["passage_embeddings"],
            )

            index_filename = f"{len(self.results['articles'])}_article.faiss"
            index_path = self.output_dir / index_filename

            if index._faiss_index is not None and faiss is not None:
                faiss.write_index(index._faiss_index, str(index_path))

            embedding_dim = 0
            if len(topic_result["passage_embeddings"]) > 0:
                embedding_dim = int(topic_result["passage_embeddings"].shape[1])

            metadata = {
                "article_url": article_data.get("article_url", "unknown"),
                "article_title": article_data.get("article_title", "unknown"),
                "topic": topic_result["topic"],
                "candidates": topic_result["candidates"],
                "num_passages": len(runtime["passage_rows"]),
                "index_file": index_filename,
                "index_backend": index.get_index_backend(),
                "embedding_dim": embedding_dim,
            }

            self._persist_outputs(
                article_data=article_data,
                metadata=metadata,
                passage_data=runtime["passage_data"],
            )

            self.results["articles"].append(metadata)
            self.results["processed"] += 1
            return metadata
        except Exception as e:
            print(f"[ERROR] Failed processing article payload: {e}")
            self.results["failed"] += 1
            return None

    def process_article(
        self,
        article: Article,
        infer_topic_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        article_data = self._extract_passages_from_article(article)
        return self._process_article_payload(
            article_data=article_data,
            infer_topic_kwargs=infer_topic_kwargs,
        )

    def process_batch(
        self,
        articles: List[Article],
        infer_topic_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        batch_results = []

        for i, article in enumerate(articles, 1):
            if verbose:
                print(f"[{i}/{len(articles)}] Processing {article.title[:50]}...")

            metadata = self.process_article(article, infer_topic_kwargs)
            if metadata:
                batch_results.append(metadata)

        return batch_results

    def process_all(
        self,
        limit: Optional[int] = None,
        skip: int = 0,
        infer_topic_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if not self.use_db or self.session is None:
            raise RuntimeError(
                "process_all() requires DB access. "
                "Initialize with use_db=True or use process_mock_articles()."
            )

        query = self.session.query(Article)

        if skip > 0:
            query = query.offset(skip)

        if limit:
            query = query.limit(limit)

        total_articles = query.count()

        if verbose:
            print(
                f"[INFO] Processing {total_articles} articles (batch size: {self.batch_size})"
            )

        for batch_start in range(0, total_articles, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_articles)
            batch_articles = query.offset(batch_start).limit(self.batch_size).all()

            if verbose:
                print(
                    f"\n[BATCH {batch_start // self.batch_size + 1}] "
                    f"Processing articles {batch_start}-{batch_end}/{total_articles}"
                )

            self.process_batch(
                batch_articles, infer_topic_kwargs=infer_topic_kwargs, verbose=verbose
            )

        self._save_summary()
        return self.results

    def process_mock_articles(
        self,
        mock_articles: List[Dict[str, Any]],
        infer_topic_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        total_articles = len(mock_articles)
        if verbose:
            print(
                f"[INFO] Processing {total_articles} mock articles "
                f"(batch size: {self.batch_size}, preprocess workers: {self.preprocess_workers}, "
                f"gpu article batch: {self.gpu_article_batch_size})"
            )

        with ProcessPoolExecutor(max_workers=self.preprocess_workers) as executor:
            for batch_start in range(0, total_articles, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_articles)
                batch_articles = mock_articles[batch_start:batch_end]

                if verbose:
                    print(
                        f"\n[BATCH {batch_start // self.batch_size + 1}] "
                        f"Processing mock articles {batch_start}-{batch_end}/{total_articles}"
                    )

                prepared_batch = list(
                    executor.map(
                        _prepare_article_payload_worker,
                        batch_articles,
                        repeat(self.min_passages),
                        repeat(infer_topic_kwargs),
                    )
                )

                valid_prepared: list[Dict[str, Any]] = []
                skipped_in_batch = 0

                for i, prepared_light in enumerate(prepared_batch, 1):
                    article_data = batch_articles[i - 1]
                    if verbose:
                        title = article_data.get("article_title", "unknown")
                        print(f"[{i}/{len(batch_articles)}] Processing {title[:50]}...")

                    if prepared_light is None:
                        skipped_in_batch += 1
                        continue

                    runtime = _extract_article_runtime_fields(
                        article_data,
                        self.min_passages,
                    )
                    if runtime is None:
                        skipped_in_batch += 1
                        continue

                    valid_prepared.append(
                        {
                            "article_data": article_data,
                            "prepared_topic": prepared_light["prepared_topic"],
                            "runtime": runtime,
                        }
                    )

                self.results["skipped"] += skipped_in_batch

                if not valid_prepared:
                    continue

                for gpu_start in range(0, len(valid_prepared), self.gpu_article_batch_size):
                    gpu_chunk = valid_prepared[
                        gpu_start : gpu_start + self.gpu_article_batch_size
                    ]

                    all_candidate_labels: list[str] = []
                    all_passages_text: list[str] = []
                    candidate_counts: list[int] = []
                    passage_counts: list[int] = []

                    for prepared_article in gpu_chunk:
                        candidate_labels = list(
                            prepared_article["prepared_topic"]["top_candidates"].keys()
                        )
                        passages_text = prepared_article["runtime"]["passages_text"]

                        all_candidate_labels.extend(candidate_labels)
                        all_passages_text.extend(passages_text)
                        candidate_counts.append(len(candidate_labels))
                        passage_counts.append(len(passages_text))

                    all_candidate_embeddings = (
                        batch_embed(all_candidate_labels)
                        if all_candidate_labels
                        else np.empty((0, 0), dtype=np.float32)
                    )
                    all_passage_embeddings = (
                        batch_embed(all_passages_text)
                        if all_passages_text
                        else np.empty((0, 0), dtype=np.float32)
                    )

                    candidate_offset = 0
                    passage_offset = 0

                    for prepared_article, candidate_count, passage_count in zip(
                        gpu_chunk, candidate_counts, passage_counts
                    ):
                        candidate_embeddings = None
                        if candidate_count > 0:
                            candidate_embeddings = all_candidate_embeddings[
                                candidate_offset : candidate_offset + candidate_count
                            ]
                        passage_embeddings = None
                        if passage_count > 0:
                            passage_embeddings = all_passage_embeddings[
                                passage_offset : passage_offset + passage_count
                            ]

                        self._finalize_prepared_article(
                            prepared_article=prepared_article,
                            candidate_embeddings=candidate_embeddings,
                            passage_embeddings=passage_embeddings,
                        )

                        candidate_offset += candidate_count
                        passage_offset += passage_count

        self._save_summary()
        return self.results

    def _save_summary(self) -> None:
        summary_path = self.output_dir / "summary.json"

        results_copy = self.results.copy()
        results_copy["articles"] = [
            {
                **article,
                "candidates": [
                    (label, float(score)) for label, score in article["candidates"]
                ],
            }
            for article in self.results["articles"]
        ]

        with open(summary_path, "w") as f:
            json.dump(results_copy, f, indent=2)

        if len(self.results["articles"]) > 0:
            print(f"\n[INFO] Summary saved to {summary_path}")

    def load_index(self, article_idx: int) -> Optional[PassageVectorIndex]:
        if article_idx >= len(self.results["articles"]):
            return None

        metadata = self.results["articles"][article_idx]
        index_path = self.output_dir / metadata["index_file"]

        if not index_path.exists():
            return None

        try:
            if faiss is None:
                print("[ERROR] FAISS is not available in the current environment")
                return None
            index = PassageVectorIndex()
            faiss_index = faiss.read_index(str(index_path))
            index._faiss_index = faiss_index
            return index
        except Exception as e:
            print(f"[ERROR] Failed to load index: {e}")
            return None

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_processed": self.results["processed"],
            "total_skipped": self.results["skipped"],
            "total_failed": self.results["failed"],
            "total_articles": len(self.results["articles"]),
            "output_dir": str(self.output_dir),
            "timestamp": self.results["timestamp"],
        }