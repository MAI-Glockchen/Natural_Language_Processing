# -----------------------------
# DB batch runner for topic inference and vector indexing
# Run with: python -m embedding_pipeline.topic_vector_db_runner
# -----------------------------

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, inspect, text
from sqlalchemy.orm import selectinload

from db.models import Article, ArticleCitation, Citation, CitationPassage
from db.session import get_session
from embedding_pipeline.topic_vector_pipeline import TopicVectorPipeline


def _select_processable_article_ids(limit: int, min_passages: int) -> list[int]:
    """Select up to `limit` real Article IDs with enough non-empty passages."""
    session = get_session()
    try:
        rows = (
            session.query(Article.article_id)
            .join(ArticleCitation, ArticleCitation.article_id == Article.article_id)
            .join(CitationPassage, CitationPassage.citation_id == ArticleCitation.citation_id)
            .filter(CitationPassage.content.is_not(None))
            .filter(func.length(func.trim(CitationPassage.content)) > 0)
            .group_by(Article.article_id)
            .having(func.count(CitationPassage.passage_id) >= min_passages)
            .order_by(Article.article_id.asc())
            .limit(limit)
            .all()
        )
        return [int(row[0]) for row in rows]
    finally:
        session.close()


def _chunk_ids(ids: list[int], num_chunks: int) -> list[list[int]]:
    if not ids:
        return []
    chunk_size = max(1, math.ceil(len(ids) / max(1, num_chunks)))
    return [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]


def _process_article_chunk(
    article_ids: list[int],
    worker_idx: int,
    output_root: str,
    min_passages: int,
    batch_size: int,
    persist_outputs_to_db: bool,
    infer_topic_kwargs: dict,
) -> dict:
    worker_output_dir = f"{output_root}/worker_{worker_idx}"
    pipeline = TopicVectorPipeline(
        output_dir=worker_output_dir,
        batch_size=max(1, min(batch_size, len(article_ids))),
        min_passages=min_passages,
        use_db=True,
        persist_outputs_to_db=persist_outputs_to_db,
    )

    articles = (
        pipeline.session.query(Article)
        .filter(Article.article_id.in_(article_ids))
        .options(selectinload(Article.citations).selectinload(Citation.passages))
        .all()
        if article_ids
        else []
    )

    results = pipeline.process_batch(
        articles=articles,
        infer_topic_kwargs=infer_topic_kwargs,
        verbose=False,
    )
    summary = pipeline.get_summary()

    return {
        "processed": summary["total_processed"],
        "skipped": summary["total_skipped"],
        "failed": summary["total_failed"],
        "articles": results,
    }


def _write_combined_summary(output_dir: str, summary: dict, results: list[dict]) -> None:
    """Write a root-level merged summary.json for both single and parallel runs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"

    combined = {
        "processed": int(summary.get("total_processed", 0)),
        "skipped": int(summary.get("total_skipped", 0)),
        "failed": int(summary.get("total_failed", 0)),
        "timestamp": datetime.now().isoformat(),
        "articles": [
            {
                **article,
                "candidates": [
                    (label, float(score)) for label, score in article.get("candidates", [])
                ],
            }
            for article in results
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"[INFO] Combined summary saved to {summary_path}")


def main() -> None:
    limit = int(os.getenv("TOPIC_PIPELINE_LIMIT", "10"))
    min_passages = int(os.getenv("TOPIC_PIPELINE_MIN_PASSAGES", "5"))
    output_dir = os.getenv("TOPIC_PIPELINE_OUTPUT_DIR", "vector_indices")
    workers = max(2, int(os.getenv("TOPIC_PIPELINE_WORKERS", "1")))
    batch_size = max(1, int(os.getenv("TOPIC_PIPELINE_BATCH_SIZE", "100")))

    persist_outputs_to_db = os.getenv("TOPIC_PIPELINE_PERSIST_DB", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    infer_topic_kwargs = {
        "top_k": 5,
        "mmr_diversity": 0.4,
    }

    selected_article_ids = _select_processable_article_ids(
        limit=limit,
        min_passages=min_passages,
    )
    if len(selected_article_ids) < limit:
        print(
            f"Only found {len(selected_article_ids)} processable real articles; "
            f"requested {limit}."
        )

    print("Starting topic inference and vector indexing pipeline (DB mode, real articles)...\n")

    if workers == 1 or len(selected_article_ids) <= 1:
        pipeline = TopicVectorPipeline(
            output_dir=output_dir,
            batch_size=batch_size,
            min_passages=min_passages,
            use_db=True,
            persist_outputs_to_db=persist_outputs_to_db,
        )

        selected_articles = (
            pipeline.session.query(Article)
            .filter(Article.article_id.in_(selected_article_ids))
            .options(
                selectinload(Article.citations).selectinload(Citation.passages)
            )
            .all()
            if selected_article_ids
            else []
        )

        results = pipeline.process_batch(
            articles=selected_articles,
            infer_topic_kwargs=infer_topic_kwargs,
            verbose=True,
        )
        summary = pipeline.get_summary()
    else:
        worker_count = min(workers, len(selected_article_ids))
        chunks = _chunk_ids(selected_article_ids, worker_count)
        print(
            f"[INFO] Parallel mode enabled with {worker_count} workers over {len(chunks)} chunks"
        )

        results = []
        summary = {
            "total_processed": 0,
            "total_skipped": 0,
            "total_failed": 0,
            "output_dir": output_dir,
        }

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_chunk = {
                executor.submit(
                    _process_article_chunk,
                    chunk,
                    idx,
                    output_dir,
                    min_passages,
                    batch_size,
                    persist_outputs_to_db,
                    infer_topic_kwargs,
                ): chunk
                for idx, chunk in enumerate(chunks, start=1)
            }

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    summary["total_processed"] += chunk_result["processed"]
                    summary["total_skipped"] += chunk_result["skipped"]
                    summary["total_failed"] += chunk_result["failed"]
                    results.extend(chunk_result["articles"])
                    print(
                        f"[INFO] Completed chunk with {len(chunk)} articles: "
                        f"processed={chunk_result['processed']}, "
                        f"skipped={chunk_result['skipped']}, "
                        f"failed={chunk_result['failed']}"
                    )
                except Exception as exc:
                    summary["total_failed"] += len(chunk)
                    print(
                        f"[ERROR] Worker failed for chunk of {len(chunk)} articles: {exc}"
                    )

    _write_combined_summary(output_dir=output_dir, summary=summary, results=results)

    print("\n" + "=" * 50)
    print("DB PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Processed: {summary['total_processed']}")
    print(f"Skipped:   {summary['total_skipped']}")
    print(f"Failed:    {summary['total_failed']}")
    print(f"Output:    {summary['output_dir']}")

    if results:
        first = results[0]
        print("\nFirst processed article:")
        print(f"Title: {first['article_title']}")
        print(f"Topic: {first['topic']}")


if __name__ == "__main__":
    main()
