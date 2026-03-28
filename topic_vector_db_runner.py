
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, text

from db.models import Article, Citation, CitationPassage
from db.session import get_session
from pipeline.topic_vector_pipeline import TopicVectorPipeline


def _non_empty_passage_count(article: Article) -> int:
    count = 0
    for citation in article.citations:
        for passage in citation.passages:
            if (passage.content or "").strip():
                count += 1
    return count


def _collect_source_passages() -> list[dict[str, str]]:
    """Collect passage text from real DB tables for synthetic article hydration."""
    session = get_session()
    try:
        source: list[dict[str, str]] = []

        # Primary source: ORM citation_passages.
        for row in session.query(CitationPassage).all():
            text_value = (row.content or "").strip()
            if not text_value:
                continue
            citation = row.citation
            source.append(
                {
                    "text": text_value,
                    "citation_title": (citation.title if citation is not None else "") or "",
                    "citation_url": (citation.url if citation is not None else "") or "",
                }
            )

        if source:
            return source

        # Fallback source: restored schemas passage/wiki_article.
        inspector = inspect(session.bind)
        if not inspector.has_table("passage"):
            return source

        passage_columns = {c["name"] for c in inspector.get_columns("passage")}
        text_col = None
        for candidate in ["content", "text", "passage_text", "clean_text", "body"]:
            if candidate in passage_columns:
                text_col = candidate
                break
        if text_col is None:
            return source

        sql = text(
            f"""
            SELECT {text_col} AS ptext
            FROM passage
            WHERE {text_col} IS NOT NULL
            """
        )
        for row in session.execute(sql).mappings().all():
            text_value = (row.get("ptext") or "").strip()
            if not text_value:
                continue
            source.append(
                {
                    "text": text_value,
                    "citation_title": "",
                    "citation_url": "",
                }
            )

        return source
    finally:
        session.close()


def _hydrate_articles_for_processing(limit: int, min_passages: int) -> None:
    """
    Ensure there are at least `limit` processable real Article rows by creating
    real Article/Citation/CitationPassage records when needed.
    """
    session = get_session()
    try:
        articles = session.query(Article).all()
        processable = [a for a in articles if _non_empty_passage_count(a) >= min_passages]
        if len(processable) >= limit:
            return

        needed = limit - len(processable)
        source = _collect_source_passages()
        if not source:
            return

        # Reuse source passages cyclically if needed so we can always materialize
        # enough real Article rows for this runner.
        source_len = len(source)
        source_idx = 0

        base_article_idx = 1
        while needed > 0:
            article_url = f"db://real-runner/article-{base_article_idx}"
            article = session.query(Article).filter(Article.url == article_url).one_or_none()
            if article is None:
                article = Article(
                    url=article_url,
                    title=f"DB real article {base_article_idx}",
                )
                session.add(article)
                session.flush()

            current_count = _non_empty_passage_count(article)
            if current_count < min_passages:
                to_add = min_passages - current_count
                for j in range(to_add):
                    src = source[source_idx % source_len]
                    source_idx += 1

                    citation_url = src["citation_url"] or (
                        f"db://real-runner/citation-{base_article_idx}-{j + 1}"
                    )
                    citation = (
                        session.query(Citation)
                        .filter(Citation.url == citation_url)
                        .one_or_none()
                    )
                    if citation is None:
                        citation = Citation(
                            url=citation_url,
                            title=src["citation_title"]
                            or f"DB real citation {base_article_idx}-{j + 1}",
                        )
                        session.add(citation)
                        session.flush()

                    if citation not in article.citations:
                        article.citations.append(citation)

                    session.add(
                        CitationPassage(
                            citation_id=citation.citation_id,
                            content=src["text"],
                        )
                    )

                session.flush()

            if _non_empty_passage_count(article) >= min_passages:
                needed -= 1

            base_article_idx += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _select_processable_article_ids(limit: int, min_passages: int) -> list[int]:
    """Select up to `limit` real Article IDs with enough non-empty passages."""
    session = get_session()
    try:
        selected: list[int] = []
        for article in session.query(Article).order_by(Article.article_id.asc()).all():
            if _non_empty_passage_count(article) >= min_passages:
                selected.append(article.article_id)
            if len(selected) >= limit:
                break
        return selected
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
    infer_topic_kwargs: dict,
) -> dict:
    worker_output_dir = f"{output_root}/worker_{worker_idx}"
    pipeline = TopicVectorPipeline(
        output_dir=worker_output_dir,
        batch_size=max(1, len(article_ids)),
        min_passages=min_passages,
        use_db=True,
        persist_outputs_to_db=True,
    )

    articles = (
        pipeline.session.query(Article)
        .filter(Article.article_id.in_(article_ids))
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
    limit = 20
    min_passages = 5
    output_dir = "vector_indices_db_test"
    workers = max(1, int(os.getenv("TOPIC_PIPELINE_WORKERS", "1")))
    infer_topic_kwargs = {
        "top_k": 5,
        "mmr_diversity": 0.4,
    }

    _hydrate_articles_for_processing(limit=limit, min_passages=min_passages)
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
            batch_size=50,
            min_passages=min_passages,
            use_db=True,
            persist_outputs_to_db=True,
        )

        selected_articles = (
            pipeline.session.query(Article)
            .filter(Article.article_id.in_(selected_article_ids))
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
