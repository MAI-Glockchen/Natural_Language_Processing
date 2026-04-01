"""DB batch runner for topic inference and vector indexing.

Run with: python -m embedding_pipeline.topic_vector_db_runner
"""

import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from db.session import get_session
from embedding_pipeline.topic_vector_pipeline import TopicVectorPipeline


def _select_processable_wiki_article_ids(limit: int, min_passages: int) -> list[int]:
    """Select up to `limit` wiki_article IDs with enough non-empty passages."""
    session = get_session()
    try:
        rows = session.execute(
            text(
                """
                SELECT wa.id
                FROM wiki_article wa
                JOIN article_document ad
                  ON ad.wiki_article_id = wa.id
                JOIN passage p
                  ON p.source_document_id = ad.source_document_id
                WHERE p.text IS NOT NULL
                  AND length(trim(p.text)) > 0
                GROUP BY wa.id
                HAVING count(*) >= :min_passages
                ORDER BY count(*) ASC, wa.id ASC
                LIMIT :limit
                """
            ),
            {"min_passages": min_passages, "limit": limit},
        ).fetchall()
        return [int(row[0]) for row in rows]
    finally:
        session.close()


def _fetch_backup_articles_payload(article_ids: list[int]) -> list[dict]:
    """Build TopicVectorPipeline-compatible payloads from backup schema tables."""
    if not article_ids:
        return []

    session = get_session()
    try:
        article_rows = session.execute(
            text(
                """
                SELECT id, url, title
                FROM wiki_article
                WHERE id = ANY(:article_ids)
                """
            ),
            {"article_ids": article_ids},
        ).fetchall()

        citation_rows = session.execute(
            text(
                """
                SELECT wiki_article_id, source_url, anchor_text, ordinal
                FROM citation
                WHERE wiki_article_id = ANY(:article_ids)
                """
            ),
            {"article_ids": article_ids},
        ).fetchall()

        passage_rows = session.execute(
            text(
                """
                SELECT ad.wiki_article_id, p.source_document_id, p.idx, p.text
                FROM article_document ad
                JOIN passage p
                  ON p.source_document_id = ad.source_document_id
                WHERE ad.wiki_article_id = ANY(:article_ids)
                  AND p.text IS NOT NULL
                  AND length(trim(p.text)) > 0
                ORDER BY ad.wiki_article_id, p.source_document_id, p.idx
                """
            ),
            {"article_ids": article_ids},
        ).fetchall()

        citation_map: dict[int, dict[int, tuple[str, str]]] = {}
        for wiki_article_id, source_url, anchor_text, ordinal in citation_rows:
            if wiki_article_id is None or ordinal is None:
                continue
            ordinal_map = citation_map.setdefault(int(wiki_article_id), {})
            ordinal_map[int(ordinal)] = (
                str(source_url or ""),
                str(anchor_text or ""),
            )

        passages_map: dict[int, list[dict]] = {}
        for wiki_article_id, source_document_id, idx, content in passage_rows:
            wa_id = int(wiki_article_id)
            ordinal = int(idx) + 1 if idx is not None else None
            citation_url = ""
            citation_title = ""

            if ordinal is not None:
                entry = citation_map.get(wa_id, {}).get(ordinal)
                if entry is not None:
                    citation_url, citation_title = entry

            passages_map.setdefault(wa_id, []).append(
                {
                    "passage_id": f"{int(source_document_id)}_{int(idx)}",
                    "text": str(content),
                    "citation_url": citation_url,
                    "citation_title": citation_title,
                }
            )

        payload = []
        for article_id, article_url, article_title in article_rows:
            wa_id = int(article_id)
            payload.append(
                {
                    "article_url": str(article_url or f"wiki_article:{wa_id}"),
                    "article_title": str(article_title or f"wiki article {wa_id}"),
                    "passage_data": passages_map.get(wa_id, []),
                }
            )

        return payload
    finally:
        session.close()


def _write_combined_summary(output_dir: str, summary: dict, results: list[dict]) -> None:
    """Write a root-level merged summary.json."""
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
    limit = min(10, int(os.getenv("TOPIC_PIPELINE_LIMIT", "10")))
    min_passages = int(os.getenv("TOPIC_PIPELINE_MIN_PASSAGES", "5"))
    output_dir = os.getenv("TOPIC_PIPELINE_OUTPUT_DIR", "vector_indices")
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

    selected_article_ids = _select_processable_wiki_article_ids(
        limit=limit,
        min_passages=min_passages,
    )
    if len(selected_article_ids) < limit:
        print(
            f"Only found {len(selected_article_ids)} processable real articles; "
            f"requested {limit}."
        )

    print("Starting topic inference and vector indexing pipeline (backup DB mode)...\n")

    pipeline = TopicVectorPipeline(
        output_dir=output_dir,
        batch_size=batch_size,
        min_passages=min_passages,
        use_db=False,
        persist_outputs_to_db=persist_outputs_to_db,
    )

    payload_articles = _fetch_backup_articles_payload(selected_article_ids)
    results_bundle = pipeline.process_mock_articles(
        mock_articles=payload_articles,
        infer_topic_kwargs=infer_topic_kwargs,
        verbose=True,
    )
    results = results_bundle.get("articles", [])
    summary = pipeline.get_summary()

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