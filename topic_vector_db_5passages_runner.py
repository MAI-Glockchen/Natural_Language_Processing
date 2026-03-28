# -----------------------------
# DB smoke test: read 5 passages and run topic/vector pipeline
# Run with: python topic_vector_db_5passages_runner.py
# -----------------------------

from db.models import Article, CitationPassage
from db.session import get_session
from pipeline.topic_vector_pipeline import TopicVectorPipeline
from sqlalchemy import inspect, text


def _pick_first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _build_article_payload_with_n_passages(num_passages: int = 5) -> dict | None:
    session = get_session()
    try:
        # Prefer a single-article sample when possible.
        articles = session.query(Article).all()
        for article in articles:
            selected = []
            for citation in article.citations:
                for passage in citation.passages:
                    if not passage.content:
                        continue
                    selected.append(
                        {
                            "passage_id": f"{citation.citation_id}_{passage.passage_id}",
                            "text": passage.content,
                            "citation_url": citation.url or "",
                            "citation_title": citation.title or "",
                        }
                    )
                    if len(selected) >= num_passages:
                        break
                if len(selected) >= num_passages:
                    break

            if len(selected) >= num_passages:
                return {
                    "article_url": article.url,
                    "article_title": article.title,
                    "passage_data": selected,
                }

        # Fallback for sparse DBs: take any N passages across citations.
        selected = []
        rows = session.query(CitationPassage).limit(num_passages).all()
        for row in rows:
            citation = row.citation
            selected.append(
                {
                    "passage_id": f"{citation.citation_id}_{row.passage_id}",
                    "text": row.content or "",
                    "citation_url": citation.url or "",
                    "citation_title": citation.title or "",
                }
            )

        if len(selected) >= num_passages:
            return {
                "article_url": "db://mixed-passages-smoke-test",
                "article_title": "DB mixed passages smoke test",
                "passage_data": selected,
            }

        # Fallback for restored backups that store data in generic passage/wiki_article
        # tables instead of the ORM-linked articles/citations/citation_passages triplet.
        inspector = inspect(session.bind)
        if inspector.has_table("passage"):
            passage_columns = {c["name"] for c in inspector.get_columns("passage")}

            id_col = _pick_first_existing(passage_columns, ["passage_id", "id"])
            text_col = _pick_first_existing(
                passage_columns,
                ["content", "text", "passage_text", "clean_text", "body"],
            )
            article_fk_col = _pick_first_existing(
                passage_columns,
                ["wiki_article_id", "article_id", "article_fk"],
            )

            if text_col is not None:
                pid_expr = id_col if id_col is not None else "NULL"
                fk_expr = article_fk_col if article_fk_col is not None else "NULL"
                sql = text(
                    f"""
                    SELECT {pid_expr} AS pid, {text_col} AS ptext, {fk_expr} AS article_fk
                    FROM passage
                    WHERE {text_col} IS NOT NULL
                    LIMIT :limit_n
                    """
                )
                rows = session.execute(sql, {"limit_n": num_passages}).mappings().all()

                article_url = "db://passage-table-smoke-test"
                article_title = "DB passage table smoke test"

                if rows and rows[0].get("article_fk") is not None and inspector.has_table(
                    "wiki_article"
                ):
                    article_columns = {
                        c["name"] for c in inspector.get_columns("wiki_article")
                    }
                    wiki_id_col = _pick_first_existing(
                        article_columns,
                        ["wiki_article_id", "article_id", "id"],
                    )
                    wiki_title_col = _pick_first_existing(
                        article_columns,
                        ["title", "article_title", "name"],
                    )
                    wiki_url_col = _pick_first_existing(
                        article_columns,
                        ["url", "article_url", "link"],
                    )

                    if wiki_id_col is not None:
                        title_expr = (
                            wiki_title_col if wiki_title_col is not None else "NULL"
                        )
                        url_expr = wiki_url_col if wiki_url_col is not None else "NULL"
                        article_sql = text(
                            f"""
                            SELECT {title_expr} AS atitle, {url_expr} AS aurl
                            FROM wiki_article
                            WHERE {wiki_id_col} = :article_fk
                            LIMIT 1
                            """
                        )
                        article_row = session.execute(
                            article_sql, {"article_fk": rows[0]["article_fk"]}
                        ).mappings().first()
                        if article_row is not None:
                            article_title = article_row.get("atitle") or article_title
                            article_url = article_row.get("aurl") or article_url

                selected = []
                for row in rows:
                    text_value = (row.get("ptext") or "").strip()
                    if not text_value:
                        continue
                    selected.append(
                        {
                            "passage_id": str(row.get("pid") or "unknown"),
                            "text": text_value,
                            "citation_url": "",
                            "citation_title": "",
                        }
                    )

                if len(selected) >= num_passages:
                    return {
                        "article_url": article_url,
                        "article_title": article_title,
                        "passage_data": selected,
                    }

        return None
    finally:
        session.close()


def main() -> None:
    payload = _build_article_payload_with_n_passages(num_passages=5)
    if payload is None:
        print("No article with at least 5 passages was found in the DB.")
        return

    pipeline = TopicVectorPipeline(
        output_dir="vector_indices_db_test",
        batch_size=1,
        min_passages=5,
        use_db=True,
        persist_outputs_to_db=True,
    )

    print("Running pipeline on 5 passages from DB...\n")
    results = pipeline.process_mock_articles(
        mock_articles=[payload],
        infer_topic_kwargs={
            "top_k": 5,
            "mmr_diversity": 0.4,
        },
        verbose=True,
    )

    summary = pipeline.get_summary()
    print("\n" + "=" * 50)
    print("DB 5-PASSAGE SMOKE TEST SUMMARY")
    print("=" * 50)
    print(f"Processed: {summary['total_processed']}")
    print(f"Skipped:   {summary['total_skipped']}")
    print(f"Failed:    {summary['total_failed']}")
    print(f"Output:    {summary['output_dir']}")

    if results["articles"]:
        item = results["articles"][0]
        print("\nResult:")
        print(f"Article: {item['article_title']}")
        print(f"Topic:   {item['topic']}")
        print(f"Passages used: {item['num_passages']}")


if __name__ == "__main__":
    main()
