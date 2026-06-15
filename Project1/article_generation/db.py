from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, RowMapping

from .config import SQL_DIR, Settings
from .models import GeneratedArticleRecord


_SCHEMA_FILE = SQL_DIR / "create_generated_articles.sql"


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_engine(settings.database_url, future=True)

    @property
    def engine(self) -> Engine:
        return self._engine

    def ensure_schema(self) -> None:
        sql = _read_sql_file(_SCHEMA_FILE)
        with self._engine.begin() as connection:
            for statement in _split_sql_statements(sql):
                connection.execute(text(statement))

    def fetch_all(self, sql: str, params: dict[str, Any] | None = None) -> list[RowMapping]:
        with self._engine.connect() as connection:
            result = connection.execute(text(sql), params or {})
            return list(result.mappings())

    def fetch_one(self, sql: str, params: dict[str, Any] | None = None) -> RowMapping | None:
        with self._engine.connect() as connection:
            result = connection.execute(text(sql), params or {})
            return result.mappings().first()

    def upsert_generated_article(self, record: GeneratedArticleRecord) -> None:
        sql = text(
            """
            INSERT INTO generated_articles (
                article_id,
                split,
                method,
                prompt_version,
                model_name,
                top_k,
                topic,
                index_file,
                generated_title,
                generated_text,
                reference_title,
                reference_text,
                rouge1_f1,
                rouge2_f1,
                rougel_f1,
                bertscore_f1,
                title_similarity,
                section_count_generated,
                section_count_reference,
                section_count_abs_diff,
                article_length_ratio
            )
            VALUES (
                :article_id,
                :split,
                :method,
                :prompt_version,
                :model_name,
                :top_k,
                :topic,
                :index_file,
                :generated_title,
                :generated_text,
                :reference_title,
                :reference_text,
                :rouge1_f1,
                :rouge2_f1,
                :rougel_f1,
                :bertscore_f1,
                :title_similarity,
                :section_count_generated,
                :section_count_reference,
                :section_count_abs_diff,
                :article_length_ratio
            )
            ON CONFLICT (article_id, split, method, prompt_version, model_name, top_k)
            DO UPDATE SET
                topic = EXCLUDED.topic,
                index_file = EXCLUDED.index_file,
                generated_title = EXCLUDED.generated_title,
                generated_text = EXCLUDED.generated_text,
                reference_title = EXCLUDED.reference_title,
                reference_text = EXCLUDED.reference_text,
                rouge1_f1 = EXCLUDED.rouge1_f1,
                rouge2_f1 = EXCLUDED.rouge2_f1,
                rougel_f1 = EXCLUDED.rougel_f1,
                bertscore_f1 = EXCLUDED.bertscore_f1,
                title_similarity = EXCLUDED.title_similarity,
                section_count_generated = EXCLUDED.section_count_generated,
                section_count_reference = EXCLUDED.section_count_reference,
                section_count_abs_diff = EXCLUDED.section_count_abs_diff,
                article_length_ratio = EXCLUDED.article_length_ratio,
                created_at = NOW()
            """
        )
        payload = {
            "article_id": record.article_id,
            "split": record.split,
            "method": record.method,
            "prompt_version": record.prompt_version,
            "model_name": record.model_name,
            "top_k": record.top_k,
            "topic": record.topic,
            "index_file": record.index_file,
            "generated_title": record.generated_title,
            "generated_text": record.generated_text,
            "reference_title": record.reference_title,
            "reference_text": record.reference_text,
            "rouge1_f1": record.rouge1_f1,
            "rouge2_f1": record.rouge2_f1,
            "rougel_f1": record.rougel_f1,
            "bertscore_f1": record.bertscore_f1,
            "title_similarity": record.title_similarity,
            "section_count_generated": record.section_count_generated,
            "section_count_reference": record.section_count_reference,
            "section_count_abs_diff": record.section_count_abs_diff,
            "article_length_ratio": record.article_length_ratio,
        }
        with self._engine.begin() as connection:
            connection.execute(sql, payload)


def _read_sql_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_sql_statements(sql: str) -> Sequence[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
