from __future__ import annotations

from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values

from config import POSTGRES_DSN


@contextmanager
def connect():
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists wiki_article (
                id bigserial primary key,
                title text not null,
                url text not null unique,
                html text not null default ''
            );

            create table if not exists citation (
                wiki_article_id bigint not null references wiki_article(id) on delete cascade,
                source_url text not null,
                source_host text not null default '',
                anchor_text text not null default '',
                ordinal int not null,
                primary key (wiki_article_id, source_url)
            );

            create table if not exists source_document (
                id bigserial primary key,
                url text not null unique,
                final_url text not null,
                status_code int not null,
                content_type text not null,
                html text not null,
                cleaned_text text not null,
                content_sha256 text not null,
                fetch_ms int not null default 0
            );

            create table if not exists article_document (
                wiki_article_id bigint not null references wiki_article(id) on delete cascade,
                source_document_id bigint not null references source_document(id) on delete cascade,
                primary key (wiki_article_id, source_document_id)
            );

            create table if not exists passage (
                source_document_id bigint not null references source_document(id) on delete cascade,
                idx int not null,
                text text not null,
                word_count int not null,
                primary key (source_document_id, idx)
            );

            create index if not exists ix_citation_source_host on citation(source_host);
            create index if not exists ix_source_document_sha on source_document(content_sha256);
            """
        )


def insert_values(sql: str, rows: list[tuple], page_size: int = 500) -> list[tuple]:
    if not rows:
        return []

    with connect() as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=page_size)
        try:
            return cur.fetchall()
        except psycopg2.ProgrammingError:
            return []