from __future__ import annotations
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_values
from config import DSN

_pool: ThreadedConnectionPool | None = None


def pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 8, dsn=DSN)
    return _pool


def init_db() -> None:
    sql = '''
    create table if not exists wiki_article(
      id bigserial primary key, title text not null, url text unique not null, html text);
    create table if not exists citation(
      id bigserial primary key, wiki_article_id bigint not null references wiki_article(id) on delete cascade,
      source_url text not null, source_host text, anchor_text text, ordinal int,
      unique(wiki_article_id, source_url, ordinal));
    create table if not exists source_document(
      id bigserial primary key, url text unique not null, final_url text, status_code int,
      content_type text, html text, cleaned_text text, content_sha256 text, fetch_ms int);
    create table if not exists article_document(
      wiki_article_id bigint not null references wiki_article(id) on delete cascade,
      source_document_id bigint not null references source_document(id) on delete cascade,
      primary key(wiki_article_id, source_document_id));
    create table if not exists passage(
      id bigserial primary key, source_document_id bigint not null references source_document(id) on delete cascade,
      idx int not null, text text not null, word_count int not null, unique(source_document_id, idx));
    create index if not exists ix_citation_url on citation(source_url);
    create index if not exists ix_passage_doc on passage(source_document_id);
    '''
    conn = pool().getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql)
    finally:
        pool().putconn(conn)


def insert_values(sql: str, rows: list[tuple], page_size: int = 1000) -> list[tuple]:
    if not rows:
        return []
    conn = pool().getconn()
    try:
        with conn, conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)
            try:
                return cur.fetchall()
            except psycopg2.ProgrammingError:
                return []
    finally:
        pool().putconn(conn)
