#!/usr/bin/env python3

import os
import sys
import time
import html as html_lib
from typing import Optional

import psycopg2
import requests
from bs4 import BeautifulSoup


API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT_SECONDS = 30
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.2


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def build_connection():
    return psycopg2.connect(
        host=get_env("PGHOST", "localhost"),
        port=get_env("PGPORT", "5432"),
        dbname=get_env("PGDATABASE", "wiki"),
        user=get_env("PGUSER", "postgres"),
        password=get_env("PGPASSWORD", ""),
    )


def ensure_text_column(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE wiki_article
            ADD COLUMN IF NOT EXISTS text text NOT NULL DEFAULT '';
            """
        )
    conn.commit()


def fetch_articles(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, url
            FROM wiki_article
            ORDER BY id;
            """
        )
        return cur.fetchall()


def clean_parsed_html_to_text(parsed_html: str) -> str:
    soup = BeautifulSoup(parsed_html, "html.parser")

    for selector in [
        "style",
        "script",
        "table",
        "sup.reference",
        ".reference",
        ".mw-editsection",
        ".navbox",
        ".metadata",
        ".toc",
        ".thumb",
        ".hatnote",
        ".portal",
        ".authority-control",
        ".mw-references-wrap",
        ".sidebar",
        ".ambox",
        ".infobox",
        ".shortdescription",
        ".noprint",
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    text = soup.get_text("\n", strip=True)
    text = html_lib.unescape(text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def fetch_wikipedia_text(title: str) -> str:
    response = requests.get(
        API_URL,
        params={
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent": "wiki-article-text-restorer/1.0"
        },
    )
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        raise RuntimeError(f"Wikipedia API error for '{title}': {payload['error']}")

    parsed_html = payload["parse"]["text"]
    return clean_parsed_html_to_text(parsed_html)


def update_article_text(conn, article_id: int, text: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE wiki_article
            SET text = %s
            WHERE id = %s;
            """,
            (text, article_id),
        )
    conn.commit()


def main() -> int:
    refill_only_empty = "--only-empty" in sys.argv

    conn = build_connection()
    try:
        ensure_text_column(conn)
        articles = fetch_articles(conn)

        total = len(articles)
        success_count = 0
        skipped_count = 0
        failed_count = 0

        for index, (article_id, title, url) in enumerate(articles, start=1):
            if refill_only_empty:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT text
                        FROM wiki_article
                        WHERE id = %s;
                        """,
                        (article_id,),
                    )
                    current_text = cur.fetchone()[0]
                if current_text and current_text.strip():
                    skipped_count += 1
                    print(f"[{index}/{total}] SKIP {article_id} {title}")
                    continue

            try:
                text = fetch_wikipedia_text(title)
                update_article_text(conn, article_id, text)
                success_count += 1
                print(f"[{index}/{total}] OK   {article_id} {title} ({len(text)} chars)")
            except Exception as ex:
                conn.rollback()
                failed_count += 1
                print(f"[{index}/{total}] FAIL {article_id} {title}: {ex}", file=sys.stderr)

            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

        print()
        print(f"Done.")
        print(f"Success: {success_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Failed:  {failed_count}")
        return 0 if failed_count == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())