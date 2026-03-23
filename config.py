from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "dbname=wiki user=postgres password=postgres host=localhost port=5432",
)

MAX_ARTICLES = _env_int("MAX_ARTICLES", 25)
MIN_CITATIONS = _env_int("MIN_CITATIONS", 20)
MAX_CITATIONS_PER_ARTICLE = _env_int("MAX_CITATIONS_PER_ARTICLE", 180)

WORKERS = _env_int("WORKERS", 32)
REQUEST_TIMEOUT = (_env_float("CONNECT_TIMEOUT", 3.0), _env_float("READ_TIMEOUT", 8.0))

PASSAGE_TARGET_WORDS = _env_int("PASSAGE_TARGET_WORDS", 100)
PASSAGE_MAX_WORDS = _env_int("PASSAGE_MAX_WORDS", 140)
PASSAGE_MIN_WORDS = _env_int("PASSAGE_MIN_WORDS", 60)

POPULAR_ARTICLES_FILE = os.getenv("POPULAR_ARTICLES_FILE", "popular_articles_2025.txt")

USER_AGENT = os.getenv(
    "USER_AGENT",
    "citation-pipeline/0.1 (+https://example.invalid; research project)",
)