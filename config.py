# -----------------------------
# Environment-based database configuration
# -----------------------------

import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env when variables are not already set."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _parse_libpq_dsn_to_sqlalchemy(dsn: str) -> str:
    """Convert a libpq DSN into a SQLAlchemy URL.

    Example input:
        dbname=wiki user=postgres password=postgres host=localhost port=5432
    """
    parts: dict[str, str] = {}
    for token in dsn.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parts[key.strip()] = value.strip()

    dbname = parts.get("dbname", "wiki")
    user = parts.get("user", "postgres")
    password = parts.get("password", "postgres")
    host = parts.get("host", "localhost")
    port = parts.get("port", "5432")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def _build_db_url() -> str:
    # Load .env from project root so local execution works without shell exports.
    _load_env_file(Path(__file__).resolve().parent / ".env")

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    postgres_dsn = os.getenv("POSTGRES_DSN")
    if postgres_dsn:
        return _parse_libpq_dsn_to_sqlalchemy(postgres_dsn)

    dbname = os.getenv("POSTGRES_DB", "wiki")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


DB_URL = _build_db_url()
