from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
BASE_DIR = Path(__file__).resolve().parent
SQL_DIR = BASE_DIR / "sql"
DEFAULT_SPLIT_FILE = PROJECT_DIR / "data" / "split.csv"


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    split_file: Path
    index_dir: Path
    top_k: int
    model_name: str
    base_url: str
    prompt_version: str
    method_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://postgres:postgres@localhost:5432/wiki",
        )
        split_file = Path(os.getenv("ARTICLE_GEN_SPLIT_FILE", str(DEFAULT_SPLIT_FILE)))
        index_dir = Path(os.getenv("ARTICLE_GEN_INDEX_DIR", ".")).resolve()
        top_k = int(os.getenv("ARTICLE_GEN_TOP_K", "20"))
        model_name = os.getenv("ARTICLE_GEN_MODEL", "unset-model")
        base_url = os.getenv("ARTICLE_GEN_BASE_URL", "http://localhost:11434")
        prompt_version = os.getenv("ARTICLE_GEN_PROMPT_VERSION", "baseline_v1")
        method_name = os.getenv("ARTICLE_GEN_METHOD", "baseline")
        return cls(
            database_url=database_url,
            split_file=split_file,
            index_dir=index_dir,
            top_k=top_k,
            model_name=model_name,
            base_url=base_url,
            prompt_version=prompt_version,
            method_name=method_name,
        )


def load_settings() -> Settings:
    return Settings.from_env()
