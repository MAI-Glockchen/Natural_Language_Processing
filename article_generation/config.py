from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SQL_DIR = BASE_DIR / "sql"


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    split_file: Path
    index_dir: Path
    embedding_model_name: str
    title_embedding_model_name: str
    embedding_device: str
    normalize_embeddings: bool
    llm_base_url: str
    model_name: str
    llm_timeout_seconds: float
    generation_temperature: float
    generation_max_tokens: int
    top_k: int
    prompt_version: str


def get_settings() -> Settings:
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/wiki",
    )
    split_file = Path(os.getenv("ARTICLE_GEN_SPLIT_FILE", str(BASE_DIR / "data" / "split.csv"))).resolve()
    index_dir = Path(os.getenv("ARTICLE_GEN_INDEX_DIR", ".")).resolve()
    embedding_model_name = os.getenv(
        "ARTICLE_GEN_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    title_embedding_model_name = os.getenv(
        "ARTICLE_GEN_TITLE_EMBEDDING_MODEL",
        embedding_model_name,
    )
    embedding_device = os.getenv("ARTICLE_GEN_EMBED_DEVICE", "cpu")
    llm_base_url = os.getenv("ARTICLE_GEN_LLM_BASE_URL", "http://localhost:8080/v1")
    model_name = os.getenv("ARTICLE_GEN_MODEL", "local-model")
    llm_timeout_seconds = float(os.getenv("ARTICLE_GEN_LLM_TIMEOUT_SECONDS", "300"))
    generation_temperature = float(os.getenv("ARTICLE_GEN_TEMPERATURE", "0.2"))
    generation_max_tokens = int(os.getenv("ARTICLE_GEN_MAX_TOKENS", "1800"))
    top_k = int(os.getenv("ARTICLE_GEN_TOP_K", "20"))
    prompt_version = os.getenv("ARTICLE_GEN_PROMPT_VERSION", "baseline_v1")
    normalize_embeddings = _parse_bool(os.getenv("ARTICLE_GEN_NORMALIZE_EMBEDDINGS", "true"))
    return Settings(
        database_url=database_url,
        split_file=split_file,
        index_dir=index_dir,
        embedding_model_name=embedding_model_name,
        title_embedding_model_name=title_embedding_model_name,
        embedding_device=embedding_device,
        normalize_embeddings=normalize_embeddings,
        llm_base_url=llm_base_url,
        model_name=model_name,
        llm_timeout_seconds=llm_timeout_seconds,
        generation_temperature=generation_temperature,
        generation_max_tokens=generation_max_tokens,
        top_k=top_k,
        prompt_version=prompt_version,
    )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
