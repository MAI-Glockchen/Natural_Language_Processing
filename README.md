# Natural Language Processing Project

This repository is a full pipeline for building and evaluating Wikipedia-style generated articles from cited source material.

> [!WARNING]
> This project processes a very large dataset and can be very slow.
> The full pipeline was executed once on GPU, and we do not recommend running it on CPU.

It includes:

- PostgreSQL-backed data model and infrastructure
- Backup/restore tooling for reproducible local datasets
- Topic inference + FAISS index construction
- Retrieval-augmented article generation with a local OpenAI-compatible LLM endpoint
- Automatic evaluation metrics persisted to database

## End-to-end workflow

1. Start PostgreSQL
2. Restore a backup dataset
3. Build topic outputs + FAISS passage indices
4. Run article generation on selected articles
5. Review stored outputs and quality metrics

## Project components

- `embedding_pipeline/`
  - Topic inference and vector indexing pipeline.
  - Builds per-article FAISS indices and `summary.json`.
- `article_generation/`
  - Baseline retrieval + prompt + generation + evaluation pipeline.
  - Stores generated articles and metrics in `generated_articles`.
- `db/`
  - SQLAlchemy models/session for application DB access.
- `db-backup-tools/`
  - Versioned DB backup/restore CLI.
- `infra/init.sql/`
  - Database initialization SQL used by Docker Postgres startup.
- `docu/`
  - PlantUML architecture/data-flow diagrams.

## Data model overview

The project uses two article layers (details in `POSTGRES.md`):

- Source/raw layer centered on `wiki_article`, `citation`, `source_document`, `passage`
- Generation/index layer centered on `articles`, `article_topic_outputs`, `faiss_passage_map`, `generated_articles`

Important mapping rule:

- `articles.article_id` is not equal to `wiki_article.id`
- bridge by URL: `articles.url = wiki_article.url`

## Prerequisites

- Docker Desktop
- Python 3.14+
- `uv` package manager

## Setup

### 1) Start PostgreSQL

```bash
docker compose up -d postgres
```

Check health:

```bash
docker compose ps
```

Open SQL shell:

```bash
docker compose exec postgres psql -U postgres -d wiki
```

### 2) Restore backup data (recommended)

```bash
python -m db-backup-tools restore latest
```

Specific version:

```bash
python -m db-backup-tools restore 2.06
```

List backups:

```bash
python -m db-backup-tools list
```

### 3) Install Python dependencies

```bash
uv sync
```

### 4) Configure environment

The app reads from environment variables and root `.env`.

Minimal local DB config for host execution:

```env
POSTGRES_DB=wiki
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

`DATABASE_URL` takes precedence if set.

## Running each stage

## A) Topic inference + vector index build

Main runner:

```bash
uv run python -m embedding_pipeline.topic_vector_db_runner
```

PowerShell with explicit tuning:

```powershell
$env:TOPIC_PIPELINE_LIMIT='1000'; $env:TOPIC_PIPELINE_BATCH_SIZE='10'; $env:TOPIC_PIPELINE_GPU_ARTICLE_BATCH='10'; $env:TOPIC_EMBED_DEVICE='cpu'; $env:TOPIC_EMBED_BATCH_SIZE='64'; uv run python -m embedding_pipeline.topic_vector_db_runner
```

Outputs:

- `vector_indices/*.faiss`
- `vector_indices/summary.json`
- Optional DB writes to `article_topic_outputs` and `faiss_passage_map` when `TOPIC_PIPELINE_PERSIST_DB=1`

Optional mock/offline demo (no DB needed):

```bash
uv run python -m embedding_pipeline.topic_vector_demo
```

## B) Baseline article generation + evaluation

Single article:

```bash
uv run python -m article_generation.run_baseline --article-id 21
```

Batch from all indexed articles:

```bash
uv run python -m article_generation.run_baseline --all --limit 100
```

Schema-only check:

```bash
uv run python -m article_generation.run_baseline --ensure-schema-only
```

The generation run retrieves passages from FAISS, prompts the LLM, evaluates output, and upserts into `generated_articles`.

## C) Reset FAISS/topic outputs (utility)

```bash
uv run python _faiss_reset.py
```

This clears:

- `faiss_passage_map`
- `article_topic_outputs`
- FAISS files and summaries under `vector_indices` and `vector_indices_db_test`

## Key environment variables

### Topic/index pipeline

- `TOPIC_PIPELINE_LIMIT` (default `10`)
- `TOPIC_PIPELINE_MIN_PASSAGES` (default `5`)
- `TOPIC_PIPELINE_OUTPUT_DIR` (default `vector_indices`)
- `TOPIC_PIPELINE_BATCH_SIZE` (default `10`)
- `TOPIC_PIPELINE_PERSIST_DB` (default `0`)
- `TOPIC_PIPELINE_PREPROCESS_WORKERS`
- `TOPIC_PIPELINE_GPU_ARTICLE_BATCH`
- `TOPIC_EMBED_DEVICE` (`auto`/`cpu`/`cuda`)
- `TOPIC_EMBED_BATCH_SIZE`

### Article generation pipeline

- `DATABASE_URL`
- `ARTICLE_GEN_INDEX_DIR`
- `ARTICLE_GEN_EMBEDDING_MODEL`
- `ARTICLE_GEN_TITLE_EMBEDDING_MODEL`
- `ARTICLE_GEN_EMBED_DEVICE`
- `ARTICLE_GEN_LLM_BASE_URL`
- `ARTICLE_GEN_MODEL`
- `ARTICLE_GEN_LLM_TIMEOUT_SECONDS`
- `ARTICLE_GEN_TEMPERATURE`
- `ARTICLE_GEN_MAX_TOKENS`
- `ARTICLE_GEN_TOP_K`
- `ARTICLE_GEN_PROMPT_VERSION`
- `ARTICLE_GEN_SPLIT_FILE`

## Docker notes

- Inside Docker compose services, DB host is `postgres`.
- Outside Docker (local Python), DB host is usually `localhost`.

Run embedding stage in Docker:

```bash
docker compose up --build embedding_pipeline
```

## Troubleshooting

- Only a few `.faiss` files created: raise `TOPIC_PIPELINE_LIMIT`.
- Local DB connection failure: check `.env` uses `POSTGRES_HOST=localhost` for host runs.
- Container DB connection failure: use `postgres` as host in container env.
- High memory usage: reduce `TOPIC_PIPELINE_BATCH_SIZE`, `TOPIC_PIPELINE_GPU_ARTICLE_BATCH`, `TOPIC_EMBED_BATCH_SIZE`, and preprocessing workers.
- Generation is slow: reduce `ARTICLE_GEN_TOP_K` and `ARTICLE_GEN_MAX_TOKENS`, or use a faster local model.

## Useful commands

Stop services:

```bash
docker compose down
```

Remove DB volume too:

```bash
docker compose down -v
```

## Related documentation

- `POSTGRES.md`
- `embedding_pipeline/PIPELINE_USAGE.md`
- `article_generation/README.md`
- `db-backup-tools/README.md`
- `docu/`
