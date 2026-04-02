# Topic Inference Pipeline Guide

This guide explains what the topic inference pipeline does, how to run it, and how to read the top passages for a query such as `musician`.

## What the Pipeline Does

The pipeline in `embedding_pipeline/topic_vector_pipeline.py` processes articles and their citation passages, then:

1. Extracts passage text from article citations.
2. Prepares topic candidates with CPU-side preprocessing.
3. Infers a compact topic label and ranked topic candidates.
4. Builds a FAISS vector index per article from passage embeddings.
5. Writes output files to `vector_indices/`.
6. Optionally persists topic/index mappings to Postgres.

Main topic inference logic is in `embedding_pipeline/topic_inference.py`.

## Output Files

A run creates:

- `vector_indices/<n>_article.faiss`
- `vector_indices/summary.json`

## Run the DB Pipeline

Use this when Postgres is running and has article / citation / passage data.

```bash
export DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/wiki'

TOPIC_PIPELINE_LIMIT=1000000 TOPIC_PIPELINE_BATCH_SIZE=10 TOPIC_PIPELINE_GPU_ARTICLE_BATCH=10 TOPIC_EMBED_DEVICE=cuda TOPIC_EMBED_BATCH_SIZE=64 uv run python -m embedding_pipeline.topic_vector_db_runner
```

Notes:

- `TOPIC_PIPELINE_LIMIT` controls how many processable articles are selected.
- `TOPIC_PIPELINE_BATCH_SIZE` controls the outer batch size.
- `TOPIC_PIPELINE_GPU_ARTICLE_BATCH` controls how many articles are embedded together on the GPU.
- `TOPIC_EMBED_BATCH_SIZE` controls SentenceTransformer batch size.
- `TOPIC_PIPELINE_PERSIST_DB=1` stores topic output and FAISS passage mapping in DB.

## Run the Mock Pipeline

Use this for offline smoke tests without DB article reads:

```bash
python -m embedding_pipeline.topic_vector_demo
```

Mock payloads are in `embedding_pipeline/mock_articles.py`.

## Read Top 5 Passages for "musician"

Use the sample reader script:

```bash
export DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/wiki'
python -m embedding_pipeline.topic_passage_search_demo --query musician --top-k 5
```

What it does:

1. Reads `vector_indices/summary.json`.
2. Loads each referenced `.faiss` index.
3. Searches by semantic similarity to the query.
4. Resolves FAISS rows back to original passage text through DB mapping tables.
5. Prints top passages with score and article title.

## Useful Variations

Search for another topic:

```bash
python -m embedding_pipeline.topic_passage_search_demo --query "rock musician" --top-k 5
```

Use a custom summary/index location:

```bash
python -m embedding_pipeline.topic_passage_search_demo   --query musician   --top-k 5   --summary vector_indices/summary.json   --index-dir vector_indices
```

## Current Defaults

Current stable defaults are:

- `TOPIC_PIPELINE_BATCH_SIZE=10`
- `TOPIC_PIPELINE_GPU_ARTICLE_BATCH=10`
- `TOPIC_EMBED_BATCH_SIZE=64`

And inside the pipeline:

- `preprocess_workers = max(1, cpu_count - 2)`

These were chosen as a stable baseline for a 16 GB RAM system.

## Performance Model

At a high level:

- **CPU worker processes** handle candidate preparation and fuzzy deduplication.
- **GPU embedding** runs in article micro-batches.
- **Topic finalization / clustering** then completes per article.
- **FAISS** stores passage vectors for later search.

This means:

- increasing `TOPIC_PIPELINE_PREPROCESS_WORKERS` mainly affects CPU throughput and RAM
- increasing `TOPIC_PIPELINE_GPU_ARTICLE_BATCH` affects both RAM and VRAM
- increasing `TOPIC_EMBED_BATCH_SIZE` affects GPU throughput and some host memory

## Quick Troubleshooting

- If no results are returned, ensure:
  - `vector_indices/summary.json` exists
  - `.faiss` files exist in `vector_indices/`
  - DB is reachable via `DATABASE_URL`
  - `faiss_passage_map` has rows for processed articles if DB lookup is expected

- If embeddings fail:
  - ensure required model dependencies are installed
  - check `torch.cuda.is_available()`
  - verify `TOPIC_EMBED_DEVICE`

- If the run gets killed or VS Code crashes:
  - reduce `TOPIC_PIPELINE_PREPROCESS_WORKERS`
  - reduce `TOPIC_PIPELINE_BATCH_SIZE`
  - reduce `TOPIC_PIPELINE_GPU_ARTICLE_BATCH`
  - reduce `TOPIC_EMBED_BATCH_SIZE`

- If only 10 articles are processed:
  - set `TOPIC_PIPELINE_LIMIT` explicitly higher

## Verifying GPU Usage

Useful checks:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

```bash
watch -n 1 nvidia-smi
```

If the Python process appears in `nvidia-smi` with compute usage / allocated VRAM, CUDA is active.