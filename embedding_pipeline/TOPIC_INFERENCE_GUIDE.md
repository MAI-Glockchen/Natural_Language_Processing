# Topic Inference Pipeline Guide

This guide explains what the topic inference pipeline does, how to run it, and how to read the top 5 passages for a query such as `musician`.

## What the Pipeline Does

The pipeline in `embedding_pipeline/topic_vector_pipeline.py` processes articles and their citation passages, then:

1. Extracts passage text from article citations.
2. Infers a compact topic label and top topic candidates via `infer_topic()`.
3. Builds a FAISS vector index per article from passage embeddings.
4. Writes output files to `vector_indices/`.
5. Optionally persists topic/index mappings to Postgres.

Main topic inference logic is in `embedding_pipeline/topic_inference.py`.

## Output Files

A run creates:

- `vector_indices/<n>_article.faiss` (one index per processed article)
- `vector_indices/summary.json` (topics, candidates, metadata)

## Run the DB Pipeline

Use this when Postgres is running and has article/citation data.

```powershell
$env:DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/wiki'
$env:TOPIC_PIPELINE_LIMIT='10'
$env:TOPIC_PIPELINE_PERSIST_DB='1'
python -m embedding_pipeline.topic_vector_db_runner
```

Notes:

- `TOPIC_PIPELINE_LIMIT` controls how many articles are processed.
- `TOPIC_PIPELINE_PERSIST_DB=1` stores topic output and FAISS passage mapping in DB.

## Run the Mock Pipeline

Use this for offline smoke tests without DB article reads:

```powershell
python -m embedding_pipeline.topic_vector_demo
```

Mock payloads are in `embedding_pipeline/mock_articles.py`.

## Read Top 5 Passages for "musician"

Use the sample reader script:

```powershell
$env:DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/wiki'
python -m embedding_pipeline.topic_passage_search_demo --query musician --top-k 5
```

What it does:

1. Reads `vector_indices/summary.json`.
2. Loads each referenced `.faiss` index.
3. Searches by semantic similarity to the query.
4. Resolves FAISS rows back to original passage text through DB mapping tables.
5. Prints top 5 passages with score and article title.

## Useful Variations

Search for another topic:

```powershell
python -m embedding_pipeline.topic_passage_search_demo --query "rock musician" --top-k 5
```

Use a custom summary/index location:

```powershell
python -m embedding_pipeline.topic_passage_search_demo --query musician --top-k 5 --summary vector_indices/summary.json --index-dir vector_indices
```

## Quick Troubleshooting

- If no results are returned, ensure:
  - `vector_indices/summary.json` exists
  - `.faiss` files exist in `vector_indices/`
  - DB is reachable via `DATABASE_URL`
  - `faiss_passage_map` has rows for processed articles
- If embeddings fail, ensure required model dependencies are installed and available.
