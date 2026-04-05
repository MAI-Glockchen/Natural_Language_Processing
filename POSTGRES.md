# POSTGRES.md

## Purpose

This database stores the pipeline state for generating Wikipedia-style articles from cited source material.

The currently relevant pipeline stages are:

1. Citation Extraction and Document Collection
2. Document Cleaning and Passage Creation
3. Topic Inference
4. Passage Embedding and Indexing
5. Article Generation
6. Evaluation

There are two parallel article representations in the database:

- a **raw/source-oriented article layer** centered on `wiki_article`
- a **generation/index-oriented article layer** centered on `articles`

These two layers are linked by **URL**, not by numeric ID.

---

## High-level schema overview

### Raw/source-oriented layer

Used for source collection, cleaning, passage creation, and reference text for evaluation.

- `wiki_article`
- `citation`
- `article_document`
- `source_document`
- `passage`

### Generation/index-oriented layer

Used for topic inference, FAISS mapping, retrieval, generation, and generated result storage.

- `articles`
- `article_topic_outputs`
- `faiss_passage_map`
- `generated_articles`

### Legacy / currently unused tables

These exist but currently have no meaningful data for the active pipeline:

- `article_citations`
- `citations`
- `citation_passages`

---

## Core design fact

## `articles` and `wiki_article` are matched by URL

This is the most important structural fact in the database.

- `articles.article_id` is **not** the same as `wiki_article.id`
- the bridge is:

```sql
articles.url = wiki_article.url
````

So:

* use `articles` for generation/indexing
* use `wiki_article` for reference text and evaluation
* join them by `url`

---

## Table-by-table documentation

---

## 1. `wiki_article`

### Role

Canonical raw Wikipedia article table.

Stores the original target article metadata and text used as the evaluation reference.

### Columns

* `id bigint primary key`
* `title text not null`
* `url text not null unique`
* `html text not null`
* `text text not null`

### Notes

* `text` is the main reference article body used for evaluation
* every active generated article should be matchable to a `wiki_article` row by URL
* `html` and `text` both exist, but evaluation uses `text`

### Incoming / outgoing relations

Referenced by:

* `article_document.wiki_article_id -> wiki_article.id`
* `citation.wiki_article_id -> wiki_article.id`

---

## 2. `citation`

### Role

Stores extracted citations for each raw Wikipedia article.

### Columns

* `wiki_article_id bigint not null`
* `source_url text not null`
* `source_host text not null`
* `anchor_text text not null`
* `ordinal integer not null`

### Primary key

* `(wiki_article_id, source_url)`

### Notes

* this table belongs to the raw/source-oriented pipeline
* it is not needed directly for the current baseline generation step
* it is useful for provenance and debugging earlier stages

### Foreign keys

* `wiki_article_id -> wiki_article.id`

---

## 3. `article_document`

### Role

Links raw Wikipedia articles to collected source documents.

### Columns

* `wiki_article_id bigint not null`
* `source_document_id bigint not null`

### Primary key

* `(wiki_article_id, source_document_id)`

### Notes

* this is the bridge between target articles and their collected source documents
* this table is part of the raw/source-oriented graph

### Foreign keys

* `wiki_article_id -> wiki_article.id`
* `source_document_id -> source_document.id`

---

## 4. `source_document`

### Role

Stores fetched source documents after collection and cleaning.

### Columns

* `id bigint primary key`
* `url text not null unique`
* `final_url text not null`
* `status_code integer not null`
* `content_type text not null`
* `cleaned_text text not null`
* `content_sha256 text not null`
* `fetch_ms integer not null`

### Notes

* `cleaned_text` is the cleaned source text after document cleaning
* one source document can produce many passages
* this is the parent table for `passage`

### Referenced by

* `article_document.source_document_id`
* `passage.source_document_id`

---

## 5. `passage`

### Role

Stores cleaned text chunks used for retrieval.

### Columns

* `source_document_id bigint not null`
* `idx integer not null`
* `text text not null`
* `word_count integer not null`

### Primary key

* `(source_document_id, idx)`

### Notes

* this is the real text unit used in retrieval
* each passage belongs to exactly one source document
* `idx` is the passage position within a source document
* the FAISS mapping ultimately resolves back into this table

### Foreign keys

* `source_document_id -> source_document.id`

---

## 6. `articles`

### Role

Generation/index-oriented article table.

This is the article identity used by topic inference, FAISS mapping, and generation.

### Columns

* `article_id integer primary key`
* `url varchar unique`
* `title varchar`
* `created_at timestamp`

### Notes

* this is the article identity used in stages 3–5
* it is matched to `wiki_article` by URL
* do not assume `article_id == wiki_article.id`

### Referenced by

* `article_topic_outputs.article_id`
* `faiss_passage_map.article_id`
* `generated_articles.article_id`
* `article_citations.article_id` (legacy path)

---

## 7. `article_topic_outputs`

### Role

Stores the inferred topic and index metadata for each article.

### Columns

* `topic_output_id integer primary key`
* `article_id integer unique`
* `topic varchar`
* `candidates_json text`
* `index_file varchar`
* `index_backend varchar`
* `embedding_dim integer`
* `created_at timestamp`
* `updated_at timestamp`

### Notes

* one row per article
* `topic` is the main inferred query topic used for retrieval
* `candidates_json` stores alternative candidate topics/terms
* `index_file` points to the FAISS index file for that article
* `embedding_dim` is the embedding vector dimension

### Foreign keys

* `article_id -> articles.article_id`

### Practical use

For baseline generation:

1. fetch article row from `articles`
2. fetch `topic` and `index_file` from this table
3. embed the query text
4. search the article-specific FAISS index

---

## 8. `faiss_passage_map`

### Role

Maps FAISS row ids back to database passages.

### Columns

* `map_id integer primary key`
* `article_id integer`
* `faiss_row_id integer`
* `index_file varchar`
* `passage_key varchar`
* `created_at timestamp`

### Unique constraint

* `(article_id, faiss_row_id)`

### Notes

This table is critical.

The FAISS file stores vectors and row positions, but not the actual relational passage text.
This table resolves a FAISS row id back to a real passage.

### Foreign keys

* `article_id -> articles.article_id`

### Important encoding rule

`passage_key` has the form:

```text
<source_document_id>_<idx>
```

Example:

```text
1077_0
```

means:

* `source_document_id = 1077`
* `passage.idx = 0`

### Therefore, mapping to `passage` is:

```sql
split_part(passage_key, '_', 1)::bigint = passage.source_document_id
split_part(passage_key, '_', 2)::int    = passage.idx
```

### Practical use

Retrieval flow:

1. search FAISS index
2. get nearest `faiss_row_id`s
3. resolve them through `faiss_passage_map`
4. join to `passage`
5. retrieve passage text

---

## 9. `generated_articles`

### Role

Stores generated article text and evaluation results for each run configuration.

### Columns

* `run_id bigint primary key`
* `article_id integer not null`
* `split text not null`
* `method text not null`
* `prompt_version text not null`
* `model_name text not null`
* `top_k integer not null`
* `topic text not null`
* `index_file text not null`
* `generated_title text not null`
* `generated_text text not null`
* `reference_title text not null`
* `reference_text text not null`
* `rouge1_f1 double precision`
* `rouge2_f1 double precision`
* `rougel_f1 double precision`
* `bertscore_f1 double precision`
* `title_similarity double precision`
* `section_count_generated integer`
* `section_count_reference integer`
* `section_count_abs_diff integer`
* `article_length_ratio double precision`
* `created_at timestamp not null default now()`

### Unique constraint

* `(article_id, split, method, prompt_version, model_name, top_k)`

### Notes

This table is the experiment/result table.

It supports:

* reruns with different prompt versions
* reruns with different models
* reruns with different `top_k`
* skipping already-generated configurations
* evaluation tracking

### Foreign keys

* `article_id -> articles.article_id`

---

## Legacy / currently unused tables

These exist, but currently have zero active rows in the working generation flow.

---

## 10. `article_citations`

### Role

Legacy bridge between `articles` and `citations`.

### Columns

* `article_id integer not null`
* `citation_id integer not null`

### Notes

Currently unused in the active pipeline.

---

## 11. `citations`

### Role

Legacy citation entity table.

### Columns

* `citation_id integer primary key`
* `url varchar`
* `title varchar`
* `created_at timestamp`

### Notes

Currently unused in the active pipeline.

---

## 12. `citation_passages`

### Role

Legacy passage table linked to `citations`.

### Columns

* `passage_id integer primary key`
* `citation_id integer`
* `content text`

### Notes

Currently unused in the active pipeline.

---

## Active pipeline relationships

## Raw/source-oriented graph

```text
wiki_article
  -> article_document
  -> source_document
  -> passage
```

and separately:

```text
wiki_article
  -> citation
```

---

## Generation/index-oriented graph

```text
articles
  -> article_topic_outputs
  -> faiss_passage_map
  -> generated_articles
```

---

## Cross-layer bridge

```text
articles.url = wiki_article.url
```

This is how generation-oriented article rows are connected to raw reference text.

---

## Baseline retrieval flow

The current baseline generation step works conceptually like this:

1. Load an article from `articles`
2. Load `topic`, `candidates_json`, and `index_file` from `article_topic_outputs`
3. Build a retrieval query
4. Embed the query
5. Search the article-specific FAISS index file
6. Get nearest `faiss_row_id`s
7. Resolve them via `faiss_passage_map`
8. Join to `passage`
9. Feed selected passage texts to the generator
10. Store generated result in `generated_articles`
11. Compare generated text to `wiki_article.text`

---

## Evaluation flow

Evaluation uses:

* generated text from `generated_articles.generated_text`
* reference text from `wiki_article.text`

The join path is:

```text
generated_articles.article_id
  -> articles.article_id
  -> articles.url
  -> wiki_article.url
  -> wiki_article.text
```

### Metrics currently stored

* `rouge1_f1`
* `rouge2_f1`
* `rougel_f1`
* `bertscore_f1`
* `title_similarity`
* `section_count_generated`
* `section_count_reference`
* `section_count_abs_diff`
* `article_length_ratio`

---

## Important implementation rules

### 1. Never join `articles.article_id` to `wiki_article.id`

That is wrong.

Use:

```sql
articles.url = wiki_article.url
```

### 2. `passage_key` must be parsed

Use:

```sql
split_part(fpm.passage_key, '_', 1)::bigint
split_part(fpm.passage_key, '_', 2)::int
```

### 3. `article_topic_outputs` is one row per article

Because `article_id` is unique there.

### 4. `generated_articles` is configuration-specific

The same article can have multiple result rows if:

* prompt version changes
* model changes
* top_k changes
* split changes
* method changes

---

## Useful SQL snippets

## Match generation article to reference article

```sql
SELECT
    a.article_id,
    a.title AS generation_title,
    w.id AS wiki_article_id,
    w.title AS reference_title
FROM articles a
JOIN wiki_article w ON w.url = a.url
WHERE a.article_id = 21;
```

## Load baseline retrieval metadata

```sql
SELECT
    a.article_id,
    a.url,
    a.title,
    ato.topic,
    ato.candidates_json,
    ato.index_file
FROM articles a
JOIN article_topic_outputs ato ON ato.article_id = a.article_id
WHERE a.article_id = 21;
```

## Resolve FAISS rows to real passages

```sql
SELECT
    fpm.faiss_row_id,
    fpm.passage_key,
    p.source_document_id,
    p.idx,
    p.text
FROM faiss_passage_map fpm
JOIN passage p
  ON p.source_document_id = split_part(fpm.passage_key, '_', 1)::bigint
 AND p.idx = split_part(fpm.passage_key, '_', 2)::int
WHERE fpm.article_id = 21
ORDER BY fpm.faiss_row_id;
```

## Count all indexed passages for an article

```sql
SELECT COUNT(*) AS available_passage_count
FROM faiss_passage_map
WHERE article_id = 21;
```

## Inspect generated result rows

```sql
SELECT
    run_id,
    article_id,
    split,
    method,
    prompt_version,
    model_name,
    top_k,
    rouge1_f1,
    rouge2_f1,
    rougel_f1,
    bertscore_f1,
    article_length_ratio,
    created_at
FROM generated_articles
ORDER BY run_id DESC;
```

---

## Current practical interpretation

For the active article-generation workflow, the most important tables are:

* `articles`
* `article_topic_outputs`
* `faiss_passage_map`
* `passage`
* `wiki_article`
* `generated_articles`

Everything else is either upstream raw-pipeline support or legacy.

---

## Summary

### Use these for retrieval/generation

* `articles`
* `article_topic_outputs`
* `faiss_passage_map`
* `passage`

### Use this for evaluation reference

* `wiki_article`

### Use this for experiment results

* `generated_articles`

### Bridge between article layers

* `articles.url = wiki_article.url`

### Bridge between FAISS and passage text

* `faiss_passage_map.passage_key -> passage(source_document_id, idx)`

### Ignore for current baseline unless needed later

* `article_citations`
* `citations`
* `citation_passages`

