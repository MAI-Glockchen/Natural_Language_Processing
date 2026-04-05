# GenArticle Pipeline

GenArticle is a baseline article-generation pipeline that retrieves passages for a target article, builds a constrained prompt, generates a draft article with a local LLM, evaluates the result against the reference Wikipedia article, and stores the output plus metrics in Postgres.

## What the pipeline does

For each article, the pipeline performs these steps:

1. Load settings
   - Reads configuration from environment variables.
   - Important settings include database URL, FAISS index directory, embedding model, LLM endpoint, model name, `top_k`, and output token limit.

2. Resolve which article IDs to process
   - Single article mode: `--article-id <id>`
   - Batch mode from database: `--all [--limit N]`
   - In batch mode, already generated articles can be skipped based on rows already present in the database.

3. Ensure output schema exists
   - Creates the `generated_articles` table if needed.

4. Fetch article bundle from the database
   - Loads the target article metadata and reference article text.
   - Also loads the topic, FAISS index file, and passage count.

5. Retrieve top passages
   - Loads the article’s FAISS index.
   - Builds a retrieval query from article title, topic, and top candidate terms.
   - Embeds the query using `sentence-transformers/all-MiniLM-L6-v2` by default.
   - Searches the FAISS index.
   - Maps row IDs back to passage rows in Postgres.
   - Applies light diversity filtering so one source does not dominate and near-duplicates are reduced.

6. Build the prompt
   - Converts the retrieved passages into a structured context block.
   - Adds strict generation instructions.
   - Requests this exact output structure:

```text
TITLE: <article title>

ARTICLE:
<plain prose article body>
```

7. Generate the article with the LLM
   - Sends the prompt to the configured OpenAI-compatible `/chat/completions` endpoint.
   - Uses the configured model, temperature, and max token budget.
   - If the first response is malformed, one repair attempt may be made.

8. Validate and parse the output
   - Splits the LLM response into generated title and article text.
   - Rejects obviously malformed outputs.

9. Evaluate the generated article
   - Computes:
     - ROUGE-1 F1
     - ROUGE-2 F1
     - ROUGE-L F1
     - BERTScore F1
     - title embedding cosine similarity
     - section count difference
     - article length ratio

10. Store the result in Postgres
    - Writes the generated text, reference text, and metrics into `generated_articles`.
    - Uses upsert semantics so reruns for the same generation configuration overwrite the existing row.

## Inputs

The pipeline expects:

- Postgres database with article metadata, reference articles, passages, and FAISS row mapping
- FAISS index files on disk
- An OpenAI-compatible LLM endpoint
- Environment variables in `.env`

Typical relevant environment variables:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/wiki
ARTICLE_GEN_INDEX_DIR=vector_indices
ARTICLE_GEN_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARTICLE_GEN_TITLE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARTICLE_GEN_EMBED_DEVICE=cpu
ARTICLE_GEN_LLM_BASE_URL=http://127.0.0.1:11434/v1
ARTICLE_GEN_MODEL=phi3.5-16k
ARTICLE_GEN_LLM_TIMEOUT_SECONDS=300
ARTICLE_GEN_TEMPERATURE=0.1
ARTICLE_GEN_MAX_TOKENS=6000
ARTICLE_GEN_TOP_K=50
ARTICLE_GEN_PROMPT_VERSION=baseline_v1
```

## Main modules

- `config.py`
  - Reads settings from environment variables.

- `db.py`
  - Database access and upsert into `generated_articles`.

- `retrieval.py`
  - Loads article bundle, encodes query, searches FAISS, and maps results to passages.

- `prompting.py`
  - Builds the baseline prompt from retrieved passages.

- `generation.py`
  - Calls the LLM and parses the result.

- `evaluation.py`
  - Computes text-quality and structural metrics.

- `run_baseline.py`
  - Orchestrates the full pipeline.

## Execution modes

### Single article

```bash
uv run python -m article_generation.run_baseline --article-id 21
```

### First 100 articles from the database

```bash
uv run python -m article_generation.run_baseline --all --limit 100
```

### Process all available articles

```bash
uv run python -m article_generation.run_baseline --all
```

## Skip-existing behavior

In batch mode, the pipeline can skip already generated rows by checking the database before processing.

The uniqueness identity is effectively:

```text
(article_id, split, method, prompt_version, model_name, top_k)
```

This means rerunning a partially aborted batch resumes from the database state:
- finished articles are skipped
- missing articles are processed

## Error handling

Batch execution does not have to abort on single-article failures.

When enabled in the current runner behavior:
- one article can fail
- the error is logged
- the pipeline continues with the next article

This is useful because malformed model output, missing index files, or temporary endpoint issues should not destroy a long batch run.

## Profiling and timing

The runner supports timing output per article. Typical stages are:

- `fetch_bundle`
- `retrieve`
- `prompt`
- `generate`
- `evaluate`
- `upsert`
- `article_total`

In observed runs, generation is usually by far the dominant cost. Retrieval and database writes are small in comparison, while evaluation is noticeable but much cheaper than long-form generation.

## Practical notes

- Large `top_k` values increase prompt size and usually slow generation a lot.
- Large `ARTICLE_GEN_MAX_TOKENS` values also increase latency.
- If generation quality is unstable, reduce prompt size first.
- If throughput matters, the first levers are usually:
  - lower `ARTICLE_GEN_TOP_K`
  - lower `ARTICLE_GEN_MAX_TOKENS`
  - use a faster local model

## Output table

Each generated article row stores:

- article identity and run settings
- generated title and generated text
- reference title and reference text
- ROUGE metrics
- BERTScore
- title similarity
- section counts
- article length ratio

This makes later comparison across prompts, models, and retrieval settings straightforward.