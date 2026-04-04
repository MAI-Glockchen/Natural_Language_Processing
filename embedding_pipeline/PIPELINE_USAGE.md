# Topic Inference & Vector Indexing Pipeline

A reusable pipeline for processing Wikipedia articles from the database, inferring topics from citation passages, and building FAISS vector indices for efficient similarity search.

## Features

✅ **Batch Processing**: Process articles in configurable batches  
✅ **Topic Inference**: Automatically infer main topics from cited passages  
✅ **Vector Indexing**: Build FAISS indices for fast similarity search  
✅ **Optimization**: Reuse passage embeddings to avoid redundant computations  
✅ **Persistence**: Save indices and metadata to disk or optionally persist mappings to DB  
✅ **Progress Tracking**: Monitor processing with detailed logging  
✅ **Parallel Preprocessing**: CPU-bound preprocessing runs in multiple worker processes  
✅ **GPU Micro-Batching**: Embeddings are computed on the GPU in article chunks for better throughput / memory balance  

---

## Quick Start

### Basic Usage

```python
from embedding_pipeline.topic_vector_pipeline import TopicVectorPipeline

pipeline = TopicVectorPipeline(
    output_dir="vector_indices",
    batch_size=10,
    min_passages=5,
)

results = pipeline.process_all(
    limit=100,
    verbose=True,
)

summary = pipeline.get_summary()
print(f"Processed: {summary['total_processed']} articles")
print(f"Output: {summary['output_dir']}")
```

### Search in a Vector Index

```python
index = pipeline.load_index(article_idx=0)

query = "climate change impact"
results = index.search(query, top_k=5)

for result in results:
    print(f"Similarity: {result['similarity_query_to_passage_score']:.4f}")
    print(f"Text: {result['text'][:100]}...")
```

### Process Single Article

```python
from db.session import get_session
from db.models import Article
from embedding_pipeline.topic_vector_pipeline import TopicVectorPipeline

session = get_session()
article = session.query(Article).first()

pipeline = TopicVectorPipeline(output_dir="vector_indices", batch_size=10, min_passages=5)

metadata = pipeline.process_article(
    article,
    infer_topic_kwargs={"top_k": 5}
)

if metadata:
    print(f"Topic: {metadata['topic']}")
    print(f"Candidates: {metadata['candidates']}")
```

---

## Configuration

### TopicVectorPipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"vector_indices"` | Directory to save FAISS indices |
| `batch_size` | int | `10` | Articles per outer batch |
| `min_passages` | int | `5` | Minimum passages required to process an article |
| `use_db` | bool | `True` | Use DB-backed article loading |
| `persist_outputs_to_db` | bool | `True` | Persist topic output and FAISS mapping rows |

### Runtime Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOPIC_PIPELINE_LIMIT` | `10` | Max number of processable articles selected by the DB runner |
| `TOPIC_PIPELINE_BATCH_SIZE` | `10` | Outer batch size used by the DB runner |
| `TOPIC_PIPELINE_PREPROCESS_WORKERS` | `max(1, cpu_count - 2)` | Number of CPU worker processes for preprocessing |
| `TOPIC_PIPELINE_GPU_ARTICLE_BATCH` | `10` | Number of articles embedded together on the GPU at once |
| `TOPIC_PIPELINE_MIN_PASSAGES` | `5` | Minimum passages required per article |
| `TOPIC_PIPELINE_OUTPUT_DIR` | `"vector_indices"` | Output directory |
| `TOPIC_PIPELINE_PERSIST_DB` | `0` | Persist topic/index mapping outputs to DB when set to true-like values |
| `TOPIC_EMBED_DEVICE` | `auto` | Embedding device: `auto`, `cpu`, or `cuda` |
| `TOPIC_EMBED_BATCH_SIZE` | `64` | SentenceTransformer embedding batch size |

### infer_topic() Options

Common parameters to pass in `infer_topic_kwargs`:

```python
infer_topic_kwargs = {
    "top_k": 5,
    "mmr_diversity": 0.4,
    "concept_cluster_threshold": 0.55,
    "position_decay": 0.85,
}
```

---

## Output Structure

### Directory Layout

```text
vector_indices/
├── 0_article.faiss
├── 1_article.faiss
├── 2_article.faiss
├── ...
└── summary.json
```

### summary.json Example

```json
{
  "processed": 3,
  "skipped": 0,
  "failed": 0,
  "timestamp": "2026-04-02T12:30:00",
  "articles": [
    {
      "article_url": "https://en.wikipedia.org/wiki/Machine_learning",
      "article_title": "Machine Learning",
      "topic": "artificial intelligence",
      "candidates": [
        ["machine learning", 8.5],
        ["artificial intelligence", 7.2],
        ["deep learning", 6.8]
      ],
      "num_passages": 42,
      "index_file": "0_article.faiss",
      "index_backend": "faiss",
      "embedding_dim": 384
    }
  ]
}
```

---

## Workflow

```text
Database Articles
       ↓
Extract Passages
       ↓
CPU Preprocessing (parallel worker processes)
       ↓
GPU Embedding (article micro-batches)
       ↓
Topic Finalization / Clustering
       ↓
Build FAISS Index
       ↓
Save to Disk / Optional DB Persistence
```

---

## Example: DB Runner

```bash
TOPIC_PIPELINE_LIMIT=1000000 TOPIC_PIPELINE_BATCH_SIZE=10 TOPIC_PIPELINE_GPU_ARTICLE_BATCH=10 TOPIC_EMBED_DEVICE=cuda TOPIC_EMBED_BATCH_SIZE=64 uv run python -m embedding_pipeline.topic_vector_db_runner
```

### Notes on Defaults

These defaults were chosen as a stable baseline for a 16 GB RAM machine:

- `TOPIC_PIPELINE_BATCH_SIZE=10`
- `TOPIC_PIPELINE_GPU_ARTICLE_BATCH=10`
- `TOPIC_EMBED_BATCH_SIZE=64`

`TOPIC_PIPELINE_PREPROCESS_WORKERS` defaults to `max(1, cpu_count - 2)`.

---

## Performance Tips

1. **Increase `TOPIC_EMBED_BATCH_SIZE` first** if the GPU has headroom.
2. **Reduce `TOPIC_PIPELINE_PREPROCESS_WORKERS` first** if RAM pressure is high.
3. **`TOPIC_PIPELINE_GPU_ARTICLE_BATCH` affects both RAM and VRAM**, not just VRAM.
4. **Larger outer `batch_size`** can improve throughput, but increases RAM use.
5. **FAISS** is used automatically when available.
6. **Passage embedding reuse** avoids re-embedding before index construction.

---

## Troubleshooting

**Q: Only 10 FAISS files are created.**  
A: The DB runner defaults to `TOPIC_PIPELINE_LIMIT=10`. Set it explicitly higher.

**Q: Search is slow?**  
A: Ensure FAISS is installed. Without FAISS, search falls back to brute force.

**Q: Out of memory?**  
A: Reduce:
- `TOPIC_PIPELINE_PREPROCESS_WORKERS`
- `TOPIC_PIPELINE_BATCH_SIZE`
- `TOPIC_PIPELINE_GPU_ARTICLE_BATCH`
- `TOPIC_EMBED_BATCH_SIZE`

**Q: CUDA is available but GPU utilization is low?**  
A: That usually means CPU preprocessing / clustering is dominant, while GPU embedding runs in short bursts.

---

## Run the Demo

```bash
python -m embedding_pipeline.topic_vector_demo
```

The demo uses `batch_size=10` and shows topic inference plus vector search on mock articles.