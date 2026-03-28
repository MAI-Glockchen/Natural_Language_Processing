# Topic Inference & Vector Indexing Pipeline

A reusable pipeline for processing Wikipedia articles from the database, inferring topics from citation passages, and building FAISS vector indices for efficient similarity search.

## Features

✅ **Batch Processing**: Process articles in configurable batches  
✅ **Topic Inference**: Automatically infer main topics from cited passages  
✅ **Vector Indexing**: Build FAISS indices for fast similarity search  
✅ **Optimization**: Reuse passage embeddings to avoid redundant computations  
✅ **Persistence**: Save indices and metadata to disk  
✅ **Progress Tracking**: Monitor processing with detailed logging  

---

## Quick Start

### Basic Usage

```python
from pipeline.topic_vector_pipeline import TopicVectorPipeline

# Initialize pipeline
pipeline = TopicVectorPipeline(
    output_dir="vector_indices",
    batch_size=10,
    min_passages=5,
)

# Process all articles from database
results = pipeline.process_all(
    limit=100,  # Process first 100 articles
    verbose=True,
)

# Get summary
summary = pipeline.get_summary()
print(f"Processed: {summary['total_processed']} articles")
print(f"Output: {summary['output_dir']}")
```

### Search in a Vector Index

```python
# Load a saved index
index = pipeline.load_index(article_idx=0)

# Search for similar passages
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

session = get_session()
article = session.query(Article).first()

metadata = pipeline.process_article(
    article,
    infer_topic_kwargs={'top_k': 5}
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
| `batch_size` | int | `10` | Articles per batch |
| `min_passages` | int | `5` | Minimum passages to process article |

### process_all() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | `None` | Max articles to process (None = all) |
| `skip` | int | `0` | Articles to skip |
| `infer_topic_kwargs` | dict | `{}` | Additional kwargs for `infer_topic()` |
| `verbose` | bool | `True` | Print progress |

### infer_topic() Options

Common parameters to pass in `infer_topic_kwargs`:

```python
infer_topic_kwargs={
    'top_k': 5,                          # Number of topic candidates
    'mmr_diversity': 0.4,                # Diversity vs relevance (0-1)
    'concept_cluster_threshold': 0.55,   # Merge similar concepts
    'position_decay': 0.85,              # Earlier passages weighted more
}
```

---

## Output Structure

### Directory Layout

```
vector_indices/
├── 0_article.faiss          # FAISS index for article 0
├── 1_article.faiss          # FAISS index for article 1
├── 2_article.faiss
├── ...
└── summary.json             # Metadata for all articles
```

### summary.json Example

```json
{
  "processed": 3,
  "skipped": 0,
  "failed": 0,
  "timestamp": "2026-03-23T14:30:00",
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
      "index_backend": "faiss"
    }
  ]
}
```

---

## Workflow

```
Database Articles
       ↓
Extract Passages (from Citations)
       ↓
Infer Topics (from Passages)
       ↓
Embed Passages (reuse topic embeddings)
       ↓
Build FAISS Index
       ↓
Save to Disk
```

---

## Example: Full Workflow

```python
from pipeline.topic_vector_pipeline import TopicVectorPipeline

# 1. Initialize
pipeline = TopicVectorPipeline(
    output_dir="my_indices",
    batch_size=5,
)

# 2. Process articles
results = pipeline.process_all(
    limit=50,
    infer_topic_kwargs={'top_k': 3},
    verbose=True,
)

# 3. Get summary
print(pipeline.get_summary())

# 4. Load and search
index = pipeline.load_index(0)
results = index.search("query text", top_k=5)

for r in results:
    print(f"Score: {r['similarity_query_to_passage_score']:.4f}")
    print(r['text'][:100])
```

---

## Performance Tips

1. **Batch Size**: Larger batches (20-50) are faster but use more memory
2. **FAISS**: Automatically used for efficient search when available
3. **Embedding Reuse**: `return_passage_embeddings=True` avoids re-embedding
4. **Database**: Close session when done to free resources

---

## Troubleshooting

**Q: Index file not created?**  
A: Check that at least `min_passages` passages exist for the article.

**Q: Search is slow?**  
A: Make sure FAISS is installed (`pip install faiss-cpu`), otherwise uses brute-force.

**Q: Out of memory?**  
A: Reduce `batch_size` or process articles in smaller chunks with `limit` and `skip`.

---

## Run the Demo

```bash
python topic_vector_demo.py
```

This processes 10 articles and demonstrates topic inference and vector search.
