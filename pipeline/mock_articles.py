# -----------------------------
# Hardcoded mock articles for offline script testing
# Matches the structure returned by TopicVectorPipeline._extract_passages_from_article
# -----------------------------

MOCK_ARTICLES = [
    {
        "article_url": "https://en.wikipedia.org/wiki/Natural_language_processing",
        "article_title": "Natural language processing",
        "passage_data": [
            {
                "passage_id": "nlp_1",
                "text": "Natural language processing combines linguistics, statistics, and machine learning to analyze text and speech.",
                "citation_url": "https://example.org/nlp-overview",
                "citation_title": "NLP Overview",
            },
            {
                "passage_id": "nlp_2",
                "text": "Transformer architectures improved language modeling by using self-attention across long contexts.",
                "citation_url": "https://example.org/transformers",
                "citation_title": "Transformer Models",
            },
            {
                "passage_id": "nlp_3",
                "text": "Tokenization, normalization, and sentence segmentation are common preprocessing steps in NLP pipelines.",
                "citation_url": "https://example.org/preprocessing",
                "citation_title": "Text Preprocessing",
            },
            {
                "passage_id": "nlp_4",
                "text": "Information retrieval systems rank passages by semantic similarity between query and document embeddings.",
                "citation_url": "https://example.org/retrieval",
                "citation_title": "Semantic Retrieval",
            },
            {
                "passage_id": "nlp_5",
                "text": "Named entity recognition identifies people, organizations, and locations in unstructured text.",
                "citation_url": "https://example.org/ner",
                "citation_title": "Named Entity Recognition",
            },
            {
                "passage_id": "nlp_6",
                "text": "Evaluation metrics for NLP tasks include precision, recall, F1 score, and task-specific benchmarks.",
                "citation_url": "https://example.org/evaluation",
                "citation_title": "NLP Evaluation",
            },
        ],
    },
    {
        "article_url": "https://en.wikipedia.org/wiki/Vector_database",
        "article_title": "Vector database",
        "passage_data": [
            {
                "passage_id": "vec_1",
                "text": "Vector databases store dense embeddings to enable fast nearest-neighbor search over semantic representations.",
                "citation_url": "https://example.org/vector-db",
                "citation_title": "Vector Databases",
            },
            {
                "passage_id": "vec_2",
                "text": "FAISS supports exact and approximate indexing structures for high-dimensional similarity search.",
                "citation_url": "https://example.org/faiss",
                "citation_title": "FAISS Documentation",
            },
            {
                "passage_id": "vec_3",
                "text": "Cosine similarity is frequently used when embeddings are normalized to unit length.",
                "citation_url": "https://example.org/cosine",
                "citation_title": "Cosine Similarity",
            },
            {
                "passage_id": "vec_4",
                "text": "Index updates require careful handling to keep passage identifiers aligned with stored vectors.",
                "citation_url": "https://example.org/index-maintenance",
                "citation_title": "Index Maintenance",
            },
            {
                "passage_id": "vec_5",
                "text": "Hybrid retrieval combines keyword matching with embedding-based retrieval for better recall.",
                "citation_url": "https://example.org/hybrid-retrieval",
                "citation_title": "Hybrid Retrieval",
            },
            {
                "passage_id": "vec_6",
                "text": "Batch embedding and caching reduce runtime when processing thousands of passages.",
                "citation_url": "https://example.org/batch-embedding",
                "citation_title": "Batch Embedding",
            },
        ],
    },
]
