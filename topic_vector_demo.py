# -----------------------------
# Example: Using the Topic-Vector Pipeline
# Run with: python topic_vector_demo.py
# -----------------------------

from pipeline.topic_vector_pipeline import TopicVectorPipeline
from pipeline.mock_articles import MOCK_ARTICLES


def main():
    """
    Example: Process articles, infer topics, and build vector indices.
    """

    # Initialize pipeline in mock/offline mode (no DB required)
    pipeline = TopicVectorPipeline(
        output_dir="vector_indices",
        batch_size=5,
        min_passages=5,
        use_db=False,
    )

    # Process hardcoded mock articles with the same payload shape as DB-derived data
    print("Starting topic inference and vector indexing pipeline (mock mode)...\n")
    results = pipeline.process_mock_articles(
        mock_articles=MOCK_ARTICLES,
        infer_topic_kwargs={
            "top_k": 5,
            "mmr_diversity": 0.4,
        },
        verbose=True,
    )

    # Print summary
    summary = pipeline.get_summary()
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"✓ Processed: {summary['total_processed']}")
    print(f"⊘ Skipped:   {summary['total_skipped']}")
    print(f"✗ Failed:    {summary['total_failed']}")
    print(f"📁 Output:   {summary['output_dir']}")

    # Option 2: Search in a processed article's index
    if len(results["articles"]) > 0:
        print("\n" + "=" * 50)
        print("EXAMPLE: Search in first processed article")
        print("=" * 50)

        article_metadata = results["articles"][0]
        print(f"Article: {article_metadata['article_title']}")
        print(f"Inferred Topic: {article_metadata['topic']}")
        print(f"Passages: {article_metadata['num_passages']}")

        # Load the FAISS index
        index = pipeline.load_index(0)
        if index:
            # Search for related passages
            query = "important research findings"
            search_results = index.search(query, top_k=3)

            print(f"\nSearch results for: '{query}'")
            for i, result in enumerate(search_results, 1):
                score = result["similarity_query_to_passage_score"]
                text_preview = result["text"][:100].replace("\n", " ")
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   {text_preview}...")


if __name__ == "__main__":
    main()
