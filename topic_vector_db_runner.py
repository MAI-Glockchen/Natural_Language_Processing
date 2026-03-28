# -----------------------------
# Real DB runner for topic inference + vector indexing
# Run with: python topic_vector_db_runner.py
# -----------------------------

from pipeline.topic_vector_pipeline import TopicVectorPipeline


def main() -> None:
    pipeline = TopicVectorPipeline(
        output_dir="vector_indices_db_test",
        batch_size=10,
        min_passages=5,
        use_db=True,
        persist_outputs_to_db=True,
    )

    print("Starting topic inference and vector indexing pipeline (DB mode)...\n")

    results = pipeline.process_all(
        limit=20,
        infer_topic_kwargs={
            "top_k": 5,
            "mmr_diversity": 0.4,
        },
        verbose=True,
    )

    summary = pipeline.get_summary()
    print("\n" + "=" * 50)
    print("DB PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Processed: {summary['total_processed']}")
    print(f"Skipped:   {summary['total_skipped']}")
    print(f"Failed:    {summary['total_failed']}")
    print(f"Output:    {summary['output_dir']}")

    if results["articles"]:
        first = results["articles"][0]
        print("\nFirst processed article:")
        print(f"Title: {first['article_title']}")
        print(f"Topic: {first['topic']}")


if __name__ == "__main__":
    main()
