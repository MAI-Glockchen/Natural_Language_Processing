from pipeline.topic_inference import infer_topic
from pipeline.vector_index import PassageVectorIndex


def process_articles(articles: list[dict]) -> list[dict]:
    """
    Infer topics and build a searchable vector index for a list of articles.

    Each article is processed independently so topics are specific to that
    article rather than averaged across the whole corpus. Passage embeddings
    computed during topic inference are reused directly when building the
    index, so each passage is only sent through the embedding model once.

    Args:
        articles: [
            {
                "article_id": str | int,
                "passages": [{"passage_id": str | int, "text": str}, ...]
            },
            ...
        ]

    Returns:
        List of result dicts, one per article:
        [
            {
                "article_id": ...,
                "topic": str,
                "candidates": list[tuple[str, float]],
                "index": PassageVectorIndex
            },
            ...
        ]
    """
    results = []

    for article in articles:
        passages = article["passages"]
        texts = [p["text"] for p in passages]

        topic_result = infer_topic(texts, top_k=5, return_passage_embeddings=True)

        index = PassageVectorIndex()
        index.add_many_precomputed(passages, topic_result["passage_embeddings"])

        results.append(
            {
                "article_id": article["article_id"],
                "topic": topic_result["topic"],
                "candidates": topic_result["candidates"],
                "index": index,
            }
        )

    return results


def run_demo():
    articles = [
        {
            "article_id": "a1",
            "passages": [
                {
                    "passage_id": "p1",
                    "text": "Artificial intelligence is a field of computer science focused on creating systems that perform tasks requiring human intelligence.",
                },
                {
                    "passage_id": "p2",
                    "text": "Machine learning is a core part of modern AI and allows models to improve by learning from data.",
                },
                {
                    "passage_id": "p3",
                    "text": "Neural networks are widely used in language processing, image recognition, and speech technologies.",
                },
                {
                    "passage_id": "p4",
                    "text": "Ethical concerns in AI include fairness, transparency, bias mitigation, and accountability.",
                },
                {
                    "passage_id": "p5",
                    "text": "Early AI research started in the mid-20th century and evolved through symbolic and statistical approaches.",
                },
            ],
        },
        {
            "article_id": "a2",
            "passages": [
                {
                    "passage_id": "p6",
                    "text": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
                },
                {
                    "passage_id": "p7",
                    "text": "Greenhouse gas emissions from fossil fuels are the primary driver of modern climate change.",
                },
                {
                    "passage_id": "p8",
                    "text": "Renewable energy sources such as solar and wind are central to decarbonisation strategies.",
                },
            ],
        },
    ]

    results = process_articles(articles)

    for r in results:
        print(f"\nArticle: {r['article_id']}")
        print(f"  Topic:  {r['topic']}")
        print(f"  Candidates:")
        for label, score in r["candidates"]:
            print(f"    - {label}: {score:.4f}")

        hits = r["index"].search(query=r["topic"], top_k=2)
        print(f"  Top passages for '{r['topic']}':")
        for hit in hits:
            print(
                f"    [{hit['passage_id']}] {hit['similarity_query_to_passage_score']:.4f}  {hit['text'][:80]}"
            )


if __name__ == "__main__":
    run_demo()
