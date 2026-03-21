# -----------------------------
# Person-2 demo workflow (no DB):
# 1) Topic inference
# 2) Passage embedding + index search
# -----------------------------

from pipeline.topic_inference import infer_topic
from pipeline.vector_index import PassageVectorIndex


def run_demo():
    # Hardcoded passages to simulate output from person 1.
    passages = [
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
    ]

    print("\n[STEP 1] Topic inference from passages")
    result = infer_topic([p["text"] for p in passages], top_k=5)
    print(f"Inferred topic: {result['topic']}")
    print("Top candidates:")
    for candidate, score in result["candidates"]:
        print(f"  - {candidate}: {score}")

    print("\n[STEP 2] Build vector index")
    index = PassageVectorIndex(dim=256)
    index.add_many(passages)
    print(f"Indexed passages: {len(passages)}")

    print("\n[STEP 3] Retrieve relevant passages using inferred topic")
    top_hits = index.search(query=result["topic"], top_k=3)
    for i, hit in enumerate(top_hits, start=1):
        print(f"\nHit {i}")
        print(f"  passage_id: {hit['passage_id']}")
        print(f"  score: {hit['score']:.4f}")
        print(f"  text: {hit['text']}")


if __name__ == "__main__":
    run_demo()
