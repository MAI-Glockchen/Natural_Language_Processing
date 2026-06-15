from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore[import-not-found]
except Exception:
    faiss = None

from embedding_pipeline.mock_articles import MOCK_ARTICLES
from embedding_pipeline.topic_inference import infer_topic
from embedding_pipeline.vector_index import PassageVectorIndex


def _topic_tokens(topic: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", topic.lower())
        if len(token) >= 4
    }


def _has_topic_overlap(topic: str, text: str) -> bool:
    tokens = _topic_tokens(topic)
    if not tokens:
        return False
    text_l = text.lower()
    return any(token in text_l for token in tokens)


def _validate_embeddings(embeddings: np.ndarray, expected_rows: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(embeddings, np.ndarray):
        errors.append("passage_embeddings is not a numpy array")
        return errors

    if embeddings.ndim != 2:
        errors.append(f"passage_embeddings.ndim={embeddings.ndim}, expected 2")
        return errors

    rows, cols = int(embeddings.shape[0]), int(embeddings.shape[1])
    if rows != expected_rows:
        errors.append(f"embedding rows={rows}, expected={expected_rows}")
    if cols <= 0:
        errors.append("embedding dimension must be > 0")

    if not np.all(np.isfinite(embeddings)):
        errors.append("embeddings contain non-finite values")

    return errors


def _validate_faiss_roundtrip(index: PassageVectorIndex, index_path: Path) -> list[str]:
    errors: list[str] = []

    if faiss is None:
        return [
            "faiss is not available; install faiss-cpu to verify FAISS structure"
        ]

    if index._faiss_index is None:
        return [
            "index backend is not FAISS; cannot validate FAISS structure"
        ]

    faiss.write_index(index._faiss_index, str(index_path))
    loaded = faiss.read_index(str(index_path))

    expected_total = len(index._rows)
    if int(loaded.ntotal) != expected_total:
        errors.append(
            f"loaded.ntotal={int(loaded.ntotal)} does not match rows={expected_total}"
        )

    expected_dim = int(index._faiss_index.d)
    if int(loaded.d) != expected_dim:
        errors.append(f"loaded.d={int(loaded.d)} does not match expected d={expected_dim}")

    return errors


def _validate_topic_retrieval(
    index: PassageVectorIndex,
    topic: str,
    candidates: list[tuple[str, float]],
    top_k: int,
) -> list[str]:
    errors: list[str] = []

    queries: list[str] = [topic]
    queries.extend(label for label, _score in candidates)

    retrieval_ok = False
    for query in queries:
        hits = index.search(query, top_k=top_k)
        if not hits:
            continue

        best_score = float(hits[0]["similarity_query_to_passage_score"])
        if not math.isfinite(best_score):
            continue

        if best_score >= 0.2:
            retrieval_ok = True
            break

        if any(_has_topic_overlap(query, hit["text"]) for hit in hits):
            retrieval_ok = True
            break

    if not retrieval_ok:
        errors.append(
            "topic retrieval did not return a confident or lexically related top result"
        )

    return errors


def run_selftest(output_dir: Path, top_k: int = 3, verbose: bool = True) -> int:
    failures: list[str] = []

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for article_idx, article in enumerate(MOCK_ARTICLES):
        title = article.get("article_title", f"article_{article_idx}")
        passages = [p["text"] for p in article.get("passage_data", []) if p.get("text")]
        passage_items = [
            {"passage_id": p["passage_id"], "text": p["text"]}
            for p in article.get("passage_data", [])
            if p.get("text")
        ]

        if verbose:
            print(f"\n[ARTICLE {article_idx}] {title}")

        topic_result = infer_topic(
            passages=passages,
            title_hints=[title],
            top_k=5,
            mmr_diversity=0.4,
            return_passage_embeddings=True,
        )

        topic = str(topic_result.get("topic", "unknown topic"))
        candidates = list(topic_result.get("candidates", []))
        embeddings = topic_result.get("passage_embeddings")

        article_errors: list[str] = []
        article_errors.extend(_validate_embeddings(embeddings, expected_rows=len(passages)))

        if not article_errors:
            vector_dim = int(embeddings.shape[1])
            index = PassageVectorIndex(dim=vector_dim)
            index.add_many_precomputed(passages=passage_items, vectors=embeddings)

            index_path = output_dir / f"selftest_{article_idx}.faiss"
            article_errors.extend(_validate_faiss_roundtrip(index, index_path))
            article_errors.extend(
                _validate_topic_retrieval(
                    index=index,
                    topic=topic,
                    candidates=candidates,
                    top_k=min(top_k, len(passage_items)),
                )
            )

        if article_errors:
            failures.extend([f"[{title}] {err}" for err in article_errors])
            if verbose:
                print("  RESULT: FAIL")
                for err in article_errors:
                    print(f"  - {err}")
        else:
            if verbose:
                print("  RESULT: PASS")
                print(f"  topic: {topic}")
                if candidates:
                    print(f"  top_candidate: {candidates[0][0]} ({float(candidates[0][1]):.4f})")

    print("\n" + "=" * 60)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} issues)")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print("SELFTEST PASSED")
    print("- Topic inference produced valid embeddings")
    print("- FAISS index roundtrip validated (d and ntotal)")
    print("- Topic-based retrieval returned relevant matches")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate topic inference + embeddings + FAISS index structure + topic retrieval"
        )
    )
    parser.add_argument(
        "--output-dir",
        default="vector_indices_selftest",
        help="Directory for temporary selftest FAISS files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval size for validation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-article logging",
    )
    args = parser.parse_args()

    exit_code = run_selftest(
        output_dir=Path(args.output_dir),
        top_k=max(1, args.top_k),
        verbose=not args.quiet,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
