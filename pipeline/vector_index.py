# -----------------------------
# Lightweight embedding + cosine search index for passages
# -----------------------------

import hashlib
import math
import re


def _normalize_text(text):
    # Normalize common short forms so semantic matches are less brittle.
    text = text.lower()
    text = re.sub(r"\bai\b", "artificial intelligence", text)
    text = re.sub(r"\bml\b", "machine learning", text)
    return text


def _tokenize(text):
    normalized = _normalize_text(text)
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", normalized)


def embed_text(text, dim=256):
    """
    Convert text to a fixed-size vector via hashed bag-of-words.
    This is a lightweight fallback for local development and demos.
    """
    vec = [0.0] * dim
    for token in _tokenize(text):
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h, 16) % dim
        vec[idx] += 1.0

    # L2 normalize to make cosine similarity stable.
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


class PassageVectorIndex:
    def __init__(self, dim=256):
        self.dim = dim
        self._rows = []

    def add_passage(self, passage_id, text):
        row = {
            "passage_id": passage_id,
            "text": text,
            "vector": embed_text(text, dim=self.dim),
        }
        self._rows.append(row)

    def add_many(self, passages):
        """
        Args:
            passages (list[dict]): [{"passage_id": str|int, "text": str}, ...]
        """
        for p in passages:
            self.add_passage(p["passage_id"], p["text"])

    def search(self, query, top_k=5):
        q_vec = embed_text(query, dim=self.dim)
        scored = []

        for row in self._rows:
            score = cosine_similarity(q_vec, row["vector"])
            scored.append(
                {
                    "passage_id": row["passage_id"],
                    "text": row["text"],
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(top_k, 1)]
