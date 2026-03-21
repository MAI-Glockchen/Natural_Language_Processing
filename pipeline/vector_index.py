# -----------------------------
# Lightweight embedding + cosine search index for passages
# -----------------------------

import hashlib
import math
import re

try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from utils.text_normalization import normalize_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE = {}


def _tokenize(text):
    normalized = normalize_text(text)
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", normalized)


def _get_model(model_name):
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_text(
    text,
    dim=256,
    model_name=DEFAULT_EMBEDDING_MODEL,
    use_fallback=True,
):
    """
    Convert text to an embedding vector.

    Primary mode:
        Uses sentence-transformers/all-MiniLM-L6-v2.

    Fallback mode:
        Uses hashed bag-of-words when sentence-transformers is unavailable.
    """
    normalized = normalize_text(text)

    if SentenceTransformer is not None:
        model = _get_model(model_name)
        vec = model.encode(normalized, normalize_embeddings=True)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    if not use_fallback:
        raise ImportError(
            "sentence-transformers is required for embedding generation when fallback is disabled."
        )

    # Lightweight fallback for local development and demos.
    vec = [0.0] * dim
    for token in _tokenize(normalized):
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
    def __init__(
        self,
        dim=256,
        model_name=DEFAULT_EMBEDDING_MODEL,
        use_fallback=True,
    ):
        self.dim = dim
        self.model_name = model_name
        self.use_fallback = use_fallback
        self._rows = []
        self._faiss_index = None

    def _can_use_faiss(self):
        return faiss is not None and np is not None

    def _ensure_faiss_index(self, vector_len):
        if self._faiss_index is None and self._can_use_faiss():
            # IndexFlatIP performs inner-product search. With normalized vectors,
            # this is equivalent to cosine similarity.
            self._faiss_index = faiss.IndexFlatIP(vector_len)

    def _vector_to_faiss_row(self, vector):
        return np.asarray([vector], dtype="float32")

    def get_index_backend(self):
        return "faiss" if self._faiss_index is not None else "brute_force"

    def add_passage(self, passage_id, text):
        vector = embed_text(
            text,
            dim=self.dim,
            model_name=self.model_name,
            use_fallback=self.use_fallback,
        )

        row = {
            "passage_id": passage_id,
            "text": text,
            "vector": vector,
        }
        self._rows.append(row)

        self._ensure_faiss_index(len(vector))
        if self._faiss_index is not None:
            self._faiss_index.add(self._vector_to_faiss_row(vector))

    def add_many(self, passages):
        """
        Args:
            passages (list[dict]): [{"passage_id": str|int, "text": str}, ...]
        """
        for p in passages:
            self.add_passage(p["passage_id"], p["text"])

    def search(self, query, top_k=5):
        if not self._rows:
            return []

        k = max(top_k, 1)
        q_vec = embed_text(
            query,
            dim=self.dim,
            model_name=self.model_name,
            use_fallback=self.use_fallback,
        )

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(
                self._vector_to_faiss_row(q_vec), k
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                row = self._rows[int(idx)]
                results.append(
                    {
                        "passage_id": row["passage_id"],
                        "text": row["text"],
                        "similarity_query_to_passage_score": float(score),
                    }
                )
            return results

        scored = []

        for row in self._rows:
            score = cosine_similarity(q_vec, row["vector"])
            scored.append(
                {
                    "passage_id": row["passage_id"],
                    "text": row["text"],
                    "similarity_query_to_passage_score": score,
                }
            )

        scored.sort(key=lambda x: x["similarity_query_to_passage_score"], reverse=True)
        return scored[:k]
