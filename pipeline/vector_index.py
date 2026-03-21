# -----------------------------
# Lightweight embedding + cosine search index for passages
# -----------------------------

import hashlib
import math
import re

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

    def add_passage(self, passage_id, text):
        row = {
            "passage_id": passage_id,
            "text": text,
            "vector": embed_text(
                text,
                dim=self.dim,
                model_name=self.model_name,
                use_fallback=self.use_fallback,
            ),
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
        q_vec = embed_text(
            query,
            dim=self.dim,
            model_name=self.model_name,
            use_fallback=self.use_fallback,
        )
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
        return scored[: max(top_k, 1)]
