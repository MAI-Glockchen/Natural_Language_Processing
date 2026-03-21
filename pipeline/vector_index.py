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
_MODEL_CACHE: dict = {}


def _get_model(model_name: str) -> "SentenceTransformer":
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _fallback_embed(text: str, dim: int) -> list[float]:
    """
    Hashed bag-of-words embedding used when sentence-transformers is unavailable.
    Produces a normalised vector of length `dim`.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", normalize_text(text))
    vec = [0.0] * dim
    for token in tokens:
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def embed_text(
    text: str,
    dim: int = 256,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_fallback: bool = True,
) -> list[float]:
    """
    Embed a single string using all-MiniLM-L6-v2, falling back to a hashed
    bag-of-words vector if sentence-transformers is not installed.
    """
    normalized = normalize_text(text)

    if SentenceTransformer is not None:
        vec = _get_model(model_name).encode(normalized, normalize_embeddings=True)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    if not use_fallback:
        raise ImportError("sentence-transformers is required (use_fallback=False).")

    return _fallback_embed(normalized, dim)


def _batch_embed(
    texts: list[str],
    dim: int,
    model_name: str,
    use_fallback: bool,
) -> "np.ndarray":
    """
    Embed a list of strings and return an (N, D) float32 numpy array.
    Uses SentenceTransformer batch encoding when available, which is
    significantly faster than calling embed_text in a loop.
    """
    normalised = [normalize_text(t) for t in texts]

    if SentenceTransformer is not None:
        vecs = _get_model(model_name).encode(
            normalised,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype="float32")

    if not use_fallback:
        raise ImportError("sentence-transformers is required (use_fallback=False).")

    return np.array([_fallback_embed(t, dim) for t in normalised], dtype="float32")


def cosine_similarity(v1, v2) -> float:
    """Cosine similarity between two vectors. Uses numpy when available."""
    if np is not None:
        return float(np.dot(v1, v2))
    return sum(a * b for a, b in zip(v1, v2))


class PassageVectorIndex:
    def __init__(
        self,
        dim: int = 256,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        use_fallback: bool = True,
    ):
        self.dim = dim
        self.model_name = model_name
        self.use_fallback = use_fallback
        self._rows: list[dict] = []
        self._faiss_index = None

    def _can_use_faiss(self) -> bool:
        return faiss is not None and np is not None

    def _init_faiss(self, vector_dim: int) -> None:
        if self._faiss_index is None and self._can_use_faiss():
            # IndexFlatIP does exact inner-product search. With L2-normalised
            # vectors this is equivalent to cosine similarity.
            self._faiss_index = faiss.IndexFlatIP(vector_dim)

    def get_index_backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "brute_force"

    def add_passage(self, passage_id, text: str) -> None:
        vector = embed_text(text, self.dim, self.model_name, self.use_fallback)
        self._rows.append({"passage_id": passage_id, "text": text, "vector": vector})
        self._init_faiss(len(vector))
        if self._faiss_index is not None:
            self._faiss_index.add(np.array([vector], dtype="float32"))

    def add_many(self, passages: list[dict]) -> None:
        """
        Batch-encode and index a list of passages.

        Args:
            passages: [{"passage_id": str | int, "text": str}, ...]
        """
        if not passages:
            return

        texts = [p["text"] for p in passages]
        vectors = _batch_embed(texts, self.dim, self.model_name, self.use_fallback)

        self._init_faiss(vectors.shape[1])

        for p, vec in zip(passages, vectors):
            self._rows.append(
                {
                    "passage_id": p["passage_id"],
                    "text": p["text"],
                    "vector": vec.tolist(),
                }
            )

        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._rows:
            return []

        k = max(top_k, 1)
        q_vec = embed_text(query, self.dim, self.model_name, self.use_fallback)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(
                np.array([q_vec], dtype="float32"), k
            )
            return [
                {
                    "passage_id": self._rows[int(idx)]["passage_id"],
                    "text": self._rows[int(idx)]["text"],
                    "similarity_query_to_passage_score": float(score),
                }
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]

        scored = sorted(
            self._rows,
            key=lambda row: cosine_similarity(q_vec, row["vector"]),
            reverse=True,
        )
        return [
            {
                "passage_id": row["passage_id"],
                "text": row["text"],
                "similarity_query_to_passage_score": cosine_similarity(
                    q_vec, row["vector"]
                ),
            }
            for row in scored[:k]
        ]
