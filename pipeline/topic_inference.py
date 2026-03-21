from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize as sk_normalize

from utils.text_normalization import normalize_text
from .vector_index import embed_text


STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "on",
        "in",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "there",
        "than",
        "into",
        "about",
        "over",
        "under",
        "after",
        "before",
        "also",
        "can",
        "could",
        "may",
        "might",
        "will",
        "would",
        "should",
        "do",
        "does",
        "did",
        "done",
        "not",
        "no",
        "yes",
        "such",
        "other",
        "many",
        "most",
        "more",
        "some",
        "any",
        "all",
        "each",
        "both",
        "between",
        "during",
        "while",
        "within",
        "without",
        "said",
        "says",
        "according",
        "including",
        "however",
        "although",
        "because",
        "since",
        "just",
        "like",
        "us",
        "we",
        "he",
        "she",
        "they",
        "his",
        "her",
        "our",
        "your",
        "one",
        "two",
        "three",
        "new",
        "first",
        "last",
        "has",
        "have",
        "had",
        "get",
        "got",
        "use",
        "used",
        "using",
        "very",
        "much",
        "well",
        "still",
        "only",
        "even",
        "now",
    }
)

NER_BLOCKLIST: frozenset[str] = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "In",
        "On",
        "At",
        "By",
        "For",
        "From",
        "And",
        "But",
        "Or",
        "If",
        "Then",
        "Early",
        "However",
        "Although",
        "Because",
        "Since",
        "According",
        "Including",
        "When",
        "Where",
    }
)

# Candidates whose cosine similarity exceeds this threshold are merged into
# the same concept cluster, so "machine learning" and "artificial intelligence"
# contribute to a single topic score rather than competing.
CONCEPT_CLUSTER_THRESHOLD: float = 0.55

# rapidfuzz token_sort_ratio threshold for surface-level deduplication
# (catches plurals, word-order variants, etc.) before embeddings are computed.
SURFACE_DEDUP_THRESHOLD: float = 88.0

_NER_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Z]{2,}(?:\s+[A-Z]{2,})*"
    r"|[A-Z][a-z]+(?:-[A-Z][a-z]+)+"
    r"|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
    r")\b"
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", normalize_text(text))


def _content_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in STOPWORDS]


def _normalize_candidate(text: str) -> str:
    return normalize_text(text).strip()


def _extract_named_entities(passages: list[str]) -> Counter:
    """Lightweight regex NER for acronyms, hyphenated names, and title-case phrases."""
    counts: Counter = Counter()
    for passage in passages:
        for match in _NER_PATTERN.findall(passage):
            candidate = match.strip()
            if candidate in NER_BLOCKLIST:
                continue
            norm = _normalize_candidate(candidate)
            if norm:
                counts[norm] += 1
    return counts


def _surface_deduplicate(phrase_scores: Counter) -> Counter:
    """
    Collapse near-identical surface forms (plurals, word-order variants) into
    their highest-scoring representative before embeddings are computed.
    """
    candidates = sorted(phrase_scores, key=lambda c: phrase_scores[c], reverse=True)
    merged: Counter = Counter()
    absorbed: set[str] = set()

    for anchor in candidates:
        if anchor in absorbed:
            continue
        score = phrase_scores[anchor]
        for other in candidates:
            if other == anchor or other in absorbed:
                continue
            if fuzz.token_sort_ratio(anchor, other) >= SURFACE_DEDUP_THRESHOLD:
                score += phrase_scores[other]
                absorbed.add(other)
        merged[anchor] = score

    return merged


def _build_embedding_cache(
    candidates: list[str],
    passages: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Embed all candidates and passages in one pass and return:
      - cache: {label: unit-vector} for every candidate
      - centroid: mean unit-vector across all passage embeddings
    """
    candidate_vecs = np.array([embed_text(c) for c in candidates], dtype=np.float32)
    candidate_vecs = sk_normalize(candidate_vecs, norm="l2")
    cache = {label: candidate_vecs[i] for i, label in enumerate(candidates)}

    passage_vecs = np.array(
        [embed_text(p) for p in passages if p.strip()], dtype=np.float32
    )
    centroid = (
        sk_normalize(passage_vecs.mean(axis=0, keepdims=True), norm="l2")[0]
        if len(passage_vecs) > 0
        else np.zeros(candidate_vecs.shape[1], dtype=np.float32)
    )

    return cache, centroid


def _cluster_into_concepts(
    phrase_scores: Counter,
    emb_cache: dict[str, np.ndarray],
    threshold: float,
) -> list[tuple[str, float]]:
    """
    Group candidates into concept clusters using agglomerative clustering on
    pre-computed embeddings. Scores within each cluster are summed so that
    synonymous phrases ("machine learning", "artificial intelligence") reinforce
    rather than compete with each other.
    """
    candidates = list(phrase_scores.keys())
    if len(candidates) == 1:
        return [(candidates[0], phrase_scores[candidates[0]])]

    mat = np.stack([emb_cache[c] for c in candidates])  # already normalised

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(mat)

    cluster_members: dict[int, list[str]] = defaultdict(list)
    cluster_scores: dict[int, float] = defaultdict(float)
    for idx, cluster_id in enumerate(labels):
        cand = candidates[idx]
        cluster_members[cluster_id].append(cand)
        cluster_scores[cluster_id] += phrase_scores[cand]

    results = []
    for cluster_id, members in cluster_members.items():
        # Prefer the label with the highest individual score; break ties by length.
        representative = max(members, key=lambda c: (phrase_scores[c], len(c)))
        results.append((representative, cluster_scores[cluster_id]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _mmr_rerank(
    candidates: list[tuple[str, float]],
    emb_cache: dict[str, np.ndarray],
    top_k: int,
    diversity: float,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance re-ranking. Picks candidates that are both
    high-scoring and dissimilar from already-selected ones, so the final
    list covers distinct topics rather than returning near-synonyms.
    """
    if len(candidates) <= top_k:
        return candidates

    labels = [c for c, _ in candidates]
    scores = np.array([s for _, s in candidates], dtype=np.float32)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    mat = np.stack([emb_cache[label] for label in labels])
    sim_matrix = mat @ mat.T

    selected: list[int] = [int(np.argmax(scores))]
    remaining = [i for i in range(len(labels)) if i != selected[0]]

    while len(selected) < top_k and remaining:
        max_sim = np.max(sim_matrix[np.ix_(remaining, selected)], axis=1)
        mmr = (1 - diversity) * scores[remaining] - diversity * max_sim
        pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)

    return [(labels[i], float(scores[i])) for i in selected]


def _deduplicate_substrings(
    candidates: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Remove candidates whose label is a substring of a higher-ranked label."""
    labels = {label for label, _ in candidates}
    return [
        (label, score)
        for label, score in candidates
        if not any(label != other and label in other for other in labels)
    ]


def infer_topic(
    passages: list[str],
    top_k: int = 5,
    unigram_weight: float = 1.0,
    bigram_weight: float = 2.0,
    trigram_weight: float = 2.5,
    entity_weight: float = 3.0,
    semantic_weight: float = 0.75,
    frequency_weight: float = 1.0,
    position_decay: float = 0.85,
    concept_cluster_threshold: float = CONCEPT_CLUSTER_THRESHOLD,
    mmr_diversity: float = 0.4,
) -> dict[str, Any]:
    """
    Infer the main topic(s) from a list of text passages.

    Semantically equivalent phrases (e.g. "machine learning" and "artificial
    intelligence") are merged into a single concept cluster so their frequency
    evidence is combined rather than split across competing candidates.
    """
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()

    for pos, passage in enumerate(passages):
        weight = position_decay**pos
        tokens = _content_tokens(_tokenize(passage))

        for t in tokens:
            unigram_counts[t] += weight
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[(w1, w2)] += weight
        for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
            if w2.lower() not in STOPWORDS:
                trigram_counts[(w1, w2, w3)] += weight

    phrase_scores: Counter = Counter()

    for (w1, w2, w3), score in trigram_counts.items():
        phrase = _normalize_candidate(f"{w1} {w2} {w3}")
        if phrase:
            phrase_scores[phrase] += score * trigram_weight

    for (w1, w2), score in bigram_counts.items():
        phrase = _normalize_candidate(f"{w1} {w2}")
        if phrase:
            phrase_scores[phrase] += score * bigram_weight

    for word, score in unigram_counts.items():
        norm = _normalize_candidate(word)
        if norm:
            phrase_scores[norm] += score * unigram_weight

    for entity, count in _extract_named_entities(passages).items():
        phrase_scores[entity] += count * entity_weight

    if not phrase_scores:
        return {"topic": "unknown topic", "candidates": []}

    phrase_scores = _surface_deduplicate(phrase_scores)

    # Limit candidates before embedding — the most expensive step.
    top_candidates: Counter = Counter(dict(phrase_scores.most_common(120)))
    candidate_labels = list(top_candidates.keys())

    # Single embedding pass for all candidates and passages.
    emb_cache, centroid = _build_embedding_cache(candidate_labels, passages)

    clustered = _cluster_into_concepts(
        top_candidates, emb_cache, threshold=concept_cluster_threshold
    )

    max_raw = max((s for _, s in clustered), default=1.0)
    final_scores: Counter = Counter()
    for label, raw_score in clustered:
        norm_freq = raw_score / max_raw
        sem_score = float(emb_cache[label] @ centroid)  # both already unit-normalised
        final_scores[label] = frequency_weight * norm_freq + semantic_weight * sem_score

    pre_mmr = final_scores.most_common(max(top_k * 4, 20))

    # MMR needs embeddings for the post-cluster representative labels only.
    mmr_cache = {label: emb_cache[label] for label, _ in pre_mmr if label in emb_cache}
    candidates = _mmr_rerank(pre_mmr, mmr_cache, top_k=top_k, diversity=mmr_diversity)

    candidates = _deduplicate_substrings(candidates)[:top_k]
    best_topic = candidates[0][0] if candidates else "unknown topic"

    return {"topic": best_topic, "candidates": candidates}
