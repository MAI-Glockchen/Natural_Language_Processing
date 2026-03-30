from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize as sk_normalize

from utils.text_normalization import normalize_text
from .vector_index import batch_embed, embed_text


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
        "when",
        "where",
        "which",
        "whose",
        "whom",
        "via",
        "across",
        "through",
        "throughout",
        "often",
        "frequently",
        "typically",
        "mainly",
        "mostly",
        "approximately",
        "various",
        "several",
    }
)

LOW_SIGNAL_BOUNDARY_TOKENS: frozenset[str] = frozenset(
    {
        "when",
        "where",
        "which",
        "while",
        "during",
        "after",
        "before",
        "through",
        "across",
        "within",
        "without",
        "into",
        "from",
        "than",
        "because",
        "since",
    }
)

LOW_SIGNAL_ACTION_TOKENS: frozenset[str] = frozenset(
    {
        "use",
        "used",
        "using",
        "based",
        "combines",
        "combine",
        "includes",
        "include",
        "supports",
        "support",
        "improved",
        "improves",
        "requires",
        "require",
    }
)

GENERIC_SINGLE_TOKENS: frozenset[str] = frozenset(
    {
        "index",
        "model",
        "models",
        "system",
        "systems",
        "data",
        "method",
        "methods",
        "approach",
        "approaches",
    }
)

WEB_NOISE_TOKENS: frozenset[str] = frozenset(
    {
        "http",
        "https",
        "www",
        "com",
        "org",
        "net",
        "protocol",
        "secure",
        "transfer",
        "archive",
        "news",
        "web",
        "online",
    }
)

WEB_SUFFIX_TOKENS: frozenset[str] = frozenset({"com", "org", "net", "www"})

WEB_CONTEXT_TOKENS: frozenset[str] = frozenset(
    {
        "news",
        "protocol",
        "secure",
        "transfer",
        "http",
        "https",
        "archive",
    }
)

HARD_BLOCKLIST_PHRASES: frozenset[str] = frozenset(
    {
        "hypertext transfer protocol",
        "hypertext transfer",
        "transfer protocol secure",
        "protocol secure",
        "hypertext",
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

# Maximum number of candidate phrases passed to the embedding model.
# Raised from 120 to handle articles with many passages without losing signal.
MAX_CANDIDATES: int = 300

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
    normalized = normalize_text(text)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_informative_candidate(candidate: str) -> bool:
    """Heuristic filter to remove low-information phrase fragments."""
    tokens = [t for t in candidate.split() if t]
    if not tokens:
        return False

    if candidate in HARD_BLOCKLIST_PHRASES:
        return False

    if len(tokens) > 5:
        return False

    if tokens[0] in LOW_SIGNAL_BOUNDARY_TOKENS:
        return False
    if tokens[-1] in LOW_SIGNAL_BOUNDARY_TOKENS:
        return False

    if len(tokens) == 1:
        tok = tokens[0]
        if tok in STOPWORDS or tok in GENERIC_SINGLE_TOKENS:
            return False
        if len(tok) < 4:
            return False

    if all(t in WEB_NOISE_TOKENS for t in tokens):
        return False

    web_suffix_hits = sum(1 for t in tokens if t in WEB_SUFFIX_TOKENS)
    web_context_hits = sum(1 for t in tokens if t in WEB_CONTEXT_TOKENS)
    if web_suffix_hits >= 1 and web_context_hits >= 1:
        return False

    # Reject short domain/newswire fragments such as "dawn com news".
    if len(tokens) <= 4 and web_suffix_hits >= 1 and len(tokens) - web_suffix_hits <= 2:
        return False

    if tokens[0] in WEB_SUFFIX_TOKENS or tokens[0] in {"http", "https", "www"}:
        return False

    # Hard reject protocol boilerplate, even when mixed with otherwise valid terms.
    token_set = set(tokens)


    if "protocol" in token_set or "hypertext" in token_set or "transfer" in token_set or "secure" in token_set:
        return False

    action_count = sum(1 for t in tokens if t in LOW_SIGNAL_ACTION_TOKENS)
    if len(tokens) >= 3 and action_count >= 1:
        return False

    # Prefer noun-like keyphrases over clause fragments.
    if len(tokens) >= 3 and any(t in STOPWORDS for t in tokens):
        return False

    return True


def _build_title_hint_vocab(title_hints: list[str] | None) -> set[str]:
    if not title_hints:
        return set()

    vocab: set[str] = set()
    for title in title_hints:
        norm_title = _normalize_candidate(title)
        if not norm_title:
            continue
        vocab.add(norm_title)
        vocab.update(_content_tokens(_tokenize(norm_title)))
    return vocab


def _candidate_noise_penalty(candidate: str, title_vocab: set[str]) -> float:
    """Return a score penalty [0, 0.85] for URL/protocol-style candidates."""
    lower = candidate.lower()
    tokens = [t for t in _tokenize(lower) if t]
    if not tokens:
        return 0.0

    noise_hits = sum(1 for t in tokens if t in WEB_NOISE_TOKENS)
    penalty = 0.0

    if noise_hits > 0:
        penalty += 0.35 * (noise_hits / len(tokens))
    if re.search(r"(?:https?|www|\.com|\.org|\.net)", lower):
        penalty += 0.35
    if noise_hits == len(tokens):
        penalty += 0.25

    if title_vocab and any(t in title_vocab for t in tokens):
        penalty *= 0.35

    return min(0.85, penalty)


def _label_tokens(text: str) -> list[str]:
    """Tokenize a label while splitting URL/slug separators like '-' and '_' ."""
    return [t.lower() for t in re.findall(r"[a-zA-Z]{2,}", normalize_text(text))]


def _canonicalize_topic_label(best_label: str, title_hints: list[str] | None) -> str:
    """Normalize noisy labels into concise topic names anchored by the article title."""
    tokens = _label_tokens(best_label)
    if not tokens:
        return best_label

    first_title = title_hints[0] if title_hints else ""
    title_tokens = _content_tokens(_tokenize(first_title)) if first_title else []

    noisy_shape = (
        any(ch in best_label for ch in ("-", "_", "/", "."))
        or sum(1 for t in tokens if t in WEB_NOISE_TOKENS) >= 2
    )

    if noisy_shape and title_tokens:
        # Anchor to the article title and keep up to two meaningful extras.
        base: list[str] = []
        seen: set[str] = set()

        for t in title_tokens:
            if t in seen:
                continue
            base.append(t)
            seen.add(t)
            if len(base) >= 2:
                break

        extras = [
            t
            for t in tokens
            if t not in seen
            and t not in STOPWORDS
            and t not in LOW_SIGNAL_BOUNDARY_TOKENS
            and t not in WEB_NOISE_TOKENS
            and len(t) >= 4
        ]

        for t in extras:
            if t in seen:
                continue
            base.append(t)
            seen.add(t)
            if len(base) >= 4:
                break

        if len(base) >= 2:
            return " ".join(base)

    # Fallback: clean separators and drop obvious web noise for readability.
    cleaned = [
        t
        for t in tokens
        if t not in WEB_NOISE_TOKENS and t not in STOPWORDS and len(t) >= 3
    ]
    if cleaned:
        return " ".join(cleaned[:5])

    return best_label


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
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Embed all candidates and passages in one pass and return:
      - cache:         {label: unit-vector} for every candidate
      - centroid:      mean unit-vector across all passage embeddings
      - passage_vecs:  (N, D) float32 array of passage embeddings, returned so
                       the caller can pass them to the vector index without
                       re-embedding
    """
    candidate_vecs = batch_embed(candidates)
    candidate_vecs = sk_normalize(candidate_vecs, norm="l2")
    cache = {label: candidate_vecs[i] for i, label in enumerate(candidates)}

    passage_vecs = batch_embed([p for p in passages if p.strip()])
    passage_vecs = sk_normalize(passage_vecs, norm="l2")

    centroid = (
        sk_normalize(passage_vecs.mean(axis=0, keepdims=True), norm="l2")[0]
        if len(passage_vecs) > 0
        else np.zeros(candidate_vecs.shape[1], dtype=np.float32)
    )

    return cache, centroid, passage_vecs


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

    mat = np.stack([emb_cache[c] for c in candidates])

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
    title_hints: list[str] | None = None,
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
    title_hint_weight: float = 2.5,
    return_passage_embeddings: bool = False,
) -> dict[str, Any]:
    """
    Infer the main topic(s) from a list of text passages.

    Semantically equivalent phrases (e.g. "machine learning" and "artificial
    intelligence") are merged into a single concept cluster so their frequency
    evidence is combined rather than split across competing candidates.

    Args:
        passages: Text passages in reading order. Earlier passages are weighted
            more heavily via position_decay.
        title_hints: Optional article/citation titles to anchor candidate topics
            toward canonical noun phrases.
        top_k: Number of topic candidates to return.
        unigram_weight: Score multiplier for single-word candidates.
        bigram_weight: Score multiplier for two-word phrases.
        trigram_weight: Score multiplier for three-word phrases.
        entity_weight: Score bonus for named entities (proper nouns, acronyms).
        semantic_weight: Weight of similarity-to-corpus-centroid in final score.
        frequency_weight: Weight of normalised phrase frequency in final score.
        position_decay: Multiplicative weight decay per passage position.
            0.85 means passage 1 is worth 85% of passage 0, etc.
        concept_cluster_threshold: Cosine similarity cutoff for merging two
            phrases into the same concept. 0.55 merges domain synonyms like
            "ML" and "AI"; raise toward 0.8 for stricter separation.
        mmr_diversity: Balance between relevance and diversity in the returned
            candidate list. 0 = ranked by score only, 1 = maximally diverse.
        title_hint_weight: Score multiplier for candidates originating from
            title_hints (article title and citation titles).
        return_passage_embeddings: If True, include the passage embedding matrix
            in the returned dict under the key "passage_embeddings" (shape N x D,
            float32, L2-normalised). Use this to avoid re-embedding when building
            a PassageVectorIndex immediately after calling infer_topic.

    Returns:
        {
            "topic": str,
            "candidates": list[tuple[str, float]],
            "passage_embeddings": np.ndarray  # only present if return_passage_embeddings=True
        }
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

    if title_hints:
        for title in title_hints:
            norm_title = _normalize_candidate(title)
            if not norm_title:
                continue

            # Full title gets a strong boost to preserve canonical topic names.
            if _is_informative_candidate(norm_title):
                phrase_scores[norm_title] += title_hint_weight * entity_weight

            title_tokens = _content_tokens(_tokenize(norm_title))
            for t in title_tokens:
                norm = _normalize_candidate(t)
                if norm and _is_informative_candidate(norm):
                    phrase_scores[norm] += title_hint_weight * unigram_weight

            for w1, w2 in zip(title_tokens, title_tokens[1:]):
                phrase = _normalize_candidate(f"{w1} {w2}")
                if phrase and _is_informative_candidate(phrase):
                    phrase_scores[phrase] += title_hint_weight * bigram_weight

            for w1, w2, w3 in zip(title_tokens, title_tokens[1:], title_tokens[2:]):
                phrase = _normalize_candidate(f"{w1} {w2} {w3}")
                if phrase and _is_informative_candidate(phrase):
                    phrase_scores[phrase] += title_hint_weight * trigram_weight

    phrase_scores = Counter(
        {
            phrase: score
            for phrase, score in phrase_scores.items()
            if _is_informative_candidate(phrase)
        }
    )

    if not phrase_scores:
        result: dict[str, Any] = {"topic": "unknown topic", "candidates": []}
        if return_passage_embeddings:
            result["passage_embeddings"] = np.empty((0,), dtype=np.float32)
        return result

    phrase_scores = _surface_deduplicate(phrase_scores)

    top_candidates: Counter = Counter(dict(phrase_scores.most_common(MAX_CANDIDATES)))
    candidate_labels = list(top_candidates.keys())

    emb_cache, centroid, passage_vecs = _build_embedding_cache(
        candidate_labels, passages
    )

    clustered = _cluster_into_concepts(
        top_candidates, emb_cache, threshold=concept_cluster_threshold
    )

    title_vocab = _build_title_hint_vocab(title_hints)

    max_raw = max((s for _, s in clustered), default=1.0)
    final_scores: Counter = Counter()
    for label, raw_score in clustered:
        norm_freq = raw_score / max_raw
        sem_score = float(emb_cache[label] @ centroid)
        base_score = frequency_weight * norm_freq + semantic_weight * sem_score
        penalty = _candidate_noise_penalty(label, title_vocab)
        final_scores[label] = base_score * (1.0 - penalty)

    pre_mmr = final_scores.most_common(max(top_k * 4, 20))
    mmr_cache = {label: emb_cache[label] for label, _ in pre_mmr if label in emb_cache}
    candidates = _mmr_rerank(pre_mmr, mmr_cache, top_k=top_k, diversity=mmr_diversity)
    candidates = _deduplicate_substrings(candidates)[:top_k]

    best_topic = (
        _canonicalize_topic_label(candidates[0][0], title_hints)
        if candidates
        else "unknown topic"
    )
    result = {"topic": best_topic, "candidates": candidates}

    if return_passage_embeddings:
        result["passage_embeddings"] = passage_vecs

    return result
