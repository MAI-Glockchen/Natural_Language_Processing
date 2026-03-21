# -----------------------------
# Infers a coarse article topic from a list of passages
# -----------------------------

import re
from collections import Counter

from utils.text_normalization import normalize_text
from .vector_index import cosine_similarity, embed_text


STOPWORDS = {
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
}


def _tokenize(text):
    normalized = normalize_text(text)
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", normalized)


def _normalize_candidate(candidate):
    normalized = normalize_text(candidate)
    if normalized == "ai":
        return "artificial intelligence"
    if normalized == "ml":
        return "machine learning"
    return normalized


def _extract_named_entities(passages):
    """
    Lightweight NER heuristic:
    - Capitalized phrases (e.g., New York Times)
    - Acronyms (e.g., AI, NASA, NLP)
    """
    entity_counts = Counter()
    entity_pattern = re.compile(
        r"\b(?:[A-Z]{2,}(?:\s+[A-Z]{2,})*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
    )

    blocklist = {
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
    }

    for passage in passages:
        for match in entity_pattern.findall(passage):
            candidate = match.strip()
            if candidate in blocklist:
                continue
            normalized = _normalize_candidate(candidate)
            if normalized:
                entity_counts[normalized] += 1

    return entity_counts


def _mean_vector(vectors):
    if not vectors:
        return []

    dim = len(vectors[0])
    mean = [0.0] * dim

    for vec in vectors:
        for i, value in enumerate(vec):
            mean[i] += value

    count = float(len(vectors))
    return [value / count for value in mean]


def infer_topic(
    passages,
    top_k=5,
    unigram_weight=1.0,
    bigram_weight=2.0,
    entity_weight=2.5,
    semantic_weight=0.75,
    frequency_weight=1.0,
):
    """
    Step 1: Clean words for counting
    For each passage, it:

        lowercases text
        keeps word-like tokens
        removes common stopwords (like “the”, “and”, “is”).
        Result: only meaningful content words remain for counting.

    Step 2: Count unigrams and bigrams
    From cleaned tokens, it builds:

        unigram counts: single words
        bigram counts: 2-word sequences in order
        Example:

        If “artificial intelligence” appears 3 times,
        artificial and intelligence increase as unigrams
        artificial intelligence increases as a bigram.

    Step 3: Build initial candidate scores
    It creates one shared candidate score table:

        adds unigram score × unigram_weight
        adds bigram score × bigram_weight
        So if bigram_weight=2.0, a repeated phrase like “artificial intelligence” is intentionally favored over single words.

    Step 4: Add named entities
    It extracts simple named entities (capitalized phrases/acronyms), normalizes some terms (AI -> artificial intelligence, ML -> machine learning), then adds:

        entity count × entity_weight
        This means entities can push a candidate higher if they appear often.

    Step 5: Early fallback
    If no candidates exist at all, it returns:

    topic: "unknown topic"
    candidates: []
    """
    unigram_counts = Counter()
    bigram_counts = Counter()

    for passage in passages:
        tokens = [t for t in _tokenize(passage) if t not in STOPWORDS]
        unigram_counts.update(tokens)
        bigram_counts.update(zip(tokens, tokens[1:]))

    phrase_scores = Counter()

    for (w1, w2), score in bigram_counts.items():
        phrase = _normalize_candidate(f"{w1} {w2}")
        phrase_scores[phrase] += score * bigram_weight

    for word, score in unigram_counts.items():
        phrase_scores[_normalize_candidate(word)] += score * unigram_weight

    entity_counts = _extract_named_entities(passages)
    for entity, score in entity_counts.items():
        phrase_scores[entity] += score * entity_weight

    if not phrase_scores:
        return {
            "topic": "unknown topic",
            "candidates": [],
        }

    # Build a lightweight semantic centrality score without external models:
    # candidate embedding similarity to the mean embedding of all passages.
    passage_vectors = [embed_text(passage) for passage in passages if passage.strip()]
    corpus_centroid = _mean_vector(passage_vectors)

    max_freq_score = max(phrase_scores.values())
    combined_scores = Counter()

    for candidate, freq_score in phrase_scores.items():
        normalized_freq = (freq_score / max_freq_score) if max_freq_score > 0 else 0.0

        if corpus_centroid:
            semantic_score = cosine_similarity(embed_text(candidate), corpus_centroid)
        else:
            semantic_score = 0.0

        combined_scores[candidate] = (
            frequency_weight * normalized_freq + semantic_weight * semantic_score
        )

    candidates = combined_scores.most_common(max(top_k, 1))
    best_topic = candidates[0][0] if candidates else "unknown topic"

    return {
        "topic": best_topic,
        "candidates": candidates,
    }
