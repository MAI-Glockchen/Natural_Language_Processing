# -----------------------------
# Splits text into short, semantically coherent passages
# -----------------------------

import logging
from typing import List
from nltk.tokenize import sent_tokenize
from config import MAX_SENTENCES_PER_PASSAGE, MIN_TEXT_LENGTH

logger = logging.getLogger(__name__)


def create_passages(text: str, max_sentences: int = MAX_SENTENCES_PER_PASSAGE) -> List[str]:
    """
    Split text into short, semantically coherent passages.

    Splits text into sentences and groups them into passages of up to
    max_sentences sentences each.

    Args:
        text (str): Cleaned text
        max_sentences (int): Max sentences per passage

    Returns:
        List[str]: List of text passages
    """
    if not text or not text.strip():
        logger.warning("Empty text input for passage creation")
        return []

    if len(text) < MIN_TEXT_LENGTH:
        logger.warning(f"Text too short ({len(text)} chars), skipping passage creation")
        return []

    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Failed to tokenize sentences: {e}")
        return []

    if not sentences:
        logger.warning("No sentences found in text")
        return []

    passages: List[str] = []
    current: List[str] = []

    for sent in sentences:
        current.append(sent)

        if len(current) >= max_sentences:
            passage = " ".join(current)
            passages.append(passage)
            current = []

    if current:
        passage = " ".join(current)
        if len(passage) >= MIN_TEXT_LENGTH:
            passages.append(passage)

    logger.info(f"Created {len(passages)} passages from text")
    return passages