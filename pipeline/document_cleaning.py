# -----------------------------
# Cleans HTML and extracts readable text
# -----------------------------

import logging
from typing import Optional
from readability import Document
from bs4 import BeautifulSoup
from config import MIN_TEXT_LENGTH

logger = logging.getLogger(__name__)


def clean_html(html: str) -> str:
    """
    Clean HTML and extract readable text.

    Uses readability library to extract main content, then cleans up whitespace.

    Args:
        html (str): Raw HTML

    Returns:
        str: Clean text
    """
    if not html or not html.strip():
        logger.warning("Empty HTML input")
        return ""

    try:
        doc = Document(html)
        main_content = doc.summary()

        if not main_content or not main_content.strip():
            logger.warning("No main content extracted from HTML")
            return ""

        soup = BeautifulSoup(main_content, "html.parser")
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())  # Normalize whitespace

        # Additional cleanup
        text = _clean_special_characters(text)

        if len(text) < MIN_TEXT_LENGTH:
            logger.warning(f"Text too short ({len(text)} chars), skipping")
            return ""

        logger.debug(f"Extracted {len(text)} characters from HTML")
        return text

    except Exception as e:
        logger.error(f"Failed to clean HTML: {e}")
        return ""


def _clean_special_characters(text: str) -> str:
    """
    Remove or replace special characters that might interfere with processing.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove HTML entities
    import html
    text = html.unescape(text)

    # Remove common artifacts
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\u200c", "")  # Zero-width non-joiner
    text = text.replace("\u200d", "")  # Zero-width joiner

    return text
