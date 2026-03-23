# -----------------------------
# Extracts all citations from a Wikipedia article
# -----------------------------

import logging
import re
from typing import List, Set
from bs4 import BeautifulSoup
import requests
from config import USER_AGENT, MAX_RETRIES, RETRY_BACKOFF

logger = logging.getLogger(__name__)


def extract_citations(wikipedia_url: str) -> List[str]:
    """
    Extract all citation URLs from a Wikipedia article robustly.

    Handles:
    - <li id="cite_note-..."> items
    - <cite> tags and nested <a> links inside spans
    - Web archives
    - Only returns HTTP/HTTPS URLs

    Args:
        wikipedia_url: URL of the Wikipedia article

    Returns:
        List[str]: List of unique citation URLs
    """
    logger.info(f"Extracting citations from: {wikipedia_url}")

    headers = {"User-Agent": USER_AGENT}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(wikipedia_url, headers=headers, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_BACKOFF * (attempt + 1)} seconds...")
                import time
                time.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                logger.error(f"Failed to fetch Wikipedia page after {MAX_RETRIES} attempts: {e}")
                return []

    soup = BeautifulSoup(response.text, "html.parser")
    citations: Set[str] = set()

    # All <li> with id starting 'cite_note'
    for li in soup.find_all("li", id=lambda x: x and x.startswith("cite_note")):
        # Use <cite> if available, else the <li> itself
        cite = li.find("cite") or li

        # Find all <a href> recursively inside <cite>
        for link in cite.find_all("a", href=True):
            url = link.get("href")
            if not url:
                continue

            # Handle relative URLs
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = "https://en.wikipedia.org" + url

            # Only keep HTTP/HTTPS
            if url.startswith("http"):
                citations.add(url)

    logger.info(f"Found {len(citations)} reference structures in the HTML code")
    return list(citations)


def validate_citation_url(url: str) -> bool:
    """
    Validate if a URL is a valid citation URL.

    Args:
        url: URL to validate

    Returns:
        bool: True if valid, False otherwise
    """
    # Filter out Wikipedia internal links
    if "en.wikipedia.org" in url and not url.endswith("/wiki/"):
        return True

    # Filter out obvious non-content URLs
    blocked_patterns = [
        r"\.wikipedia\.org/wiki/Special:",
        r"\.wikipedia\.org/wiki/Help:",
        r"\.wikipedia\.org/wiki/Talk:",
        r"\.wikipedia\.org/wiki/User:",
        r"\.wikipedia\.org/wiki/Wikipedia:",
        r"\.wikipedia\.org/wiki/Portal:",
        r"\.wikipedia\.org/wiki/File:",
        r"\.wikipedia\.org/wiki/Category:",
        r"\.wikipedia\.org/wiki/Template:",
        r"\.wikipedia\.org/wiki/Category_tree:",
        r"\.wikipedia\.org/wiki/Template_talk:",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, url):
            return False

    return True
