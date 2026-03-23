# -----------------------------
# Fetches HTML documents from URLs
# -----------------------------

import logging
import time
from typing import Optional
from pathlib import Path
import hashlib
import json
import requests
from config import USER_AGENT, RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_BACKOFF, CACHE_DIR, CACHE_TTL

logger = logging.getLogger(__name__)


class DocumentCache:
    """
    Simple file-based cache for fetched documents.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a URL."""
        cache_key = self._get_cache_key(url)
        return self.cache_dir / f"{cache_key}.json"

    def get(self, url: str) -> Optional[str]:
        """
        Get cached document if available and not expired.

        Args:
            url: URL to fetch

        Returns:
            Cached HTML content or None
        """
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Check TTL
            if time.time() - cache_data.get("timestamp", 0) > CACHE_TTL:
                logger.info(f"Cache expired for: {url}")
                return None

            logger.debug(f"Cache hit for: {url}")
            return cache_data.get("content")

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache for {url}: {e}")
            return None

    def set(self, url: str, content: str) -> None:
        """
        Cache document content.

        Args:
            url: URL of the document
            content: HTML content to cache
        """
        cache_path = self._get_cache_path(url)
        cache_data = {
            "content": content,
            "timestamp": time.time()
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
        except IOError as e:
            logger.error(f"Failed to write cache for {url}: {e}")


cache = DocumentCache()


def fetch_document(url: str) -> Optional[str]:
    """
    Fetch HTML document from URL with caching and retry logic.

    Args:
        url: Citation URL

    Returns:
        str|None: HTML text or None if failed
    """
    # Check cache first
    cached = cache.get(url)
    if cached:
        logger.debug(f"Using cached content for: {url}")
        return cached

    logger.info(f"Fetching: {url}")

    for attempt in range(MAX_RETRIES):
        try:
            # Rate limiting
            if attempt > 0:
                time.sleep(RATE_LIMIT_DELAY * attempt)

            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=30
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                logger.warning(f"Non-HTML content for: {url}")
                return None

            content = response.text

            # Cache the content
            cache.set(url, content)

            return content

        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_BACKOFF * (attempt + 1)} seconds...")
                time.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                logger.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {e}")
                return None

    return None