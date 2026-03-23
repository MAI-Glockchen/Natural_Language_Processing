# -----------------------------
# Global configuration for the pipeline
# -----------------------------
import os
from pathlib import Path

# PostgreSQL connection URL
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://user:password@localhost:5432/wiki_db")

# User-Agent for web requests to avoid blocking by websites
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
)

# Rate limiting configuration
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))  # seconds between requests

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.0"))  # seconds

# Caching configuration
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds

# Parallel processing configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Passage creation configuration
MAX_SENTENCES_PER_PASSAGE = int(os.getenv("MAX_SENTENCES_PER_PASSAGE", "3"))

# Minimum text length for valid content
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "200"))
