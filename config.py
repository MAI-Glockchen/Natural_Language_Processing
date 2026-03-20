# -----------------------------
# Global configuration for the pipeline
# -----------------------------

# PostgreSQL connection URL
DB_URL = "postgresql+psycopg2://wiki_user:wiki_pass@localhost:5432/wiki_db"

# User-Agent for web requests to avoid blocking by websites
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/114.0.0.0 Safari/537.36")