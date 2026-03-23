# -----------------------------
# DB package initializer
# Central imports for convenience
# -----------------------------

from .models import Base, Article, Citation, ArticleCitation, CitationPassage
from .session import (
    get_session,
    get_session_no_commit,
    get_db_session,
    init_db,
    drop_db,
    get_or_create_article,
    get_or_create_citation,
    get_article_by_id,
    get_citation_by_id,
    get_passages_by_citation_id,
    get_passages_by_content_pattern,
    count_articles,
    count_citations,
    count_passages,
    get_article_citations,
    get_citation_articles,
    get_statistics,
)

__all__ = [
    # Models
    "Base",
    "Article",
    "Citation",
    "ArticleCitation",
    "CitationPassage",
    # Session management
    "get_session",
    "get_session_no_commit",
    "get_db_session",
    "init_db",
    "drop_db",
    # Convenience functions
    "get_or_create_article",
    "get_or_create_citation",
    "get_article_by_id",
    "get_citation_by_id",
    "get_passages_by_citation_id",
    "get_passages_by_content_pattern",
    "count_articles",
    "count_citations",
    "count_passages",
    "get_article_citations",
    "get_citation_articles",
    "get_statistics",
]
