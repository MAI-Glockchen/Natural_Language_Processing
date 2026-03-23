# -----------------------------
# PostgreSQL connection & session setup
# -----------------------------

from __future__ import annotations
from contextlib import contextmanager
from typing import Generator, Optional
import logging
from sqlalchemy import create_engine, event, text, func
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool
from config import DB_URL
from db.models import Base, Article, Citation, CitationPassage, ArticleCitation

logger = logging.getLogger(__name__)


# Create engine with optimized connection pooling
engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_pre_ping=True,  # Automatically retry failed connections
    pool_size=20,  # Increased pool size for better concurrency
    max_overflow=40,  # Max additional connections
    pool_timeout=30,  # Timeout for getting a connection
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,  # Set to True for SQL debugging
    connect_args={"connect_timeout": 10}  # Connection timeout
)


# Create tables if they don't exist (run once at startup)
def init_db() -> None:
    """
    Initialize database tables if they don't exist.
    
    This should be called once at application startup.
    """
    logger.info("Initializing database tables...")
    with engine.connect() as conn:
        Base.metadata.create_all(bind=conn)
        logger.info("Database tables initialized successfully")


# Drop all tables (for development/testing)
def drop_db() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data! Use only for development/testing.
    """
    logger.warning("Dropping all database tables...")
    with engine.connect() as conn:
        Base.metadata.drop_all(bind=conn)
        logger.info("Database tables dropped successfully")


# Create session factory with scoped sessions for thread safety
SessionLocal = scoped_session(
    sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Get a database session with automatic cleanup.
    
    This is a context manager that ensures proper session lifecycle.
    
    Yields:
        Session: Database session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_session_no_commit() -> Generator[Session, None, None]:
    """
    Get a database session without automatic commit.
    
    Use this when you want manual control over commits.
    
    Yields:
        Session: Database session
    """
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def close_session(session: Session) -> None:
    """
    Close a database session.

    Args:
        session: Session to close
    """
    session.close()


# Event listener for cleanup on session close
@event.listens_for(SessionLocal, "close")
def remove_session(session):
    session.remove()


# Utility functions for common database operations
def get_or_create_article(url: str, title: Optional[str] = None) -> Article:
    """
    Get an article by URL or create it if it doesn't exist.
    
    Args:
        url: Article URL
        title: Article title (optional)
    
    Returns:
        Article: The article object
    """
    with get_session() as session:
        article = session.query(Article).filter_by(url=url).first()
        if not article:
            article = Article(url=url, title=title)
            session.add(article)
        return article


def get_or_create_citation(url: str, title: Optional[str] = None) -> Citation:
    """
    Get a citation by URL or create it if it doesn't exist.
    
    Args:
        url: Citation URL
        title: Citation title (optional)
    
    Returns:
        Citation: The citation object
    """
    with get_session() as session:
        citation = session.query(Citation).filter_by(url=url).first()
        if not citation:
            citation = Citation(url=url, title=title)
            session.add(citation)
        return citation


def get_article_by_id(article_id: int) -> Optional[Article]:
    """
    Get an article by its ID.
    
    Args:
        article_id: Article ID
    
    Returns:
        Article or None if not found
    """
    with get_session() as session:
        return session.query(Article).filter_by(article_id=article_id).first()


def get_citation_by_id(citation_id: int) -> Optional[Citation]:
    """
    Get a citation by its ID.
    
    Args:
        citation_id: Citation ID
    
    Returns:
        Citation or None if not found
    """
    with get_session() as session:
        return session.query(Citation).filter_by(citation_id=citation_id).first()


def get_passages_by_citation_id(citation_id: int) -> list[CitationPassage]:
    """
    Get all passages for a citation.
    
    Args:
        citation_id: Citation ID
    
    Returns:
        List of passages
    """
    with get_session() as session:
        return session.query(CitationPassage).filter_by(citation_id=citation_id).all()


def get_passages_by_content_pattern(pattern: str) -> list[CitationPassage]:
    """
    Get passages matching a content pattern.
    
    Args:
        pattern: Search pattern (SQL LIKE)
    
    Returns:
        List of matching passages
    """
    with get_session() as session:
        return session.query(CitationPassage).filter(
            CitationPassage.content.ilike(f"%{pattern}%")
        ).all()


def count_articles() -> int:
    """
    Get the total number of articles.
    
    Returns:
        Number of articles
    """
    with get_session() as session:
        return session.query(func.count(Article.article_id)).scalar()


def count_citations() -> int:
    """
    Get the total number of citations.
    
    Returns:
        Number of citations
    """
    with get_session() as session:
        return session.query(func.count(Citation.citation_id)).scalar()


def count_passages() -> int:
    """
    Get the total number of passages.
    
    Returns:
        Number of passages
    """
    with get_session() as session:
        return session.query(func.count(CitationPassage.passage_id)).scalar()


def get_article_citations(article_id: int) -> list[Citation]:
    """
    Get all citations for an article.
    
    Args:
        article_id: Article ID
    
    Returns:
        List of citations
    """
    with get_session() as session:
        return session.query(Citation).join(
            ArticleCitation,
            ArticleCitation.citation_id == Citation.citation_id
        ).filter(
            ArticleCitation.article_id == article_id
        ).all()


def get_citation_articles(citation_id: int) -> list[Article]:
    """
    Get all articles that cite a document.
    
    Args:
        citation_id: Citation ID
    
    Returns:
        List of articles
    """
    with get_session() as session:
        return session.query(Article).join(
            ArticleCitation,
            ArticleCitation.article_id == Article.article_id
        ).filter(
            ArticleCitation.citation_id == citation_id
        ).all()


def get_statistics() -> dict:
    """
    Get database statistics.
    
    Returns:
        Dictionary with counts for articles, citations, and passages
    """
    return {
        "articles": count_articles(),
        "citations": count_citations(),
        "passages": count_passages()
    }


# Initialize database on module import (can be disabled if needed)
try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    logger.warning("Database initialization failed. Tables may not exist.")
