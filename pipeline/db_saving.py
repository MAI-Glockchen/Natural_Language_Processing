# -----------------------------
# Stores passages in the PostgreSQL database
# -----------------------------

from __future__ import annotations
from typing import List, Optional, Tuple
from contextlib import contextmanager
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from db.session import get_session, get_or_create_article, get_or_create_citation
from db.models import Article, Citation, CitationPassage
from config import MIN_TEXT_LENGTH

logger = logging.getLogger(__name__)


@contextmanager
def get_db_session() -> Session:
    """
    Get a database session with automatic cleanup.
    
    Yields:
        Session: Database session
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def save_passages_to_db(
    article_url: str,
    passages: List[str],
    citation_url: str,
    session: Optional[Session] = None
) -> Tuple[bool, int]:
    """
    Save passages to the database with proper error handling.

    Args:
        article_url: Wikipedia article URL
        passages: List of text passages
        citation_url: Citation URL
        session: Optional database session (will create new if not provided)

    Returns:
        Tuple[bool, int]: (success, number of passages saved)
    """
    if not session:
        session = get_session()

    try:
        # Validate passages
        valid_passages = [
            p for p in passages
            if p and len(p) >= MIN_TEXT_LENGTH
        ]

        if not valid_passages:
            logger.warning(f"No valid passages to save for {citation_url}")
            return False, 0
        
        logger.info(f"Saving {len(valid_passages)} passages for {citation_url}")
        
        # Get or create article
        article = session.query(Article).filter_by(url=article_url).first()
        if not article:
            article = Article(
                url=article_url,
                title=article_url.split("/")[-1]
            )
            session.add(article)
        
        # Get or create citation
        citation = session.query(Citation).filter_by(url=citation_url).first()
        if not citation:
            citation = Citation(
                url=citation_url,
                title=citation_url.split("/")[-1]
            )
            session.add(citation)
        
        # Link article to citation if not already linked
        if citation not in article.citations:
            article.citations.append(citation)
        
        # Save passages in batch
        saved_count = 0
        for p in valid_passages:
            cp = CitationPassage(
                citation_id=citation.citation_id,
                content=p
            )
            session.add(cp)
            saved_count += 1
        
        logger.info(f"Successfully saved {saved_count} passages")
        return True, saved_count
        
    except Exception as e:
        logger.error(f"Failed to save passages for {citation_url}: {e}")
        session.rollback()
        return False, 0
        
    finally:
        if session:
            session.close()


def save_passages_batch(
    article_url: str,
    citations_passages: List[Tuple[str, List[str]]],
    session: Optional[Session] = None
) -> int:
    """
    Save multiple citations and their passages in a batch.

    Args:
        article_url: Wikipedia article URL
        citations_passages: List of (citation_url, passages) tuples
        session: Optional database session

    Returns:
        int: Number of passages saved
    """
    if not session:
        session = get_session()

    try:
        total_saved = 0

        for citation_url, passages in citations_passages:
            success, count = save_passages_to_db(
                article_url, passages, citation_url, session
            )
            if success:
                total_saved += count
        
        logger.info(f"Batch saved {total_saved} passages total")
        return total_saved

    except Exception as e:
        logger.error(f"Batch save failed: {e}")
        session.rollback()
        raise
        
    finally:
        if session:
            session.close()


def save_passages_batch_fast(
    article_url: str,
    citations_passages: List[Tuple[str, List[str]]],
    session: Optional[Session] = None
) -> int:
    """
    Save multiple citations and their passages using bulk insert for better performance.
    
    Args:
        article_url: Wikipedia article URL
        citations_passages: List of (citation_url, passages) tuples
        session: Optional database session
    
    Returns:
        int: Number of passages saved
    """
    if not session:
        session = get_session()
    
    try:
        total_saved = 0
        
        # First pass: get or create articles and citations
        article = session.query(Article).filter_by(url=article_url).first()
        if not article:
            article = Article(
                url=article_url,
                title=article_url.split("/")[-1]
            )
            session.add(article)
        
        # Get or create all citations
        citation_map = {}
        for citation_url, _ in citations_passages:
            citation = session.query(Citation).filter_by(url=citation_url).first()
            if not citation:
                citation = Citation(
                    url=citation_url,
                    title=citation_url.split("/")[-1]
                )
                session.add(citation)
            citation_map[citation_url] = citation
        
        # Link article to all citations
        for citation in citation_map.values():
            if citation not in article.citations:
                article.citations.append(citation)
        
        # Second pass: bulk insert passages
        from sqlalchemy.orm import bulk_save_to_db
        
        for citation_url, passages in citations_passages:
            citation = citation_map[citation_url]
            
            # Filter valid passages
            valid_passages = [
                p for p in passages
                if p and len(p) >= MIN_TEXT_LENGTH
            ]
            
            if not valid_passages:
                continue
            
            # Create passage objects
            passages_to_save = [
                CitationPassage(
                    citation_id=citation.citation_id,
                    content=p
                )
                for p in valid_passages
            ]
            
            # Bulk insert
            session.bulk_save_objects(passages_to_save)
            total_saved += len(passages_to_save)
        
        logger.info(f"Batch fast saved {total_saved} passages total")
        return total_saved
        
    except Exception as e:
        logger.error(f"Batch fast save failed: {e}")
        session.rollback()
        raise
        
    finally:
        if session:
            session.close()


def get_passages_for_citation(citation_url: str) -> List[str]:
    """
    Get all passages for a citation URL.
    
    Args:
        citation_url: Citation URL
    
    Returns:
        List of passage contents
    """
    with get_db_session() as session:
        citation = session.query(Citation).filter_by(url=citation_url).first()
        if not citation:
            logger.warning(f"Citation not found: {citation_url}")
            return []
        
        passages = session.query(CitationPassage).filter_by(
            citation_id=citation.citation_id
        ).order_by(CitationPassage.passage_id).all()
        
        return [p.content for p in passages]


def get_passages_for_article(article_url: str) -> List[Tuple[str, List[str]]]:
    """
    Get all passages for an article with their citation URLs.
    
    Args:
        article_url: Article URL
    
    Returns:
        List of (citation_url, [passages]) tuples
    """
    with get_db_session() as session:
        article = session.query(Article).filter_by(url=article_url).first()
        if not article:
            logger.warning(f"Article not found: {article_url}")
            return []
        
        results = []
        for citation in article.citations:
            passages = session.query(CitationPassage).filter_by(
                citation_id=citation.citation_id
            ).order_by(CitationPassage.passage_id).all()
            
            results.append((citation.url, [p.content for p in passages]))
        
        return results


def get_statistics() -> dict:
    """
    Get database statistics.
    
    Returns:
        Dictionary with counts for articles, citations, and passages
    """
    with get_db_session() as session:
        from sqlalchemy import func
        
        return {
            "articles": session.query(func.count(Article.article_id)).scalar(),
            "citations": session.query(func.count(Citation.citation_id)).scalar(),
            "passages": session.query(func.count(CitationPassage.passage_id)).scalar()
        }


def delete_passages_for_citation(citation_url: str) -> bool:
    """
    Delete all passages for a citation.
    
    Args:
        citation_url: Citation URL
    
    Returns:
        bool: True if successful
    """
    with get_db_session() as session:
        citation = session.query(Citation).filter_by(url=citation_url).first()
        if not citation:
            logger.warning(f"Citation not found: {citation_url}")
            return False
        
        session.query(CitationPassage).filter_by(
            citation_id=citation.citation_id
        ).delete()
        
        logger.info(f"Deleted all passages for {citation_url}")
        return True


def delete_citation(citation_url: str) -> bool:
    """
    Delete a citation and all its passages.
    
    Args:
        citation_url: Citation URL
    
    Returns:
        bool: True if successful
    """
    with get_db_session() as session:
        citation = session.query(Citation).filter_by(url=citation_url).first()
        if not citation:
            logger.warning(f"Citation not found: {citation_url}")
            return False
        
        session.delete(citation)
        
        logger.info(f"Deleted citation {citation_url}")
        return True


def delete_article(article_url: str) -> bool:
    """
    Delete an article and all its citations and passages.
    
    Args:
        article_url: Article URL
    
    Returns:
        bool: True if successful
    """
    with get_db_session() as session:
        article = session.query(Article).filter_by(url=article_url).first()
        if not article:
            logger.warning(f"Article not found: {article_url}")
            return False
        
        session.delete(article)
        
        logger.info(f"Deleted article {article_url}")
        return True
