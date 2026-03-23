# -----------------------------
# Stores passages in the PostgreSQL database
# -----------------------------

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from db.session import get_session
from db.models import Article, Citation, CitationPassage
from config import MIN_TEXT_LENGTH

logger = logging.getLogger(__name__)


def save_passages_to_db(
    article_url: str,
    passages: List[str],
    citation_url: str,
    session: Optional[Session] = None
) -> bool:
    """
    Save passages to the database with proper error handling.

    Args:
        article_url: Wikipedia article URL
        passages: List of text passages
        citation_url: Citation URL
        session: Optional database session (will create new if not provided)

    Returns:
        bool: True if successful, False otherwise
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
            return False

        logger.info(f"Saving {len(valid_passages)} passages for {citation_url}")

        # Create article if not exists
        article = session.query(Article).filter_by(url=article_url).first()
        if not article:
            article = Article(
                url=article_url,
                title=article_url.split("/")[-1]
            )
            session.add(article)

        # Create citation if not exists
        citation = session.query(Citation).filter_by(url=citation_url).first()
        if not citation:
            citation = Citation(
                url=citation_url,
                title=citation_url.split("/")[-1]
            )
            session.add(citation)

        # Link article to citation
        if citation not in article.citations:
            article.citations.append(citation)

        # Save passages
        for p in valid_passages:
            cp = CitationPassage(
                citation_id=citation.citation_id,
                content=p
            )
            session.add(cp)

        session.commit()
        logger.info(f"Successfully saved {len(valid_passages)} passages")
        return True

    except Exception as e:
        logger.error(f"Failed to save passages for {citation_url}: {e}")
        session.rollback()
        return False

    finally:
        if session:
            session.close()


def save_passages_batch(
    article_url: str,
    citations_passages: List[tuple],
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
            if save_passages_to_db(article_url, passages, citation_url, session):
                total_saved += len(passages)

        session.commit()
        logger.info(f"Batch saved {total_saved} passages total")
        return total_saved

    except Exception as e:
        logger.error(f"Batch save failed: {e}")
        session.rollback()
        raise

    finally:
        if session:
            session.close()
