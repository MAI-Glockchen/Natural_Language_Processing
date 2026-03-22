# pipeline/db_saving.py
# -----------------------------
# Stores passages in the PostgreSQL database
# -----------------------------

from db.session import session
from db.models import Article, Citation, CitationPassage

def save_passages_to_db(article_url, passages, citation_url):
    """
    Args:
        article_url (str): Wikipedia article URL
        passages (List[str]): List of text passages
        citation_url (str): Citation URL
    """
    # Create article if not exists
    article = session.query(Article).filter_by(url=article_url).first()
    if not article:
        article = Article(url=article_url, title=article_url.split("/")[-1])
        session.add(article)
        session.commit()

    # Create citation if not exists
    citation = session.query(Citation).filter_by(url=citation_url).first()
    if not citation:
        citation = Citation(url=citation_url, title=citation_url.split("/")[-1])
        session.add(citation)
        session.commit()

    # Link article to citation
    if citation not in article.citations:
        article.citations.append(citation)
        session.commit()

    # Save passages
    for p in passages:
        cp = CitationPassage(citation_id=citation.citation_id, content=p)
        session.add(cp)
    session.commit()