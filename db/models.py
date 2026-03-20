# -----------------------------
# SQLAlchemy models
# Defines the PostgreSQL tables for articles, citations, and passages
# -----------------------------

from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    article_id = Column(Integer, primary_key=True)
    url = Column(String, unique=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Relation to citation (article <-> citation)
    citations = relationship("Citation", secondary="article_citations", back_populates="articles")

class Citation(Base):
    __tablename__ = "citations"
    citation_id = Column(Integer, primary_key=True)
    url = Column(String, unique=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Relation to article
    articles = relationship("Article", secondary="article_citations", back_populates="citations")
    # Relation to passage
    passages = relationship("CitationPassage", back_populates="citation")

# n: m table btw article and citation
class ArticleCitation(Base):
    __tablename__ = "article_citations"
    article_id = Column(Integer, ForeignKey("articles.article_id"), primary_key=True)
    citation_id = Column(Integer, ForeignKey("citations.citation_id"), primary_key=True)

# Passages of one citation
class CitationPassage(Base):
    __tablename__ = "citation_passages"
    passage_id = Column(Integer, primary_key=True)
    citation_id = Column(Integer, ForeignKey("citations.citation_id"))
    content = Column(Text)
    citation = relationship("Citation", back_populates="passages")