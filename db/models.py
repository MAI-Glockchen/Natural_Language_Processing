# -----------------------------
# SQLAlchemy models
# Defines the PostgreSQL tables for articles, citations, and passages
# -----------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Index, func
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
import datetime

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

Base = declarative_base()


class Article(Base):
    """
    Represents a Wikipedia article with its metadata.
    
    Attributes:
        article_id: Primary key
        url: Wikipedia article URL (unique)
        title: Article title
        created_at: Timestamp when the article was created
    """
    __tablename__ = "articles"
    
    article_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    url: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False, index=True
    )
    
    # Relation to citations (many-to-many)
    citations: Mapped[list["Citation"]] = relationship(
        "Citation",
        secondary="article_citations",
        back_populates="articles",
        lazy="selectin"
    )
    
    __table_args__ = (
        Index('idx_articles_url', 'url'),
        Index('idx_articles_created_at', 'created_at'),
    )


class Citation(Base):
    """
    Represents a cited document from a Wikipedia article.
    
    Attributes:
        citation_id: Primary key
        url: Citation document URL (unique)
        title: Citation document title
        created_at: Timestamp when the citation was created
    """
    __tablename__ = "citations"
    
    citation_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    url: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False, index=True
    )
    
    # Relation to articles (many-to-many)
    articles: Mapped[list["Article"]] = relationship(
        "Article",
        secondary="article_citations",
        back_populates="citations",
        lazy="selectin"
    )
    
    # Relation to passages (one-to-many)
    passages: Mapped[list["CitationPassage"]] = relationship(
        "CitationPassage",
        back_populates="citation",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    __table_args__ = (
        Index('idx_citations_url', 'url'),
        Index('idx_citations_created_at', 'created_at'),
    )


class ArticleCitation(Base):
    """
    Junction table for the many-to-many relationship between articles and citations.
    
    Attributes:
        article_id: Foreign key to articles
        citation_id: Foreign key to citations
    """
    __tablename__ = "article_citations"
    
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.article_id", ondelete="CASCADE"),
        primary_key=True
    )
    citation_id: Mapped[int] = mapped_column(
        ForeignKey("citations.citation_id", ondelete="CASCADE"),
        primary_key=True
    )
    
    __table_args__ = (
        Index('idx_article_citations_article_id', 'article_id'),
        Index('idx_article_citations_citation_id', 'citation_id'),
    )


class CitationPassage(Base):
    """
    Represents a passage extracted from a citation document.
    
    Attributes:
        passage_id: Primary key
        citation_id: Foreign key to the citation
        content: Text content of the passage
    """
    __tablename__ = "citation_passages"
    
    passage_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    citation_id: Mapped[int] = mapped_column(
        ForeignKey("citations.citation_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Relation to citation
    citation: Mapped["Citation"] = relationship(
        "Citation",
        back_populates="passages",
        lazy="selectin"
    )
    
    __table_args__ = (
        Index('idx_passages_citation_id', 'citation_id'),
        Index('idx_passages_content', 'content'),
    )
