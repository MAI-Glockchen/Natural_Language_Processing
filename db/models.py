# -----------------------------
# SQLAlchemy models
# Defines the PostgreSQL tables for articles, citations, and passages
# -----------------------------

from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, UniqueConstraint
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
    citations = relationship(
        "Citation", secondary="article_citations", back_populates="articles"
    )
    topic_output = relationship(
        "ArticleTopicOutput", back_populates="article", uselist=False
    )
    faiss_passages = relationship("FaissPassageMap", back_populates="article")


class Citation(Base):
    __tablename__ = "citations"
    citation_id = Column(Integer, primary_key=True)
    url = Column(String, unique=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Relation to article
    articles = relationship(
        "Article", secondary="article_citations", back_populates="citations"
    )
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


class ArticleTopicOutput(Base):
    __tablename__ = "article_topic_outputs"

    topic_output_id = Column(Integer, primary_key=True)
    article_id = Column(
        Integer, ForeignKey("articles.article_id"), unique=True, index=True
    )
    topic = Column(String)
    candidates_json = Column(Text)
    index_file = Column(String)
    index_backend = Column(String)
    embedding_dim = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    article = relationship("Article", back_populates="topic_output")


class FaissPassageMap(Base):
    __tablename__ = "faiss_passage_map"
    __table_args__ = (
        UniqueConstraint("article_id", "faiss_row_id", name="uq_faiss_map_article_row"),
    )

    map_id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.article_id"), index=True)
    faiss_row_id = Column(Integer, index=True)
    index_file = Column(String)
    passage_key = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    article = relationship("Article", back_populates="faiss_passages")
