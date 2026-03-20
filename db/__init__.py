# -----------------------------
# DB package initializer
# Optional: central imports for convenience
# -----------------------------

from .models import Base, Article, Citation, ArticleCitation, CitationPassage
from .session import session