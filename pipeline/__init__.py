# -----------------------------
# Pipeline package initializer
# Optional: central imports for convenience
# -----------------------------

from .citation_extraction import extract_citations
from .document_fetching import fetch_document
from .document_cleaning import clean_html
from .passage_creation import create_passages
from .db_saving import save_passages_to_db