# -----------------------------
# Pipeline package initializer
# Optional: central imports for convenience
# -----------------------------

from .citation_extraction import extract_citations
from .document_fetching import fetch_document
from .passage_creation import create_passages
from .topic_inference import infer_topic
from .vector_index import PassageVectorIndex


def _missing_dependency_factory(feature_name, install_hint):
    def _missing_dependency(*args, **kwargs):
        raise ImportError(
            f"{feature_name} is unavailable because an optional dependency is missing. "
            f"Install with: {install_hint}"
        )

    return _missing_dependency


try:
    from .document_cleaning import clean_html
except Exception:
    clean_html = _missing_dependency_factory(
        "clean_html", "pip install readability-lxml beautifulsoup4"
    )


try:
    from .db_saving import save_passages_to_db
except Exception:
    save_passages_to_db = _missing_dependency_factory(
        "save_passages_to_db", "pip install sqlalchemy psycopg2-binary"
    )
