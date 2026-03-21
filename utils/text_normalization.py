# -----------------------------
# Shared text normalization utilities
# -----------------------------

import re
import unicodedata

try:
    import ftfy
except Exception:
    ftfy = None


def normalize_text(text):
    """Normalize text using library-backed and unicode-safe cleanup."""
    if text is None:
        return ""

    normalized = str(text)

    # Fix mojibake and common unicode glitches when available.
    if ftfy is not None:
        normalized = ftfy.fix_text(normalized)

    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized
