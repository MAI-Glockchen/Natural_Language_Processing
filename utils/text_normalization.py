# -----------------------------
# Shared text normalization utilities
# -----------------------------

import re
import unicodedata

try:
    import ftfy
except Exception:
    ftfy = None

try:
    from acronym_expander import AcronymExpander
except Exception:
    AcronymExpander = None


_ACRONYM_EXPANDER = AcronymExpander() if AcronymExpander is not None else None


def _expand_abbreviations(text):
    """
    Expand abbreviations using a dictionary-backed acronym library.
    Example: AI -> Artificial Intelligence, ML -> Machine Learning.
    """
    if _ACRONYM_EXPANDER is None:
        return text

    try:
        return _ACRONYM_EXPANDER.expand(text)
    except Exception:
        return text


def _collapse_expanded_pattern(text):
    """
    Collapse patterns produced by some expanders:
    'AI (Artificial Intelligence)' -> 'Artificial Intelligence'.
    """
    pattern = re.compile(r"\b([A-Za-z]{2,10})\s*\(\s*([^()]+?)\s*\)")

    def _replace(match):
        short_form = match.group(1)
        long_form = match.group(2).strip()

        # Only collapse when short form looks like an abbreviation token.
        if short_form.isupper() or short_form.lower() == short_form:
            return long_form
        return match.group(0)

    return pattern.sub(_replace, text)


def normalize_text(text):
    """Normalize text using library-backed and unicode-safe cleanup."""
    if text is None:
        return ""

    normalized = str(text)

    # Fix mojibake and common unicode glitches when available.
    if ftfy is not None:
        normalized = ftfy.fix_text(normalized)

    normalized = _expand_abbreviations(normalized)
    normalized = _collapse_expanded_pattern(normalized)

    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized
