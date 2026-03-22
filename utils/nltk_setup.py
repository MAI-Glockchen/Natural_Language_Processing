# -----------------------------
# NLTK setup for sentence tokenizer
# -----------------------------
# PUNKT: standard tokenizer for sentence segmentation
# Handles abbreviations like "Dr.", "e.g.", "i.e."
# Downloads tokenizer if not already available
# -----------------------------

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("[INFO] Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')