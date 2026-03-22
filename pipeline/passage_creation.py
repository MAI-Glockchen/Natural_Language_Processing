# -----------------------------
# Splits text into short, semantically coherent passages
# -----------------------------

from nltk.tokenize import sent_tokenize

def create_passages(text, max_sentences=3):
    """
    Args:
        text (str): Cleaned text
        max_sentences (int): Max sentences per passage
    Returns:
        List[str]: List of text passages
    """
    sentences = sent_tokenize(text)
    passages = []
    current = []
    for sent in sentences:
        current.append(sent)
        if len(current) >= max_sentences:
            passages.append(" ".join(current))
            current = []
    if current:
        passages.append(" ".join(current))
    return passages