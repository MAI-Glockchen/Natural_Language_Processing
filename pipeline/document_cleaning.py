# -----------------------------
# Cleans HTML and extracts readable text
# -----------------------------

from readability import Document
from bs4 import BeautifulSoup

def clean_html(html):
    """
    Args:
        html (str): Raw HTML
    Returns:
        str: Clean text
    """
    try:
        doc = Document(html)
        main_content = doc.summary()
        soup = BeautifulSoup(main_content, "html.parser")
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())  # Remove extra whitespace
        return text
    except:
        return ""