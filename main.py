# -----------------------------
# Entry point to run the Wikipedia citation pipeline
# -----------------------------

import time
from utils.nltk_setup import *
from pipeline import extract_citations, fetch_document, clean_html, create_passages, save_passages_to_db

def process_wikipedia_article(wikipedia_url, max_docs=20):
    """
    Pipeline:
    1. Extract citations
    2. Fetch documents
    3. Clean HTML
    4. Create passages
    5. Save passages to PostgreSQL
    """
    citations = extract_citations(wikipedia_url)
    for url in citations[:max_docs]:
        html = fetch_document(url)
        if html:
            clean_text = clean_html(html)
            if len(clean_text) > 200:  # Filter too-short content
                passages = create_passages(clean_text)
                save_passages_to_db(wikipedia_url, passages, url)
        time.sleep(1)  # Prevent IP blocking
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    process_wikipedia_article(wiki_url)
    print("[DONE] All passages stored in PostgreSQL")