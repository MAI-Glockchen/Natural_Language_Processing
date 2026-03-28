# -----------------------------
# Entry point to run the Wikipedia citation pipeline
# -----------------------------

import time
from utils.nltk_setup import *
from pipeline import (extract_citations, fetch_document, clean_html, create_passages,
                      save_passages_to_db, collect_valid_articles)

# max_docs could be removed , but stays for performance reasons
def process_wikipedia_article(wikipedia_url, max_docs=50):
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
        try:
            html = fetch_document(url)

            if html:
                clean_text = clean_html(html)

                # Filter out very short or useless content
                if len(clean_text) > 200:
                    passages = create_passages(clean_text)

                    # Save passages to database
                    save_passages_to_db(wikipedia_url, passages, url)

            time.sleep(1)  # Avoid being blocked by servers

        except Exception as e:
            print(f"[ERROR] Failed processing citation {url}: {e}")

    print(f"[INFO] Finished article: {wikipedia_url}")



if __name__ == "__main__":
    # Step 1: collect 420 valid English Wikipedia articles
    article_urls = collect_valid_articles(target_count=5, min_citations=20)

    # Step 2: run your existing pipeline for each article
    for url in article_urls:
        print(f"[INFO] Processing full pipeline for {url}")
        try:
            process_wikipedia_article(url)
        except Exception as e:
            print(f"[ERROR] Failed processing article {url}: {e}")