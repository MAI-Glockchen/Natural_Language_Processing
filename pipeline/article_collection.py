"""
Collect top Wikipedia articles and filter them based on citation quality.
"""

import requests
import time

from config import USER_AGENT
from .citation_extraction import extract_citations
from .document_fetching import fetch_document
from .wikipedia_top_articles import get_top_wikipedia_articles

def collect_valid_articles(target_count=420, min_citations=20):
    """
    Collects a list of Wikipedia article URLs with at least min_citations accessible references.
    """
    valid_articles = []

    while len(valid_articles) < target_count:
        candidate_urls = get_top_wikipedia_articles()

        for url in candidate_urls:
            citations = extract_citations(url)

            accessible = []
            for c in citations:
                if len(accessible) >= min_citations:
                    # Early exit: we already have enough reachable citations
                    break
                try:
                    r = requests.head(c, headers={"User-Agent": USER_AGENT}, timeout=5)
                    if r.status_code == 200:
                        accessible.append(c)
                except:
                    continue

            if len(accessible) >= min_citations:
                valid_articles.append(url)
                print(f"[INFO] Added valid article: {url} (at least {len(accessible)} accessible citations)")

            if len(valid_articles) >= target_count:
                break

    print(f"[INFO] Collected {len(valid_articles)} valid articles")
    return valid_articles