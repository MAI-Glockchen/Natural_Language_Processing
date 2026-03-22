# -----------------------------
# Functionality to fetch random English Wikipedia articles, filter them
# by accessible citations, and save them in the database.
# -----------------------------

import time
import random
import requests
from config import USER_AGENT
from . import extract_citations, fetch_document, create_passages, save_passages_to_db

MAX_CITATIONS = 20  # minimum accessible citations required


def get_random_wikipedia_articles(n=50, lang="en"):
    """Fetch a batch of random Wikipedia article URLs using the API."""
    url = f"https://{lang}.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": USER_AGENT
    }

    params = {
        "action": "query",
        "list": "random",
        "rnlimit": n,
        "format": "json",
        "rnnamespace": 0,  # only articles
    }
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    articles = [f"https://{lang}.wikipedia.org/wiki/{a['title'].replace(' ', '_')}"
                for a in r.json()['query']['random']]
    return articles


def collect_valid_articles(target_count=420, batch_size=50, lang="en"):
    """
    Collects up to `target_count` random Wikipedia articles that have at least
    MAX_CITATIONS accessible citations.
    """
    valid_articles = []

    while len(valid_articles) < target_count:
        batch = get_random_wikipedia_articles(batch_size, lang)
        for article_url in batch:
            print(f"[INFO] Processing {article_url}")
            citations = extract_citations(article_url)
            accessible = [url for url in citations if fetch_document(url) is not None]
            if len(accessible) >= MAX_CITATIONS and article_url not in valid_articles:
                valid_articles.append(article_url)
                print(f"[INFO] Added article: {article_url} with {len(accessible)} accessible citations")
            time.sleep(1)  # avoid rate limits

        print(f"[INFO] Collected {len(valid_articles)} valid articles so far")

    random.shuffle(valid_articles)
    return valid_articles[:target_count]


def process_and_save_articles(article_urls, max_docs_per_article=20):
    """
    For each article URL, fetch and clean documents, create passages,
    and save everything in the database.
    """
    for article_url in article_urls:
        citations = extract_citations(article_url)
        documents = []

        for url in citations[:max_docs_per_article]:
            html = fetch_document(url)
            if html:
                text = create_passages(html)  # reuse pipeline for passages
                if len(text) > 0:
                    documents.append(text)

            time.sleep(1)  # polite scraping

        for citation_url, doc_text in zip(citations[:max_docs_per_article], documents):
            save_passages_to_db(article_url, doc_text, citation_url)