# -----------------------------
# Fetch top Wikipedia articles (candidate batch)
# -----------------------------

import requests
import time
from config import USER_AGENT

def get_top_wikipedia_articles(year=2025, lang="en"):
    """
    Fetch top viewed Wikipedia articles per month.
    Returns a list of article URLs.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "accept": "application/json"
    }

    articles = set()

    for month in range(1, 13):
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia.org/all-access/{year}/{month:02d}/all-days"

        print(f"[INFO] Fetching top articles for {year}-{month:02d}")

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            for article in data["items"][0]["articles"]:
                title = article["article"]

                # Skip special pages like "Main_Page"
                if ":" in title:
                    continue

                article_url = f"https://{lang}.wikipedia.org/wiki/{title}"
                articles.add(article_url)

        except Exception as e:
            print(f"[WARNING] Failed month {month}: {e}")

        time.sleep(1)  # polite API usage

    print(f"[INFO] Collected {len(articles)} candidate articles")
    return list(articles)