# -----------------------------
# Extracts all citations from a Wikipedia article
# -----------------------------

from bs4 import BeautifulSoup
import requests
from config import USER_AGENT

def extract_citations(wikipedia_url):
    """
    Extract all citation URLs from a Wikipedia article robustly.

    Handles:
    - <li id="cite_note-..."> items
    - <cite> tags and nested <a> links inside spans
    - Web archives
    - Only returns HTTP/HTTPS URLs

    Returns:
        List[str]: List of citation URLs
    """

    print(f"[INFO] Extracting citations from: {wikipedia_url}")

    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(wikipedia_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch Wikipedia page: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    citations = set()

    # All <li> with id starting 'cite_note'
    for li in soup.find_all("li", id=lambda x: x and x.startswith("cite_note")):
        # Use <cite> if available, else the <li> itself
        cite = li.find("cite") or li

        # Find all <a href> recursively inside <cite>
        for link in cite.find_all("a", href=True):
            url = link['href']

            # Handle relative URLs
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = "https://en.wikipedia.org" + url

            # Only keep HTTP/HTTPS
            if url.startswith("http"):
                citations.add(url)

    print(f"[INFO] Found {len(citations)} reference structures in the html code")
    return list(citations)