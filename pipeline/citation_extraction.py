# -----------------------------
# Extracts all citations from a Wikipedia article
# -----------------------------

from bs4 import BeautifulSoup
import requests
from config import USER_AGENT

def extract_citations(wikipedia_url):
    """
    Args:
        wikipedia_url (str): URL of the Wikipedia article
    Returns:
        List[str]: List of citation URLs
    """
    print(f"[INFO] Extracting citations from: {wikipedia_url}")
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(wikipedia_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    citations = set()
    # Find all <li> tags with id="cite_note-..."
    for li in soup.find_all("li", id=lambda x: x and x.startswith("cite_note")):
        cite = li.find("cite")
        if cite:
            for link in cite.find_all("a", href=True):
                url = link['href']
                if url.startswith("//"):
                    url = "https:" + url
                elif url.startswith("/"):
                    url = "https://en.wikipedia.org" + url
                if url.startswith("http"):
                    citations.add(url)

    print(f"[INFO] Found {len(citations)} citations")
    return list(citations)