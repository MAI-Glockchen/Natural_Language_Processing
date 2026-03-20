# -----------------------------
# Fetches HTML documents from URLs
# -----------------------------

import requests
from config import USER_AGENT

def fetch_document(url):
    """
    Args:
        url (str): Citation URL
    Returns:
        str|None: HTML text or None if failed
    """
    try:
        print(f"[INFO] Fetching: {url}")
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if response.status_code != 200:
            return None
        if "text/html" not in response.headers.get("Content-Type", ""):
            return None
        return response.text
    except:
        print(f"[WARNING] Failed to fetch: {url}")
        return None