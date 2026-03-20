import requests
from bs4 import BeautifulSoup
from readability import Document
import nltk
from nltk.tokenize import sent_tokenize
import time

# Needs to be executed first
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# -----------------------------
# 1. Citation Extraction
# -----------------------------

def extract_citations(wikipedia_url):
    print(f"[INFO] Extracting citations from: {wikipedia_url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.get(wikipedia_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    citations = set()

    # Suche alle <li> mit id="cite_note-..."
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


# -----------------------------
# 2. Document Collection
# -----------------------------

def fetch_document(url):
    try:
        print(f"[INFO] Fetching: {url}")
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        if "text/html" not in response.headers.get("Content-Type", ""):
            return None

        return response.text
    except:
        print(f"[WARNING] Failed to fetch: {url}")
        return None


# -----------------------------
# 3. Document Cleaning
# -----------------------------

def clean_html(html):
    try:
        doc = Document(html)
        main_content = doc.summary()

        soup = BeautifulSoup(main_content, "html.parser")
        text = soup.get_text(separator=" ")

        # basic cleaning
        text = " ".join(text.split())

        return text
    except:
        return ""


# -----------------------------
# 4. Passage Creation
# -----------------------------

def create_passages(text, max_sentences=3):
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


# -----------------------------
# Pipeline
# -----------------------------

def process_wikipedia_article(wikipedia_url, max_docs=20):
    citations = extract_citations(wikipedia_url)

    documents = []

    for url in citations[:max_docs]:
        html = fetch_document(url)
        if html:
            clean_text = clean_html(html)
            if len(clean_text) > 200:  # filter garbage
                documents.append(clean_text)

        time.sleep(1)  # avoid getting blocked

    print(f"[INFO] Collected {len(documents)} documents")

    all_passages = []

    for doc in documents:
        passages = create_passages(doc)
        all_passages.extend(passages)

    print(f"[INFO] Created {len(all_passages)} passages")

    return all_passages


# -----------------------------
# Example Run
# -----------------------------

if __name__ == "__main__":

    wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    passages = process_wikipedia_article(wiki_url)

    print(passages[:5])

    # Save to file
    with open("passages.txt", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(p + "\n")

    print("[DONE] Passages saved to passages.txt")