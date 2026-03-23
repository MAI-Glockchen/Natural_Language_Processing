import requests
from collections import Counter

BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month}/all-days"

HEADERS = {
    "User-Agent": "MyResearchBot/1.0 (your_email@example.com)"
}

counter = Counter()

for month in range(1, 13):
    url = BASE.format(year=2025, month=str(month).zfill(2))
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    data = r.json()
    articles = data["items"][0]["articles"]

    for a in articles:
        title = a["article"]
        views = a["views"]

        if title.startswith("Special:"):
            continue

        counter[title] += views

top_2000 = [title for title, _ in counter.most_common(2000)]

with open("top_2000_wikipedia_2025.txt", "w", encoding="utf-8") as f:
    for t in top_2000:
        f.write(t + "\n")