from __future__ import annotations
from db import init_db, insert_values
from wiki import get_popular_articles, extract_citations
from fetch import fetch_all
from clean import clean_html, make_passages
from util import sha256


def run() -> dict[str, int]:
    print("init db", flush=True)
    init_db()

    print("get popular articles", flush=True)
    articles = get_popular_articles()
    print(f"articles: {len(articles)}", flush=True)

    art_rows = [(a["title"], a["url"], a.get("html", "")) for a in articles]
    art_map = dict(insert_values(
        'insert into wiki_article(title,url,html) values %s on conflict(url) do update set title=excluded.title, html=excluded.html returning id,url',
        art_rows
    ))
    print(f"stored articles: {len(art_map)}", flush=True)

    citations = []
    for i, a in enumerate(articles, start=1):
        print(f"extract citations {i}/{len(articles)}: {a['title']}", flush=True)
        article_url = a["url"]
        for c in extract_citations(article_url):
            c["wiki_url"] = article_url
            citations.append(c)

    print(f"citations: {len(citations)}", flush=True)

    cit_rows = [
        (art_map[c["wiki_url"]], c["url"], c.get("source_host"), c.get("anchor_text", ""), i + 1)
        for i, c in enumerate(citations)
        if c["wiki_url"] in art_map
    ]
    if cit_rows:
        insert_values(
            'insert into citation(wiki_article_id,source_url,source_host,anchor_text,ordinal) values %s on conflict do nothing',
            cit_rows
        )
    print("citations stored", flush=True)

    urls = sorted({c["url"] for c in citations})
    print(f"fetch documents: {len(urls)}", flush=True)
    docs = fetch_all(urls)

    doc_rows = []
    for d in docs:
        cleaned = clean_html(d.html)
        if len(cleaned) < 200:
            continue
        doc_rows.append((d.url, d.final_url, d.status_code, d.content_type, d.html, cleaned, sha256(cleaned), d.fetch_ms))

    print(f"documents after cleaning: {len(doc_rows)}", flush=True)

    doc_map = {}
    if doc_rows:
        doc_map = dict(insert_values(
            'insert into source_document(url,final_url,status_code,content_type,html,cleaned_text,content_sha256,fetch_ms) '
            'values %s on conflict(url) do update set final_url=excluded.final_url,status_code=excluded.status_code,content_type=excluded.content_type,html=excluded.html,cleaned_text=excluded.cleaned_text,content_sha256=excluded.content_sha256,fetch_ms=excluded.fetch_ms returning id,url',
            doc_rows, page_size=250
        ))
    print(f"stored documents: {len(doc_map)}", flush=True)

    bridge = [(art_map[c["wiki_url"]], doc_map[c["url"]]) for c in citations if c["url"] in doc_map]
    if bridge:
        insert_values('insert into article_document(wiki_article_id,source_document_id) values %s on conflict do nothing', bridge)
    print(f"bridges: {len(bridge)}", flush=True)

    passages = []
    for row in doc_rows:
        doc_id = doc_map[row[0]]
        for idx, txt, wc in make_passages(row[5]):
            passages.append((doc_id, idx, txt, wc))

    if passages:
        insert_values('insert into passage(source_document_id,idx,text,word_count) values %s on conflict do nothing', passages, page_size=1000)

    print(f"passages: {len(passages)}", flush=True)

    return {
        'articles': len(articles),
        'citations': len(citations),
        'documents': len(doc_rows),
        'passages': len(passages),
    }