from __future__ import annotations

from clean import clean_html, make_passages
from db import init_db, insert_values
from fetch import fetch_article_documents_until_threshold, source_host
from util import sha256
from wiki import extract_citations, get_popular_articles


def _strip_nul(text: str) -> str:
    return (text or "").replace("\x00", "")


def run() -> dict[str, int]:
    print("init db", flush=True)
    init_db()

    print("get popular articles", flush=True)
    articles = get_popular_articles()
    print(f"articles: {len(articles)}", flush=True)

    art_rows = [
        (_strip_nul(a["title"]), _strip_nul(a["url"]), _strip_nul(a.get("html", "")))
        for a in articles
    ]
    art_result = insert_values(
        "insert into wiki_article(title,url,html) values %s "
        "on conflict(url) do update set title=excluded.title, html=excluded.html "
        "returning id,url",
        art_rows,
    )
    art_map = {url: article_id for article_id, url in art_result}
    print(f"stored articles: {len(art_map)}", flush=True)

    qualified_articles = 0
    citations_total = 0
    documents_total = 0
    passages_total = 0

    for i, a in enumerate(articles, start=1):
        print(f"extract citations {i}/{len(articles)}: {a['title']}", flush=True)

        article_url = a["url"]
        article_id = art_map.get(article_url)
        if article_id is None:
            print(f"skip article without id: {a['title']}", flush=True)
            continue

        raw_citations = extract_citations(article_url)
        total = len(raw_citations)
        print(f"raw citations: {total}", flush=True)

        if total == 0:
            print(f"drop article: {a['title']} (no citations)", flush=True)
            continue

        max_missing = total // 10
        urls = [c["url"] for c in raw_citations]

        print(
            f"fetch article docs: {a['title']} | total={total} allowed_missing={max_missing}",
            flush=True,
        )
        docs, missing, aborted = fetch_article_documents_until_threshold(urls, max_missing)
        print(
            f"fetch result: {a['title']} | docs={len(docs)} missing={missing} aborted={aborted}",
            flush=True,
        )

        if aborted:
            print(
                f"drop article: {a['title']} "
                f"(missing={missing}, allowed={max_missing}, total={total})",
                flush=True,
            )
            continue

        available = {d.url for d in docs}
        available_citations = []
        for c in raw_citations:
            if c["url"] in available:
                c["wiki_url"] = article_url
                c["source_host"] = source_host(c["url"])
                available_citations.append(c)

        print(f"available citations: {len(available_citations)}/{total}", flush=True)

        if len(available_citations) < total - max_missing:
            print(
                f"drop article late: {a['title']} "
                f"(available={len(available_citations)}, total={total}, required={total - max_missing})",
                flush=True,
            )
            continue

        qualified_articles += 1

        cit_rows = [
            (
                article_id,
                _strip_nul(c["url"]),
                _strip_nul(c.get("source_host") or ""),
                _strip_nul(c.get("anchor_text", "")),
                idx + 1,
            )
            for idx, c in enumerate(available_citations)
        ]
        if cit_rows:
            insert_values(
                "insert into citation(wiki_article_id,source_url,source_host,anchor_text,ordinal) "
                "values %s on conflict do nothing",
                cit_rows,
            )
        citations_total += len(available_citations)

        dedup_docs: dict[str, object] = {}
        for d in docs:
            if d.url not in dedup_docs:
                dedup_docs[d.url] = d

        doc_rows = []
        for d in dedup_docs.values():
            raw_html = _strip_nul(d.html)
            cleaned = clean_html(raw_html)
            if len(cleaned) < 200:
                continue
            doc_rows.append(
                (
                    _strip_nul(d.url),
                    _strip_nul(d.final_url),
                    d.status_code,
                    _strip_nul(d.content_type),
                    raw_html,
                    cleaned,
                    sha256(cleaned),
                    d.fetch_ms,
                )
            )

        print(f"documents after cleaning: {len(doc_rows)}", flush=True)

        doc_map = {}
        if doc_rows:
            doc_result = insert_values(
                "insert into source_document(url,final_url,status_code,content_type,html,cleaned_text,content_sha256,fetch_ms) "
                "values %s "
                "on conflict(url) do update set "
                "final_url=excluded.final_url,"
                "status_code=excluded.status_code,"
                "content_type=excluded.content_type,"
                "html=excluded.html,"
                "cleaned_text=excluded.cleaned_text,"
                "content_sha256=excluded.content_sha256,"
                "fetch_ms=excluded.fetch_ms "
                "returning id,url",
                doc_rows,
                page_size=250,
            )
            doc_map = {url: doc_id for doc_id, url in doc_result}

        documents_total += len(doc_rows)

        bridge = [
            (article_id, doc_map[c["url"]])
            for c in available_citations
            if c["url"] in doc_map
        ]
        if bridge:
            insert_values(
                "insert into article_document(wiki_article_id,source_document_id) "
                "values %s on conflict do nothing",
                bridge,
            )

        passages = []
        for row in doc_rows:
            doc_id = doc_map[row[0]]
            for idx, txt, wc in make_passages(row[5]):
                passages.append((doc_id, idx, txt, wc))

        if passages:
            insert_values(
                "insert into passage(source_document_id,idx,text,word_count) "
                "values %s on conflict do nothing",
                passages,
                page_size=1000,
            )

        passages_total += len(passages)

        print(
            f"stored article: {a['title']} | "
            f"citations={len(available_citations)} documents={len(doc_rows)} passages={len(passages)}",
            flush=True,
        )

    return {
        "articles_total": len(articles),
        "articles_qualified": qualified_articles,
        "citations_available": citations_total,
        "documents": documents_total,
        "passages": passages_total,
    }