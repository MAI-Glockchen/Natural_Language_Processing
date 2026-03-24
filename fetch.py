from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import perf_counter
from typing import Iterable
from urllib.parse import urlparse

import requests

from config import REQUEST_TIMEOUT, USER_AGENT, WORKERS
from models import FetchedDoc


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=WORKERS,
        pool_maxsize=WORKERS,
        max_retries=0,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _status_from_http(status_code: int) -> str:
    if status_code == 403:
        return "http_403"
    if status_code == 404:
        return "http_404"
    if status_code == 429:
        return "http_429"
    return f"http_{status_code}"


def _fetch_one(url: str) -> FetchedDoc:
    started = perf_counter()
    try:
        with _session() as s:
            r = s.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=False,
            )
            fetch_ms = int((perf_counter() - started) * 1000)
            ct = (r.headers.get("content-type") or "").lower()

            if r.status_code != 200:
                return FetchedDoc(
                    url=url,
                    final_url=str(r.url),
                    status_code=r.status_code,
                    content_type=ct,
                    html="",
                    fetch_ms=fetch_ms,
                    fetch_status=_status_from_http(r.status_code),
                )

            if "text/html" not in ct:
                return FetchedDoc(
                    url=url,
                    final_url=str(r.url),
                    status_code=r.status_code,
                    content_type=ct,
                    html="",
                    fetch_ms=fetch_ms,
                    fetch_status="non_html",
                )

            text = r.text or ""
            if not text.strip():
                return FetchedDoc(
                    url=url,
                    final_url=str(r.url),
                    status_code=r.status_code,
                    content_type=ct,
                    html="",
                    fetch_ms=fetch_ms,
                    fetch_status="empty",
                )

            return FetchedDoc(
                url=url,
                final_url=str(r.url),
                status_code=r.status_code,
                content_type=ct,
                html=text,
                fetch_ms=fetch_ms,
                fetch_status="ok",
            )
    except requests.Timeout:
        fetch_ms = int((perf_counter() - started) * 1000)
        return FetchedDoc(
            url=url,
            final_url=url,
            status_code=0,
            content_type="",
            html="",
            fetch_ms=fetch_ms,
            fetch_status="timeout",
        )
    except Exception:
        fetch_ms = int((perf_counter() - started) * 1000)
        return FetchedDoc(
            url=url,
            final_url=url,
            status_code=0,
            content_type="",
            html="",
            fetch_ms=fetch_ms,
            fetch_status="exception",
        )


def fetch_all(urls: Iterable[str]) -> list[FetchedDoc]:
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return []

    out: list[FetchedDoc] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for doc in ex.map(_fetch_one, unique_urls):
            out.append(doc)
    return out


def fetch_article_documents_until_threshold(
    urls: list[str],
    max_missing: int,
) -> tuple[list[FetchedDoc], int, bool, dict[str, int]]:
    """
    Returns:
        (documents_ok, missing_count, aborted_early, status_counts)

    "missing" means fetch_status != "ok".
    """
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return [], 0, False, {}

    docs: list[FetchedDoc] = []
    missing = 0
    aborted = False
    status_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(unique_urls))) as ex:
        pending = {ex.submit(_fetch_one, url): url for url in unique_urls}

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)

            for fut in done:
                pending.pop(fut, None)

                try:
                    doc = fut.result()
                except Exception:
                    doc = FetchedDoc(
                        url="",
                        final_url="",
                        status_code=0,
                        content_type="",
                        html="",
                        fetch_ms=0,
                        fetch_status="exception",
                    )

                status_counts[doc.fetch_status] = status_counts.get(doc.fetch_status, 0) + 1

                if doc.fetch_status == "ok":
                    docs.append(doc)
                    continue

                missing += 1
                if missing > max_missing:
                    aborted = True
                    for pf in pending:
                        pf.cancel()
                    return docs, missing, aborted, status_counts

    return docs, missing, aborted, status_counts


def source_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""