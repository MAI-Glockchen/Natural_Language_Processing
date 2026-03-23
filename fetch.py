from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Iterable
from urllib.parse import urlparse

import requests

from config import REQUEST_TIMEOUT, USER_AGENT, WORKERS
from models import FetchedDoc


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=WORKERS,
        pool_maxsize=WORKERS,
        max_retries=0,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _fetch_one(url: str) -> FetchedDoc | None:
    try:
        with _session() as s:
            r = s.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=False,
            )
            ct = (r.headers.get("content-type") or "").lower()
            if r.status_code != 200:
                return None
            if "text/html" not in ct:
                return None

            text = r.text or ""
            if not text.strip():
                return None

            return FetchedDoc(
                url=url,
                final_url=str(r.url),
                status_code=r.status_code,
                content_type=ct,
                html=text,
                fetch_ms=0,
            )
    except Exception:
        return None


def fetch_all(urls: Iterable[str]) -> list[FetchedDoc]:
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return []

    out: list[FetchedDoc] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for doc in ex.map(_fetch_one, unique_urls):
            if doc is not None:
                out.append(doc)
    return out


def fetch_article_documents_until_threshold(
    urls: list[str],
    max_missing: int,
) -> tuple[list[FetchedDoc], int, bool]:
    """
    Returns:
        (documents, missing_count, aborted_early)

    Aborts as soon as missing_count > max_missing.
    """
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return [], 0, False

    docs: list[FetchedDoc] = []
    missing = 0
    aborted = False

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(unique_urls))) as ex:
        pending = {ex.submit(_fetch_one, url): url for url in unique_urls}

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)

            for fut in done:
                _ = pending.pop(fut)
                doc = None
                try:
                    doc = fut.result()
                except Exception:
                    doc = None

                if doc is None:
                    missing += 1
                    if missing > max_missing:
                        aborted = True
                        for pf in pending:
                            pf.cancel()
                        return docs, missing, aborted
                else:
                    docs.append(doc)

    return docs, missing, aborted


def source_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""