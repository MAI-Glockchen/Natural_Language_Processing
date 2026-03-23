from __future__ import annotations
import time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import WORKERS, HTTP_TIMEOUT, USER_AGENT
from models import FetchedDoc


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=2, connect=2, read=2, backoff_factor=0.2, status_forcelist=(429, 500, 502, 503, 504))
    a = HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS, max_retries=r)
    s.mount('http://', a); s.mount('https://', a)
    s.headers['User-Agent'] = USER_AGENT
    return s


def _fetch_one(s: requests.Session, url: str) -> FetchedDoc | None:
    t0 = time.perf_counter()
    try:
        r = s.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, stream=False)
        ctype = (r.headers.get('content-type') or '').lower()
        if r.status_code != 200 or 'text/html' not in ctype:
            return None
        return FetchedDoc(url, r.url, r.status_code, ctype[:120], r.text, int((time.perf_counter() - t0) * 1000))
    except Exception:
        return None


def fetch_all(urls: list[str]) -> list[FetchedDoc]:
    s = _session()
    out: list[FetchedDoc] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_fetch_one, s, u): u for u in urls}
        for f in as_completed(futs):
            doc = f.result()
            if doc:
                out.append(doc)
    return out
