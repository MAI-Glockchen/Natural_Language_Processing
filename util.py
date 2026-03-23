from __future__ import annotations
import hashlib, re
from urllib.parse import urlparse, urlunparse

_ws = re.compile(r'\s+')
_bad = re.compile(r'^(javascript:|mailto:|tel:|#)', re.I)


def normalize_url(url: str) -> str | None:
    if not url or _bad.match(url):
        return None
    p = urlparse(url.strip())
    if p.scheme not in ('http', 'https') or not p.netloc:
        return None
    q = '&'.join(sorted(x for x in p.query.split('&') if x and not x.startswith('utm_')))
    path = p.path or '/'
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, '', q, ''))


def host(url: str) -> str:
    return urlparse(url).netloc.lower()


def squash(text: str) -> str:
    return _ws.sub(' ', text).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', 'ignore')).hexdigest()
