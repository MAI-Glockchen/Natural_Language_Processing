from __future__ import annotations
import os

DSN = os.getenv('POSTGRES_DSN', 'dbname=wiki user=postgres password=postgres host=localhost port=5432')
USER_AGENT = os.getenv('USER_AGENT', 'citation-pipeline/1.0')
MAX_ARTICLES = int(os.getenv('MAX_ARTICLES', '25'))
MIN_CITATIONS = int(os.getenv('MIN_CITATIONS', '20'))
MAX_CITATIONS_PER_ARTICLE = int(os.getenv('MAX_CITATIONS_PER_ARTICLE', '180'))
WORKERS = int(os.getenv('WORKERS', '32'))
HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '7'))
CHUNK_WORDS = int(os.getenv('CHUNK_WORDS', '180'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '35'))
MAX_DOC_CHARS = int(os.getenv('MAX_DOC_CHARS', '400000'))
LANG = os.getenv('WIKI_LANG', 'en')
POPULAR_URL = f'https://{LANG}.wikipedia.org/wiki/Special:MostVisitedPages?offset=0&limit={MAX_ARTICLES}'
