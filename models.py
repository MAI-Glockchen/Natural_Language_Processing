from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FetchedDoc:
    url: str
    final_url: str
    status_code: int
    content_type: str
    html: str
    fetch_ms: int
    fetch_status: str