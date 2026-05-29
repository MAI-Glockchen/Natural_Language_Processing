from __future__ import annotations

import csv
from pathlib import Path


class SplitRepository:
    def __init__(self, split_file: Path) -> None:
        self._split_file = split_file

    def article_ids_for_split(self, split_name: str) -> list[int]:
        if not self._split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self._split_file}. Create a CSV with columns article_id,split."
            )
        result: list[int] = []
        with self._split_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if (row.get("split") or "").strip().lower() != split_name.lower():
                    continue
                article_id = (row.get("article_id") or "").strip()
                if article_id:
                    result.append(int(article_id))
        return result
