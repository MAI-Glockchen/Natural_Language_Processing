from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class GenerationOutput:
    title: str
    text: str
    raw_response: str


class GenerationService:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> GenerationOutput:
        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write neutral encyclopedia-style articles from retrieved evidence. "
                        "Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(f"{self._base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

        content = str(data["choices"][0]["message"]["content"]).strip()
        title, text = _split_title_and_text(content)
        return GenerationOutput(title=title, text=text, raw_response=content)


def _split_title_and_text(content: str) -> tuple[str, str]:
    stripped = content.strip()
    if not stripped:
        return "", ""

    title = ""
    body = stripped

    for line in stripped.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.upper().startswith("TITLE:"):
            title = clean.split(":", 1)[1].strip()
            break

    marker = "ARTICLE:"
    marker_index = stripped.find(marker)
    if marker_index >= 0:
        body = stripped[marker_index + len(marker):].strip()
    else:
        nonempty = [line.strip() for line in stripped.splitlines() if line.strip()]
        if nonempty:
            if not title:
                title = nonempty[0].lstrip("#").strip()
            body_lines = nonempty[1:] if len(nonempty) > 1 else nonempty
            body = "\n".join(body_lines).strip()

    if not title:
        nonempty = [line.strip() for line in stripped.splitlines() if line.strip()]
        if nonempty:
            title = nonempty[0].replace("TITLE:", "", 1).strip()

    return title, body
