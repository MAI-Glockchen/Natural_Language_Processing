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
                    "content": "You write neutral encyclopedia-style articles from retrieved evidence.",
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
    lines = [line.rstrip() for line in content.splitlines()]
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return "", ""
    title = nonempty[0].strip().lstrip("#").strip()
    if len(nonempty) == 1:
        return title, title
    body_start = lines.index(nonempty[1])
    body = "\n".join(lines[body_start:]).strip()
    return title, body
