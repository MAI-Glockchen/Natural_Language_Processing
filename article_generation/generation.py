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

    def generate(self, prompt: str, target_title: str) -> GenerationOutput:
        content = self._chat(
            system_prompt=(
                "You write neutral encyclopedia-style articles from retrieved evidence. "
                "You must obey the requested output format exactly. "
                "Output only TITLE/ARTICLE. "
                "Do not write tables, charts, lists, Q&A, or meta-commentary. "
                "Do not invent unrelated topics."
            ),
            user_prompt=prompt,
            temperature=self._temperature,
        ).strip()

        if not _looks_like_structured_output(content) or _is_bad_article_body(content, target_title):
            repair_prompt = (
                f"Rewrite the following into a neutral encyclopedia-style prose article about {target_title}.\n"
                f"The title must be exactly: {target_title}\n"
                "Output only this format:\n"
                f"TITLE: {target_title}\n\n"
                "ARTICLE:\n"
                "<plain prose article body>\n\n"
                "Do not output tables, lists, markdown, commentary, Q&A, or instructions.\n"
                "Use only the article content that is actually about the target title.\n\n"
                "CONTENT TO REWRITE:\n"
                f"{content}"
            )
            content = self._chat(
                system_prompt=(
                    "You repair malformed outputs into plain prose encyclopedia articles. "
                    "Output only repaired TITLE/ARTICLE text."
                ),
                user_prompt=repair_prompt,
                temperature=0.0,
            ).strip()

        title, text = _split_title_and_text(content)
        if not text or _is_bad_article_body(content, target_title):
            raise ValueError("Model output did not contain a valid article body")

        return GenerationOutput(title=title, text=text, raw_response=content)

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(f"{self._base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])


def _looks_like_structured_output(content: str) -> bool:
    stripped = content.lstrip()
    return stripped.startswith("TITLE:") and "\nARTICLE:\n" in stripped


def _is_bad_article_body(content: str, target_title: str) -> bool:
    lowered = content.lower()
    title_lower = target_title.lower()

    bad_markers = (
        "document:",
        "question:",
        "comparison table",
        "relevance score",
        "|---",
        "here's a breakdown",
        "here is a breakdown",
        "based on your understanding",
    )
    if any(marker in lowered for marker in bad_markers):
        return True

    _, body = _split_title_and_text(content)
    if not body:
        return True

    body_lower = body.lower()
    if title_lower not in body_lower[:800]:
        return True

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if sum(1 for line in lines if "|" in line) >= 3:
        return True

    return False


def _split_title_and_text(content: str) -> tuple[str, str]:
    stripped = content.strip()
    if stripped.startswith("TITLE:"):
        lines = stripped.splitlines()
        first_line = lines[0].strip()
        title = first_line[len("TITLE:") :].strip()
        marker = "\nARTICLE:\n"
        if marker in stripped:
            body = stripped.split(marker, 1)[1].strip()
            return title, body

    lines = [line.rstrip() for line in content.splitlines()]
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return "", ""

    first_line = nonempty[0].strip()
    if first_line.startswith("TITLE:"):
        title = first_line[len("TITLE:") :].strip()
    else:
        title = first_line.lstrip("#").strip()

    if len(nonempty) == 1:
        return title, title

    body_start = lines.index(nonempty[1])
    body = "\n".join(lines[body_start:]).strip()
    return title, body