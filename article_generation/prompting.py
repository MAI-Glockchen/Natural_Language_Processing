from __future__ import annotations

from .retrieval import ArticleBundle, RetrievedPassage


class PromptBuilder:
    def build_baseline_prompt(self, bundle: ArticleBundle, passages: list[RetrievedPassage]) -> str:
        context_blocks = []
        for passage in passages:
            context_blocks.append(
                f"[Passage {passage.rank} | relevance={passage.score:.4f}]\n{passage.text.strip()}"
            )
        context = "\n\n".join(context_blocks)

        return (
            "Write a neutral Wikipedia-style article in plain text using only the supplied passages.\n"
            "Do not mention the passages, retrieval, scores, or any source keys.\n"
            "Do not invent facts. If a detail is uncertain or unsupported, leave it out.\n"
            "Prefer an encyclopedic tone over a review or summary tone.\n"
            "Write at least 4 paragraphs including a lead paragraph.\n"
            "The first line must be exactly: TITLE: <article title>\n"
            "Then a blank line.\n"
            "Then a line that is exactly: ARTICLE:\n"
            "Then the article body in plain text paragraphs.\n"
            "Do not output markdown headings, bullet points, or code fences.\n\n"
            f"Target title: {bundle.article_title}\n"
            f"Retrieval topic: {bundle.topic}\n\n"
            "Retrieved passages:\n"
            f"{context}"
        )
