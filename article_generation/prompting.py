from __future__ import annotations

from .retrieval import ArticleBundle, RetrievedPassage


class PromptBuilder:
    def build_baseline_prompt(self, bundle: ArticleBundle, passages: list[RetrievedPassage]) -> str:
        context_blocks = []
        for passage in passages:
            context_blocks.append(
                f"[Passage {passage.rank} | key={passage.passage_key} | score={passage.score:.4f}]\n{passage.text.strip()}"
            )
        context = "\n\n".join(context_blocks)

        return (
            "Write a Wikipedia-style article in plain text using only the supplied passages.\n"
            "Do not mention the passages, retrieval, or source keys.\n"
            "Be factual, neutral, and concise.\n"
            "If details are uncertain or conflicting, prefer cautious wording.\n"
            "Start with the article title alone on the first line, then a blank line, then the article body.\n"
            "Do not output markdown fences.\n\n"
            f"Target title: {bundle.article_title}\n"
            f"Topic query used for retrieval: {bundle.topic}\n\n"
            "Retrieved passages:\n"
            f"{context}"
        )
