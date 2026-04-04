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
            "Write a neutral encyclopedia-style article in plain text using only the supplied passages.\n"
            "Use only supported facts from the passages. Do not invent details.\n"
            "If a detail is uncertain, conflicting, or weakly supported, leave it out.\n"
            "Do not mention passages, retrieval, scores, evidence, or source metadata.\n"
            "Do not write like a review, recommendation, or box-office report.\n"
            "Write like a Wikipedia article, not like a short summary.\n"
            "Avoid repeating the same fact in different wording.\n"
            "Prefer broad coverage of the most important facts over over-expanding one aspect.\n"
            "If the passages support them, cover distinct aspects such as premise or subject, background, production or development, release, reception, performance, and other notable facts.\n"
            "If the passages do not support some of these aspects, omit them.\n"
            "Write at least 4 paragraphs including a lead paragraph.\n"
            "The lead paragraph should identify the subject clearly and summarize the most important facts.\n"
            "The article body should be in plain text paragraphs only.\n"
            "Do not output markdown headings, bullet points, tables, or code fences.\n"
            "The first line must be exactly: TITLE: <article title>\n"
            "Then a blank line.\n"
            "Then a line that is exactly: ARTICLE:\n"
            "Then the article body.\n\n"
            f"Target title: {bundle.article_title}\n"
            f"Retrieval topic: {bundle.topic}\n\n"
            "Retrieved passages:\n"
            f"{context}"
        )