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

        target_words = self._target_word_count(bundle.available_passage_count, len(passages))
        target_paragraphs = self._target_paragraph_count(bundle.available_passage_count, len(passages))

        return (
            "Write a neutral encyclopedia-style article in plain text using only the supplied passages.\n"
            "Use only supported facts from the passages. Do not invent details.\n"
            "If a detail is uncertain, conflicting, or weakly supported, leave it out.\n"
            "Do not mention passages, retrieval, scores, evidence, or source metadata.\n"
            "Do not write commentary, advice, analysis notes, a Q&A answer, or a summary about the evidence.\n"
            "Do not address the reader.\n"
            "Do not write like a review, recommendation, or box-office report.\n"
            "Do not output tables, lists, charts, headings, markdown, or code fences.\n"
            "Write continuous prose paragraphs only.\n"
            "Prefer broad coverage of the most important facts over over-expanding one aspect.\n"
            "Avoid repeating the same fact in different wording.\n"
            "Higher relevance passages matter more than lower relevance passages.\n"
            "Use lower-scoring passages only for extra detail, not to override stronger evidence.\n"
            "If supported by the passages, cover subject, background, production or development, release, reception, and performance.\n"
            "If an aspect is not supported, omit it.\n"
            f"This article has {bundle.available_passage_count} indexed passages in total.\n"
            f"The prompt contains {len(passages)} selected passages.\n"
            f"Aim for about {target_words} words and about {target_paragraphs} paragraphs.\n"
            "The title must be exactly the target title.\n"
            "The first line must be exactly: TITLE: <article title>\n"
            "Then a blank line.\n"
            "Then a line that is exactly: ARTICLE:\n"
            "Then the article body in plain prose paragraphs.\n"
            "If you cannot follow this exact format, output nothing.\n\n"
            f"Target title: {bundle.article_title}\n"
            f"Retrieval topic: {bundle.topic}\n\n"
            "Retrieved passages:\n"
            f"{context}"
        )

    @staticmethod
    def _target_word_count(available_passage_count: int, prompted_passage_count: int) -> int:
        effective = min(available_passage_count, max(prompted_passage_count, 1) * 2)
        return max(350, min(effective * 80, 6000))

    @staticmethod
    def _target_paragraph_count(available_passage_count: int, prompted_passage_count: int) -> int:
        effective = min(available_passage_count, max(prompted_passage_count, 1) * 2)
        return max(4, min(effective // 6, 10))