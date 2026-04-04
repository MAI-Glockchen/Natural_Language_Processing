from __future__ import annotations

import argparse

from .config import Settings, get_settings
from .db import Database
from .evaluation import EvaluationService
from .generation import GenerationService
from .models import GeneratedArticleRecord
from .prompting import PromptBuilder
from .retrieval import RetrievalService
from .splits import SplitRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline article generation")
    parser.add_argument("--ensure-schema-only", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--article-id", type=int)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    _print_settings(settings, args)

    database = Database(settings)
    database.ensure_schema()

    if args.ensure_schema_only:
        print("generated_articles schema is ready")
        return

    retrieval = RetrievalService(
        database=database,
        index_dir=settings.index_dir,
        embedding_model_name=settings.embedding_model_name,
        embed_device=settings.embedding_device,
        normalize_embeddings=settings.normalize_embeddings,
    )
    evaluation = EvaluationService(
        title_embedding_model_name=settings.title_embedding_model_name,
        embed_device=settings.embedding_device,
    )
    generation = GenerationService(
        base_url=settings.llm_base_url,
        model_name=settings.model_name,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
    )
    prompting = PromptBuilder()

    article_ids = _resolve_article_ids(args, settings)
    processed = 0
    for article_id in article_ids:
        bundle = retrieval.fetch_article_bundle(article_id)
        passages = retrieval.retrieve_top_k(bundle, settings.top_k)
        prompt = prompting.build_baseline_prompt(bundle, passages)
        generation_output = generation.generate(prompt)
        metrics = evaluation.evaluate(
            generated_title=generation_output.title or bundle.article_title,
            generated_text=generation_output.text,
            reference_title=bundle.reference_title,
            reference_text=bundle.reference_text,
        )
        record = GeneratedArticleRecord(
            article_id=bundle.article_id,
            split=args.split,
            method="baseline",
            prompt_version=settings.prompt_version,
            model_name=settings.model_name,
            top_k=settings.top_k,
            topic=bundle.topic,
            index_file=bundle.index_file,
            generated_title=generation_output.title or bundle.article_title,
            generated_text=generation_output.text,
            reference_title=bundle.reference_title,
            reference_text=bundle.reference_text,
            rouge1_f1=metrics.rouge1_f1,
            rouge2_f1=metrics.rouge2_f1,
            rougel_f1=metrics.rougel_f1,
            bertscore_f1=metrics.bertscore_f1,
            title_similarity=metrics.title_similarity,
            section_count_generated=metrics.section_count_generated,
            section_count_reference=metrics.section_count_reference,
            section_count_abs_diff=metrics.section_count_abs_diff,
            article_length_ratio=metrics.article_length_ratio,
        )
        database.upsert_generated_article(record)
        processed += 1
        print(
            f"[{processed}/{len(article_ids)}] article_id={bundle.article_id} title={bundle.article_title!r} "
            f"rougeL={metrics.rougel_f1:.4f} bertscore={metrics.bertscore_f1:.4f}"
        )


def _resolve_article_ids(args: argparse.Namespace, settings: Settings) -> list[int]:
    if args.article_id is not None:
        return [args.article_id]
    repository = SplitRepository(settings.split_file)
    article_ids = repository.article_ids_for_split(args.split)
    if args.limit > 0:
        article_ids = article_ids[: args.limit]
    if not article_ids:
        raise ValueError(f"No article ids found for split={args.split!r}")
    return article_ids


def _print_settings(settings: Settings, args: argparse.Namespace) -> None:
    print("Using article generation settings:")
    print(f"  database_url={settings.database_url}")
    print(f"  split_file={settings.split_file}")
    print(f"  index_dir={settings.index_dir}")
    print(f"  embedding_model_name={settings.embedding_model_name}")
    print(f"  title_embedding_model_name={settings.title_embedding_model_name}")
    print(f"  embedding_device={settings.embedding_device}")
    print(f"  normalize_embeddings={settings.normalize_embeddings}")
    print(f"  llm_base_url={settings.llm_base_url}")
    print(f"  model_name={settings.model_name}")
    print(f"  llm_timeout_seconds={settings.llm_timeout_seconds}")
    print(f"  generation_temperature={settings.generation_temperature}")
    print(f"  generation_max_tokens={settings.generation_max_tokens}")
    print(f"  top_k={settings.top_k}")
    print(f"  prompt_version={settings.prompt_version}")
    print("Run arguments:")
    print(f"  ensure_schema_only={args.ensure_schema_only}")
    print(f"  split={args.split}")
    print(f"  article_id={args.article_id}")
    print(f"  limit={args.limit}")


if __name__ == "__main__":
    main()