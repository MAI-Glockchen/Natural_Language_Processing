from __future__ import annotations

import argparse
import cProfile
import pstats
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .config import Settings, get_settings
from .db import Database
from .evaluation import EvaluationService
from .generation import GenerationService
from .models import GeneratedArticleRecord
from .prompting import PromptBuilder
from .retrieval import RetrievalService
from .splits import SplitRepository


@dataclass(slots=True)
class StageTimings:
    fetch_bundle_seconds: float = 0.0
    retrieve_seconds: float = 0.0
    prompt_seconds: float = 0.0
    generate_seconds: float = 0.0
    evaluate_seconds: float = 0.0
    upsert_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return (
            self.fetch_bundle_seconds
            + self.retrieve_seconds
            + self.prompt_seconds
            + self.generate_seconds
            + self.evaluate_seconds
            + self.upsert_seconds
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline article generation")
    parser.add_argument("--ensure-schema-only", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--article-id", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignore split files and generate from all available articles in the database",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip articles already present in generated_articles for the same split/method/prompt/model/top_k",
    )
    parser.add_argument("--profile", action="store_true", help="Enable cProfile for the whole run")
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "tottime", "ncalls", "time"],
        help="Sort order for cProfile output",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=40,
        help="How many cProfile rows to print",
    )
    parser.add_argument(
        "--profile-dump",
        type=Path,
        help="Optional path to dump raw cProfile stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile:
        _run_with_cprofile(args)
    else:
        _run(args)


def _run(args: argparse.Namespace) -> None:
    run_started = perf_counter()
    settings = get_settings()
    _print_settings(settings, args)

    init_started = perf_counter()
    database = Database(settings)
    database.ensure_schema()

    if args.ensure_schema_only:
        print("generated_articles schema is ready")
        print(f"[timing] total={perf_counter() - run_started:.3f}s")
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
    init_seconds = perf_counter() - init_started

    resolve_started = perf_counter()
    effective_split = _effective_split_name(args)
    article_ids = _resolve_article_ids(args, settings, database)
    requested_count = len(article_ids)
    skipped_existing = 0
    if args.skip_existing and args.article_id is None:
        existing_ids = _existing_article_ids(database, article_ids, effective_split, settings)
        if existing_ids:
            skipped_existing = sum(1 for article_id in article_ids if article_id in existing_ids)
            article_ids = [article_id for article_id in article_ids if article_id not in existing_ids]
    resolve_seconds = perf_counter() - resolve_started

    if skipped_existing:
        print(
            f"Skipping {skipped_existing} already generated article(s); "
            f"processing {len(article_ids)} of requested {requested_count}."
        )

    if not article_ids:
        print("Nothing to do: all requested articles already exist in generated_articles.")
        total_seconds = perf_counter() - run_started
        print("\nTiming summary:")
        print(f"  init={init_seconds:.3f}s")
        print(f"  resolve_article_ids={resolve_seconds:.3f}s")
        print("  fetch_bundle_total=0.000s")
        print("  retrieve_total=0.000s")
        print("  prompt_total=0.000s")
        print("  generate_total=0.000s")
        print("  evaluate_total=0.000s")
        print("  upsert_total=0.000s")
        print(f"  grand_total={total_seconds:.3f}s")
        return

    processed = 0
    failed = 0
    cumulative = StageTimings()
    for article_id in article_ids:
        article_started = perf_counter()

        try:
            started = perf_counter()
            bundle = retrieval.fetch_article_bundle(article_id)
            fetch_bundle_seconds = perf_counter() - started

            started = perf_counter()
            passages = retrieval.retrieve_top_k(bundle, settings.top_k)
            retrieve_seconds = perf_counter() - started

            started = perf_counter()
            prompt = prompting.build_baseline_prompt(bundle, passages)
            prompt_seconds = perf_counter() - started

            started = perf_counter()
            generation_output = generation.generate(prompt, target_title=bundle.article_title)
            generate_seconds = perf_counter() - started

            generated_title = bundle.article_title

            started = perf_counter()
            metrics = evaluation.evaluate(
                generated_title=generated_title,
                generated_text=generation_output.text,
                reference_title=bundle.reference_title,
                reference_text=bundle.reference_text,
            )
            evaluate_seconds = perf_counter() - started

            record = GeneratedArticleRecord(
                article_id=bundle.article_id,
                split=effective_split,
                method="baseline",
                prompt_version=settings.prompt_version,
                model_name=settings.model_name,
                top_k=settings.top_k,
                topic=bundle.topic,
                index_file=bundle.index_file,
                generated_title=generated_title,
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

            started = perf_counter()
            database.upsert_generated_article(record)
            upsert_seconds = perf_counter() - started

            timings = StageTimings(
                fetch_bundle_seconds=fetch_bundle_seconds,
                retrieve_seconds=retrieve_seconds,
                prompt_seconds=prompt_seconds,
                generate_seconds=generate_seconds,
                evaluate_seconds=evaluate_seconds,
                upsert_seconds=upsert_seconds,
            )
            _accumulate_timings(cumulative, timings)

            processed += 1

            print(
                f"[{processed + failed}/{len(article_ids)}] "
                f"article_id={bundle.article_id} "
                f"title={bundle.article_title!r} "
                f"prompted_passages={len(passages)} "
                f"available_passages={bundle.available_passage_count} "
                f"prompt_chars={len(prompt)} "
                f"generated_chars={len(generation_output.text)} "
                f"fetch_bundle={fetch_bundle_seconds:.3f}s "
                f"retrieve={retrieve_seconds:.3f}s "
                f"prompt={prompt_seconds:.3f}s "
                f"generate={generate_seconds:.3f}s "
                f"evaluate={evaluate_seconds:.3f}s "
                f"upsert={upsert_seconds:.3f}s "
                f"article_total={perf_counter() - article_started:.3f}s "
                f"rouge1={metrics.rouge1_f1:.4f} "
                f"rouge2={metrics.rouge2_f1:.4f} "
                f"rougeL={metrics.rougel_f1:.4f} "
                f"bertscore={metrics.bertscore_f1:.4f} "
                f"title_sim={metrics.title_similarity:.4f} "
                f"sections={metrics.section_count_generated}/{metrics.section_count_reference} "
                f"section_diff={metrics.section_count_abs_diff} "
                f"len_ratio={metrics.article_length_ratio:.4f}"
            )
        except Exception as exc:
            failed += 1
            print(
                f"[{processed + failed}/{len(article_ids)}] "
                f"article_id={article_id} "
                f"status=error "
                f"article_total={perf_counter() - article_started:.3f}s "
                f"error={type(exc).__name__}: {exc}"
            )

    total_seconds = perf_counter() - run_started
    print("\nRun summary:")
    print(f"  requested={requested_count}")
    print(f"  skipped_existing={skipped_existing}")
    print(f"  attempted={len(article_ids)}")
    print(f"  succeeded={processed}")
    print(f"  failed={failed}")

    print("\nTiming summary:")
    print(f"  init={init_seconds:.3f}s")
    print(f"  resolve_article_ids={resolve_seconds:.3f}s")
    print(f"  fetch_bundle_total={cumulative.fetch_bundle_seconds:.3f}s")
    print(f"  retrieve_total={cumulative.retrieve_seconds:.3f}s")
    print(f"  prompt_total={cumulative.prompt_seconds:.3f}s")
    print(f"  generate_total={cumulative.generate_seconds:.3f}s")
    print(f"  evaluate_total={cumulative.evaluate_seconds:.3f}s")
    print(f"  upsert_total={cumulative.upsert_seconds:.3f}s")
    print(f"  grand_total={total_seconds:.3f}s")


def _run_with_cprofile(args: argparse.Namespace) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        _run(args)
    finally:
        profiler.disable()

    if args.profile_dump is not None:
        args.profile_dump.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile_dump))
        print(f"cProfile stats dumped to {args.profile_dump}")

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.profile_sort)
    print("\ncProfile summary:")
    stats.print_stats(args.profile_top)


def _accumulate_timings(total: StageTimings, current: StageTimings) -> None:
    total.fetch_bundle_seconds += current.fetch_bundle_seconds
    total.retrieve_seconds += current.retrieve_seconds
    total.prompt_seconds += current.prompt_seconds
    total.generate_seconds += current.generate_seconds
    total.evaluate_seconds += current.evaluate_seconds
    total.upsert_seconds += current.upsert_seconds


def _resolve_article_ids(args: argparse.Namespace, settings: Settings, database: Database) -> list[int]:
    if args.article_id is not None:
        return [args.article_id]
    if args.all:
        article_ids = _all_article_ids(database, args.limit)
        if not article_ids:
            raise ValueError("No article ids found in database")
        return article_ids
    repository = SplitRepository(settings.split_file)
    article_ids = repository.article_ids_for_split(args.split)
    if args.limit > 0:
        article_ids = article_ids[: args.limit]
    if not article_ids:
        raise ValueError(f"No article ids found for split={args.split!r}")
    return article_ids


def _all_article_ids(database: Database, limit: int) -> list[int]:
    sql = """
        SELECT a.article_id
        FROM articles a
        JOIN article_topic_outputs ato ON ato.article_id = a.article_id
        ORDER BY a.article_id
    """
    if limit > 0:
        sql += "\n LIMIT :limit"
        params = {"limit": limit}
    else:
        params = {}
    rows = database.fetch_all(sql, params)
    return [int(row["article_id"]) for row in rows]


def _effective_split_name(args: argparse.Namespace) -> str:
    if args.all:
        return "all"
    return args.split


def _existing_article_ids(
    database: Database,
    article_ids: list[int],
    split: str,
    settings: Settings,
) -> set[int]:
    if not article_ids:
        return set()

    rows = database.fetch_all(
        """
        SELECT article_id
        FROM generated_articles
        WHERE article_id = ANY(:article_ids)
          AND split = :split
          AND method = :method
          AND prompt_version = :prompt_version
          AND model_name = :model_name
          AND top_k = :top_k
        """,
        {
            "article_ids": article_ids,
            "split": split,
            "method": "baseline",
            "prompt_version": settings.prompt_version,
            "model_name": settings.model_name,
            "top_k": settings.top_k,
        },
    )
    return {int(row["article_id"]) for row in rows}


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
    print(f"  all={args.all}")
    print(f"  article_id={args.article_id}")
    print(f"  limit={args.limit}")
    print(f"  skip_existing={args.skip_existing}")
    print(f"  profile={args.profile}")
    print(f"  profile_sort={args.profile_sort}")
    print(f"  profile_top={args.profile_top}")
    print(f"  profile_dump={args.profile_dump}")


if __name__ == "__main__":
    main()
