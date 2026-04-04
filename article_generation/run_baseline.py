from __future__ import annotations

import argparse

from .config import load_settings
from .db import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline article generation pipeline.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to operate on.",
    )
    parser.add_argument(
        "--ensure-schema-only",
        action="store_true",
        help="Create result tables and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    database = Database(settings)
    database.ensure_schema()
    print("generated_articles schema is ready")

    if args.ensure_schema_only:
        return

    print(f"split={args.split}")
    print("Baseline retrieval/generation is not wired in yet in this batch.")


if __name__ == "__main__":
    main()
