# -----------------------------
# Entry point to run the Wikipedia citation pipeline
# -----------------------------

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from pathlib import Path

from utils.nltk_setup import *
from pipeline import (
    extract_citations,
    fetch_document,
    clean_html,
    create_passages,
    save_passages_to_db,
    collect_valid_articles
)
from config import MAX_WORKERS, RATE_LIMIT_DELAY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def process_wikipedia_article(
    wikipedia_url: str,
    max_docs: int = 50
) -> int:
    """
    Process a single Wikipedia article through the full pipeline.

    Pipeline:
    1. Extract citations
    2. Fetch documents
    3. Clean HTML
    4. Create passages
    5. Save passages to PostgreSQL

    Args:
        wikipedia_url: URL of the Wikipedia article
        max_docs: Maximum number of citations to process

    Returns:
        int: Number of passages saved
    """
    logger.info(f"Processing article: {wikipedia_url}")

    # Step 1: Extract citations
    citations = extract_citations(wikipedia_url)
    if not citations:
        logger.warning(f"No citations found for {wikipedia_url}")
        return 0

    logger.info(f"Found {len(citations)} citations, processing up to {max_docs}")
    citations = citations[:max_docs]

    # Step 2-5: Process each citation
    total_passages = 0
    failed_count = 0

    for idx, url in enumerate(citations, 1):
        try:
            # Fetch document
            html = fetch_document(url)
            if not html:
                logger.warning(f"Failed to fetch document: {url}")
                failed_count += 1
                continue

            # Clean HTML
            clean_text = clean_html(html)
            if not clean_text:
                logger.warning(f"Failed to clean document: {url}")
                failed_count += 1
                continue

            # Create passages
            passages = create_passages(clean_text)
            if not passages:
                logger.warning(f"Failed to create passages: {url}")
                failed_count += 1
                continue

            # Save to database
            if save_passages_to_db(wikipedia_url, passages, url):
                total_passages += len(passages)
            else:
                logger.error(f"Failed to save passages for {url}")
                failed_count += 1

            # Rate limiting
            if idx < len(citations):
                time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Failed processing citation {url}: {e}")
            failed_count += 1

    logger.info(f"Finished article: {wikipedia_url} - Saved {total_passages} passages, {failed_count} failed")
    return total_passages


def process_articles_parallel(
    article_urls: List[str],
    max_docs: int = 50,
    max_workers: int = MAX_WORKERS
) -> dict:
    """
    Process multiple Wikipedia articles in parallel.

    Args:
        article_urls: List of Wikipedia article URLs
        max_docs: Maximum number of citations per article
        max_workers: Maximum number of parallel workers

    Returns:
        dict: Statistics about processing
    """
    logger.info(f"Starting parallel processing of {len(article_urls)} articles with {max_workers} workers")

    results = {
        "total_articles": len(article_urls),
        "successful": 0,
        "failed": 0,
        "total_passages": 0,
        "total_failed": 0
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(process_wikipedia_article, url, max_docs): url
            for url in article_urls
        }

        # Collect results
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                passages = future.result()
                results["successful"] += 1
                results["total_passages"] += passages
                logger.info(f"Completed: {url}")

            except Exception as e:
                results["failed"] += 1
                results["total_failed"] += 1
                logger.error(f"Failed: {url} - {e}")

    logger.info(f"Processing complete: {results}")
    return results


def collect_and_process_articles(
    target_count: int = 420,
    min_citations: int = 20,
    max_docs: int = 50,
    max_workers: int = MAX_WORKERS
) -> dict:
    """
    Main entry point: collect valid articles and process them.

    Args:
        target_count: Number of articles to collect
        min_citations: Minimum citations per article
        max_docs: Maximum citations to process per article
        max_workers: Maximum parallel workers

    Returns:
        dict: Processing statistics
    """
    logger.info(f"Collecting {target_count} valid Wikipedia articles (min {min_citations} citations)")

    # Step 1: Collect valid articles
    article_urls = collect_valid_articles(target_count=target_count, min_citations=min_citations)
    logger.info(f"Collected {len(article_urls)} valid articles")

    if not article_urls:
        logger.error("No valid articles found")
        return {"error": "No valid articles found"}

    # Step 2: Process articles in parallel
    return process_articles_parallel(
        article_urls=article_urls,
        max_docs=max_docs,
        max_workers=max_workers
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia Citation Pipeline")
    parser.add_argument(
        "--target-count",
        type=int,
        default=420,
        help="Number of articles to collect (default: 420)"
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=20,
        help="Minimum citations per article (default: 20)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50,
        help="Maximum citations to process per article (default: 50)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: from config)"
    )

    args = parser.parse_args()

    if args.workers:
        from config import MAX_WORKERS
        MAX_WORKERS = args.workers

    logger.info("Starting Wikipedia Citation Pipeline")
    results = collect_and_process_articles(
        target_count=args.target_count,
        min_citations=args.min_citations,
        max_docs=args.max_docs,
        max_workers=MAX_WORKERS
    )

    if "error" in results:
        logger.error(results["error"])
        exit(1)

    logger.info("Pipeline completed successfully")
