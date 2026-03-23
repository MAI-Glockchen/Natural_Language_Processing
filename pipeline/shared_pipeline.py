# -----------------------------
# Shared pipeline steps for Wikipedia article generation
# -----------------------------

import logging
from typing import List, Tuple, Optional
from pathlib import Path

from pipeline.citation_extraction import extract_citations, validate_citation_url
from pipeline.document_fetching import fetch_document, cache
from pipeline.document_cleaning import clean_html
from pipeline.passage_creation import create_passages
from pipeline.db_saving import save_passages_to_db
from config import MAX_WORKERS, RATE_LIMIT_DELAY

logger = logging.getLogger(__name__)


class PipelineResult:
    """
    Container for pipeline processing results.
    """

    def __init__(self, success: bool = False, passages: List[str] = None,
                 error: str = None, metadata: dict = None):
        self.success = success
        self.passages = passages or []
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "passages": self.passages,
            "error": self.error,
            "metadata": self.metadata
        }


def process_single_citation(
    citation_url: str,
    wikipedia_url: str
) -> PipelineResult:
    """
    Process a single citation through the full pipeline.

    Args:
        citation_url: URL of the cited document
        wikipedia_url: URL of the Wikipedia article

    Returns:
        PipelineResult: Result of the processing
    """
    logger.info(f"Processing citation: {citation_url}")

    try:
        # Step 1: Fetch document
        html = fetch_document(citation_url)
        if not html:
            return PipelineResult(
                success=False,
                error=f"Failed to fetch document: {citation_url}"
            )

        # Step 2: Clean HTML
        clean_text = clean_html(html)
        if not clean_text:
            return PipelineResult(
                success=False,
                error=f"Failed to clean document: {citation_url}"
            )

        # Step 3: Create passages
        passages = create_passages(clean_text)
        if not passages:
            return PipelineResult(
                success=False,
                error=f"Failed to create passages: {citation_url}"
            )

        # Step 4: Save to database
        success = save_passages_to_db(wikipedia_url, passages, citation_url)

        if success:
            return PipelineResult(
                success=True,
                passages=passages,
                metadata={
                    "citation_url": citation_url,
                    "wikipedia_url": wikipedia_url,
                    "passage_count": len(passages)
                }
            )
        else:
            return PipelineResult(
                success=False,
                error=f"Failed to save passages for {citation_url}"
            )

    except Exception as e:
        logger.error(f"Error processing {citation_url}: {e}")
        return PipelineResult(
            success=False,
            error=str(e)
        )


def process_article_citations(
    wikipedia_url: str,
    citations: List[str],
    max_docs: int = 50,
    max_workers: int = MAX_WORKERS
) -> Tuple[List[PipelineResult], int]:
    """
    Process all citations for a Wikipedia article.

    Args:
        wikipedia_url: URL of the Wikipedia article
        citations: List of citation URLs
        max_docs: Maximum number of citations to process
        max_workers: Maximum parallel workers

    Returns:
        Tuple[List[PipelineResult], int]: Results and total passages saved
    """
    # Filter and limit citations
    valid_citations = [
        url for url in citations[:max_docs]
        if validate_citation_url(url)
    ]

    logger.info(f"Processing {len(valid_citations)} citations for {wikipedia_url}")

    results: List[PipelineResult] = []
    total_passages = 0

    with __import__('concurrent.futures').ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_citation, url, wikipedia_url): url
            for url in valid_citations
        }

        # Collect results
        for future in __import__('concurrent.futures').as_completed(futures):
            url = futures[future]
            result = future.result()
            results.append(result)

            if result.success:
                total_passages += len(result.passages)
                logger.debug(f"Completed: {url} - {len(result.passages)} passages")
            else:
                logger.warning(f"Failed: {url} - {result.error}")

    logger.info(f"Processed {len(results)} citations for {wikipedia_url} - {total_passages} passages total")
    return results, total_passages


def process_wikipedia_article(
    wikipedia_url: str,
    max_docs: int = 50
) -> PipelineResult:
    """
    Process a single Wikipedia article through the full pipeline.

    This is a convenience function that extracts citations and processes them.

    Args:
        wikipedia_url: URL of the Wikipedia article
        max_docs: Maximum number of citations to process

    Returns:
        PipelineResult: Combined result of all citation processing
    """
    logger.info(f"Processing article: {wikipedia_url}")

    # Extract citations
    citations = extract_citations(wikipedia_url)
    if not citations:
        return PipelineResult(
            success=False,
            error=f"No citations found for {wikipedia_url}"
        )

    logger.info(f"Found {len(citations)} citations")

    # Process citations
    results, total_passages = process_article_citations(
        wikipedia_url=wikipedia_url,
        citations=citations,
        max_docs=max_docs
    )

    # Aggregate results
    all_passages = []
    errors = []

    for result in results:
        if result.success:
            all_passages.extend(result.passages)
        else:
            errors.append(result.error)

    if all_passages:
        return PipelineResult(
            success=True,
            passages=all_passages,
            metadata={
                "wikipedia_url": wikipedia_url,
                "citation_count": len(citations),
                "processed_count": len(results),
                "passage_count": len(all_passages),
                "error_count": len(errors)
            }
        )
    elif errors:
        return PipelineResult(
            success=False,
            error=f"All citations failed: {errors}",
            metadata={"wikipedia_url": wikipedia_url}
        )
    else:
        return PipelineResult(
            success=False,
            error="No valid passages generated"
        )
