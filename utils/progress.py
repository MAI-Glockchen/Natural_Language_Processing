# -----------------------------
# Progress tracking utilities
# -----------------------------

import logging
import time
from typing import Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks progress of a processing pipeline.
    """

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of what's being processed
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.completed = 0
        self.failed = 0
        self.successful = 0

    def update(self, success: bool = True) -> None:
        """
        Update progress.

        Args:
            success: Whether the current item was processed successfully
        """
        self.current += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1

    def _elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def _eta(self) -> Optional[float]:
        """Estimate time remaining in seconds."""
        if self.current == 0:
            return None
        elapsed = self._elapsed()
        rate = self.current / elapsed
        remaining = self.total - self.current
        if rate == 0:
            return None
        return remaining / rate

    def _progress(self) -> float:
        """Get progress as percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    def log(self) -> None:
        """Log current progress."""
        eta = self._eta()
        eta_str = f" ({eta:.1f}s remaining)" if eta else ""
        logger.info(
            f"[{self.description}] {self.current}/{self.total} "
            f"({self._progress():.1f}%){eta_str}"
        )

    def __call__(self, success: bool = True) -> None:
        """Update progress and log."""
        self.update(success)
        if self.current % 10 == 0 or self.current == self.total:
            self.log()


def process_with_progress(
    items: Any,
    processor: Callable[[Any], Any],
    max_workers: int = 4,
    description: str = "Processing"
) -> list:
    """
    Process items with progress tracking.

    Args:
        items: Iterable of items to process
        processor: Function to apply to each item
        max_workers: Maximum number of parallel workers
        description: Description for progress logging

    Returns:
        List of processed results
    """
    total = len(items)
    tracker = ProgressTracker(total, description)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(processor, item): item
            for item in items
        }

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
                tracker.update(success=True)
            except Exception as e:
                logger.error(f"Failed processing {item}: {e}")
                tracker.update(success=False)

    return results

