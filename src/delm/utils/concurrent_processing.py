"""
DELM Concurrent Processing
=========================
Run an I/O-bound callable over a collection using threads.

Tailored for DELM's LLM extraction workload:
* Uses ThreadPoolExecutor exclusively (no process backend switch).
* Preserves input order.
* Propagates the first worker exception after all tasks finish.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Sequence, Union, Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")  # input  type
R = TypeVar("R")  # output type


class ConcurrentProcessor:
    """Thin wrapper over ThreadPoolExecutor.

<<<<<<< HEAD
    Args:
        max_workers: Number of threads. ``None`` (or <= 0) picks a heuristic
            default ``min(32, os.cpu_count() + 4)``. A value of 1 forces
            sequential execution.
=======
    Parameters
    ----------
    max_workers : Optional[int], optional
        Number of threads. ``None`` (or <= 0) picks a heuristic default
        ``min(32, os.cpu_count() + 4)``. A value of 1 forces sequential mode.
>>>>>>> ad04d3dddfe7e9c168c2221c5933c22d45bd42d1
    """

    def __init__(self, *, max_workers: Optional[int] = None) -> None:
        if max_workers is None or max_workers <= 0:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        self.max_workers: int = max_workers
        log.debug(
            "ConcurrentProcessor initialised with ThreadPoolExecutor, max_workers=%d",
            self.max_workers,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_concurrently(
        self,
        items: Sequence[T],
        fn: Callable[[T], R],
    ) -> List[R]:
        """Apply ``fn`` to each element of ``items`` (optionally) in parallel.

        Results are returned in the same order as ``items``.

        Args:
            items: The items to process.
            fn: The function to apply to each item.

        Returns:
            A list of results corresponding to each input item.

        Raises:
            Exception: If a worker raises, the first exception is re‑raised after all futures complete.
        """
        if not items:
            log.debug("No items to process, returning empty list")
            return []

        log.debug(
            "Starting concurrent processing of %d items with max_workers: %d",
            len(items),
            self.max_workers,
        )

        # Sequential fallback
        if self.max_workers <= 1:
            log.debug("max_workers <= 1; running sequentially")
            results = []
            for i, item in enumerate(items):
                try:
                    log.debug("Processing item %d/%d sequentially", i + 1, len(items))
                    result = fn(item)
                    results.append(result)
                    log.debug("Item %d/%d processed successfully", i + 1, len(items))
                except Exception as e:
                    log.error(
                        "Error processing item %d/%d: %s",
                        i + 1,
                        len(items),
                        e,
                        exc_info=True,
                    )
                    raise
                    # This should never happen as the function is expected to be error safe, but just in case.
            log.debug("Sequential processing completed: %d results", len(results))
            return results

        first_exc: Optional[BaseException] = None
        results: List[R] = [None] * len(items)  # type: ignore[assignment]

        try:
            log.debug("Creating ThreadPoolExecutor with %d workers", self.max_workers)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                log.debug("Submitting %d tasks to executor", len(items))
                future_to_idx = {
                    executor.submit(fn, item): idx for idx, item in enumerate(items)
                }

                log.debug("Processing %d futures as they complete", len(future_to_idx))
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        log.debug(
                            "Collecting result for item %d/%d", idx + 1, len(items)
                        )
                        results[idx] = future.result()
                        log.debug(
                            "Item %d/%d processed successfully", idx + 1, len(items)
                        )
                    except BaseException as exc:  # noqa: BLE001
                        log.error(
                            "Worker raised an exception on item %d/%d: %s",
                            idx + 1,
                            len(items),
                            exc,
                            exc_info=True,
                        )
                        if first_exc is None:
                            first_exc = exc
        except KeyboardInterrupt:
            log.warning("Parallel processing interrupted by user; aborting")
            raise

        log.debug("Concurrent processing completed: %d results", len(results))

        if first_exc is not None:
            log.error(
                "Raising first exception encountered during processing: %s", first_exc
            )
            raise first_exc

        return results
