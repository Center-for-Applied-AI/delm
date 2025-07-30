"""
DELM Concurrent Processing
=========================
Asynchronous concurrent processing utilities for handling I/O-bound operations.
"""

import logging
import asyncio
from typing import Any, Callable, List

# Module-level logger
log = logging.getLogger(__name__)

class ConcurrentProcessor:
    """
    Handles concurrent processing with asynchronous execution.
    
    We use asyncio instead of ThreadPoolExecutor because our primary use case is
    making API calls to LLM services, which are I/O-bound operations. asyncio is
    more efficient for I/O-bound tasks as it can handle many concurrent operations
    with minimal overhead compared to threads. It also provides better control
    over concurrency limits and more graceful error handling for network requests.
    """
    
    def __init__(self, max_workers: int = 4):
        log.debug("Initializing ConcurrentProcessor with max_workers: %d", max_workers)
        self.max_workers = max_workers
    
    def process_concurrently(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Process items concurrently with asynchronous execution."""
        log.debug("Starting concurrent processing of %d items with max_workers: %d", len(items), self.max_workers)
        
        try:
            loop = asyncio.get_running_loop()
            log.debug("Already in async context, using synchronous fallback")
            return self._process_concurrently_sync(items, process_func, **kwargs)
        except RuntimeError:
            log.debug("No running event loop, creating new async context")
            return asyncio.run(self._process_concurrently_async(items, process_func, **kwargs))
    
    def _process_concurrently_sync(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Synchronous fallback for when we're already in an async context."""
        log.debug("Using synchronous fallback for %d items", len(items))
        # For now, process sequentially to avoid complexity
        # In a future version, we could implement proper async-aware concurrent processing
        results = []
        for i, item in enumerate(items):
            try:
                log.debug("Processing item %d/%d", i + 1, len(items))
                result = process_func(item, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but continue processing other items
                log.error("Error processing item %d: %s", i, e)
                results.append(None)
        
        log.debug("Synchronous processing completed: %d results", len(results))
        return results
    
    async def _process_concurrently_async(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Internal async method to process items."""
        log.debug("Starting async concurrent processing with semaphore limit: %d", self.max_workers)
        semaphore = asyncio.Semaphore(self.max_workers)
        results = [None] * len(items)  # Pre-allocate results list
        
        async def _process_with_semaphore(item, index):
            async with semaphore:
                try:
                    log.debug("Processing async item %d/%d", index + 1, len(items))
                    # Check if the function is async
                    if asyncio.iscoroutinefunction(process_func):
                        log.debug("Using async function for item %d", index)
                        result = await process_func(item, **kwargs)
                    else:
                        # Run sync function in thread pool to avoid blocking
                        log.debug("Using sync function in thread pool for item %d", index)
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, process_func, item, **kwargs)
                    
                    results[index] = result
                    log.debug("Item %d processed successfully", index)
                    return result
                except Exception as e:
                    # Log error but continue processing other items
                    # TODO: How should we handle errors here?
                    log.error("Error processing async item %d: %s", index, e)
                    results[index] = None
                    return None
        
        # Create tasks for all items
        log.debug("Creating %d async tasks", len(items))
        tasks = [_process_with_semaphore(item, i) for i, item in enumerate(items)]
        
        # Process all items concurrently without progress bar
        # Progress bar will be handled at a higher level
        log.debug("Starting concurrent execution of %d tasks", len(tasks))
        await asyncio.gather(*tasks, return_exceptions=True)
        
        log.debug("Async concurrent processing completed: %d results", len(results))
        return results 