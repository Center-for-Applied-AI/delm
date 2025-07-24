"""
DELM Concurrent Processing
=========================
Asynchronous concurrent processing utilities for handling I/O-bound operations.
"""

import asyncio
from typing import Any, Callable, List

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
        self.max_workers = max_workers
    
    def process_concurrently(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Process items concurrently with asynchronous execution."""
        try:
            loop = asyncio.get_running_loop()
            return self._process_concurrently_sync(items, process_func, **kwargs)
        except RuntimeError:
            return asyncio.run(self._process_concurrently_async(items, process_func, **kwargs))
    
    def _process_concurrently_sync(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Synchronous fallback for when we're already in an async context."""
        # For now, process sequentially to avoid complexity
        # In a future version, we could implement proper async-aware concurrent processing
        results = []
        for item in items:
            try:
                result = process_func(item, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but continue processing other items
                # TODO: How should we handle errors here?
                print(f"Error processing item: {e}")
                results.append(None)
        return results
    
    async def _process_concurrently_async(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Internal async method to process items."""
        semaphore = asyncio.Semaphore(self.max_workers)
        results = [None] * len(items)  # Pre-allocate results list
        
        async def _process_with_semaphore(item, index):
            async with semaphore:
                try:
                    # Check if the function is async
                    if asyncio.iscoroutinefunction(process_func):
                        result = await process_func(item, **kwargs)
                    else:
                        # Run sync function in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, process_func, item, **kwargs)
                    
                    results[index] = result
                    return result
                except Exception as e:
                    # Log error but continue processing other items
                    # TODO: How should we handle errors here?
                    print(f"Error processing item {index}: {e}")
                    results[index] = None
                    return None
        
        # Create tasks for all items
        tasks = [_process_with_semaphore(item, i) for i, item in enumerate(items)]
        
        # Process all items concurrently without progress bar
        # Progress bar will be handled at a higher level
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results 