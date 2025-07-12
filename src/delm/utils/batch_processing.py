"""
DELM Batch Processing
====================
Asynchronous batch processing utilities for handling concurrent operations.
"""

import asyncio
from typing import Any, Callable, List
from tqdm.auto import tqdm

from ..exceptions import ProcessingError


class BatchProcessor:
    """
    Handles batch processing with asynchronous execution.
    
    We use asyncio instead of ThreadPoolExecutor because our primary use case is
    making API calls to LLM services, which are I/O-bound operations. asyncio is
    more efficient for I/O-bound tasks as it can handle many concurrent operations
    with minimal overhead compared to threads. It also provides better control
    over concurrency limits and more graceful error handling for network requests.
    """
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batch(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Process items in batches with asynchronous execution."""
        try:
            loop = asyncio.get_running_loop()
            return self._process_batch_sync(items, process_func, **kwargs)
        except RuntimeError:
            return asyncio.run(self._process_batch_async(items, process_func, **kwargs))
    
    def _process_batch_sync(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Synchronous fallback for when we're already in an async context."""
        # For now, process sequentially to avoid complexity
        # In a future version, we could implement proper async-aware batching
        results = []
        for item in items:
            try:
                result = process_func(item, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but continue processing other items
                print(f"Error processing item: {e}")
                results.append(None)
        return results
    
    async def _process_batch_async(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Internal async method to process items."""
        semaphore = asyncio.Semaphore(self.max_workers)
        results = []
        
        async def _process_with_semaphore(item):
            async with semaphore:
                try:
                    # Check if the function is async
                    if asyncio.iscoroutinefunction(process_func):
                        return await process_func(item, **kwargs)
                    else:
                        # Run sync function in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, process_func, item, **kwargs)
                except Exception as e:
                    # Log error but continue processing other items
                    print(f"Error processing item: {e}")
                    return None
        
        # Process items in batches
        for batch_start in tqdm(range(0, len(items), self.batch_size), desc="Processing batches"):
            batch = items[batch_start:batch_start + self.batch_size]
            batch_tasks = [_process_with_semaphore(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Exception in batch processing: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            results.extend(processed_results)
        
        return results 