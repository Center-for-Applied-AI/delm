"""
DELM Retry Handler
==================
Retry handling with exponential backoff for robust API calls.
"""

import logging
import time
from typing import Any, Callable
import traceback

# Module-level logger
log = logging.getLogger(__name__)


class RetryHandler:
    """Handle retries with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def execute_with_retry(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: The function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function execution.

        Raises:
            Exception: The last exception from the function execution if all attempts fail.
        """
        exceptions = []
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                log.debug("Executing function with retry: %s", func.__name__)
                result = func(*args, **kwargs)
                end_time = time.time()
                log.debug("Function execution completed in %.3fs", end_time - start_time)
                return result
            except Exception as e:
                exceptions.append(e)
                log.warning("Exception on attempt %d: %s", attempt + 1, e)
                if log.isEnabledFor(logging.DEBUG):
                    traceback.print_exc()
                    
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    log.info("Attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    log.error("All %d attempts failed. Last error: %s", self.max_retries + 1, e)
        raise exceptions[-1]