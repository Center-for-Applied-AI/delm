"""
DELM Retry Handler
==================
Retry handling with exponential backoff for robust API calls.
"""

import logging
import time
from typing import Any, Callable
import traceback

from delm.exceptions import APIError

# Module-level logger
log = logging.getLogger(__name__)


class RetryHandler:
    """Handle retries with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                log.debug("Executing function with retry: %s", func.__name__)
                result = func(*args, **kwargs)
                end_time = time.time()
                log.debug("Function execution completed in %.3fs", end_time - start_time)
                return result
            except Exception as e:
                last_exception = e
                log.warning("Exception on attempt %d: %s", attempt + 1, e)
                if log.isEnabledFor(logging.DEBUG):
                    traceback.print_exc()
                    
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    log.info("Attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    log.error("All %d attempts failed. Last error: %s", self.max_retries + 1, e)
        
        if last_exception:
            raise APIError(
                f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}",
                {
                    "max_retries": self.max_retries,
                    "base_delay": self.base_delay,
                    "last_exception": str(last_exception)
                }
            ) from last_exception
        else:
            raise APIError("Unknown error occurred during retry", {"max_retries": self.max_retries}) 