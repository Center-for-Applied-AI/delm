"""
DELM Retry Handler
==================
Retry handling with exponential backoff for robust API calls.
"""

import time
from typing import Any, Callable
import traceback

from delm.exceptions import APIError


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
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                print(f"[RETRY HANDLER] Exception on attempt {attempt + 1}:")
                traceback.print_exc()
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"All {self.max_retries + 1} attempts failed. Last error: {e}")
        
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