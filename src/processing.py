"""
DELM Processing Utilities
=========================
Batch processing, cost tracking, and other processing components.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

import pandas as pd


class BatchProcessor:
    """Handles batch processing with parallel execution."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batch(self, items: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """Process items in batches with parallel execution."""
        results = []
        
        for batch_start in range(0, len(items), self.batch_size):
            batch = items[batch_start:batch_start + self.batch_size]
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_item = {
                    executor.submit(process_func, item, **kwargs): item 
                    for item in batch
                }
                
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        print(f"Error processing item: {e}")
                        batch_results.append(None)
            
            results.extend(batch_results)
        
        return results


class CostTracker:
    """Track API costs and usage statistics."""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.error_count = 0
        
        # Cost per 1K tokens (approximate)
        self.cost_rates = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            'gemini-1.5-pro': {'input': 0.0035, 'output': 0.0105},
            'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
        }
    
    def add_request(self, model: str, input_tokens: int, output_tokens: int, success: bool = True):
        """Record a request and calculate cost."""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        self.total_tokens += input_tokens + output_tokens
        
        if model in self.cost_rates:
            rates = self.cost_rates[model]
            cost = (input_tokens * rates['input'] + output_tokens * rates['output']) / 1000
            self.total_cost += cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost and usage summary."""
        return {
            'total_requests': self.request_count,
            'successful_requests': self.request_count - self.error_count,
            'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
            'total_tokens': self.total_tokens,
            'total_cost_usd': round(self.total_cost, 4),
            'avg_cost_per_request': round(self.total_cost / self.request_count, 4) if self.request_count > 0 else 0
        }
    
    def reset(self):
        """Reset all counters."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.error_count = 0


class ResponseParser:
    """Enhanced response parsing with validation and error handling."""
    
    def __init__(self, schema_type: str = "simple"):
        self.schema_type = schema_type
    
    def parse_response(self, response: Any, paragraph: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse API response into structured DataFrame."""
        if response is None:
            return pd.DataFrame()
        
        if self.schema_type == "nested":
            return self._parse_nested_response(response, paragraph, metadata)
        else:
            return self._parse_simple_response(response, paragraph, metadata)
    
    def _parse_nested_response(self, response: Any, paragraph: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse nested response (like CommodityData with instances)."""
        if not hasattr(response, 'instances') or not response.instances:
            return pd.DataFrame()
        
        data = []
        for instance in response.instances:
            row: Dict[str, Any] = {
                'paragraph': paragraph,
                'good': getattr(instance, 'good', ''),
                'good_subtype': getattr(instance, 'good_subtype', None),
                'price_expectation': getattr(instance, 'price_expectation', False),
                'price_lower': getattr(instance, 'price_lower', None),
                'price_upper': getattr(instance, 'price_upper', None),
                'unit': getattr(instance, 'unit', None),
                'currency': getattr(instance, 'currency', None),
                'horizon': getattr(instance, 'horizon', None),
                'quote': getattr(instance, 'quote', ''),
                'confidence': getattr(instance, 'confidence', False),
            }
            
            # Add metadata if provided
            if metadata:
                row.update(metadata)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _parse_simple_response(self, response: Any, paragraph: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse simple response (current DELM format)."""
        if isinstance(response, dict):
            row: Dict[str, Any] = {'paragraph': paragraph}
            row.update(response)
            
            if metadata:
                row.update(metadata)
            
            return pd.DataFrame([row])
        
        return pd.DataFrame()


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
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"All {self.max_retries + 1} attempts failed. Last error: {e}")
        
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unknown error occurred during retry") 