# Utilities

Supporting modules that keep DELM reliable and observable.

## Concurrency and Retry

::: delm.utils.concurrent_processing.ConcurrentProcessor
    options:
      show_source: false

::: delm.utils.retry_handler.RetryHandler
    options:
      show_source: false

## Cost Tracking

::: delm.utils.cost_tracker.CostTracker
    options:
      show_source: false

::: delm.utils.cost_estimation
    options:
      show_source: false
      members:
        - estimate_input_token_cost
        - estimate_total_cost

## Semantic Cache

::: delm.utils.semantic_cache.SemanticCache
    options:
      show_source: false

::: delm.utils.semantic_cache.FilesystemJSONCache
    options:
      show_source: false

::: delm.utils.semantic_cache.SQLiteWALCache
    options:
      show_source: false

::: delm.utils.semantic_cache.LMDBCache
    options:
      show_source: false

::: delm.utils.semantic_cache.SemanticCacheFactory
    options:
      show_source: false
