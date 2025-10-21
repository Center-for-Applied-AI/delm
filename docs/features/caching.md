# Semantic Caching

Learn how to use DELM's semantic caching to reduce costs and improve performance by avoiding duplicate API calls.

## What is Semantic Caching?

Semantic caching stores the results of LLM API calls based on the semantic similarity of input text. When you process text that's similar to previously processed text, DELM can return cached results instead of making new API calls.

### Benefits

- **Cost reduction**: Avoid paying for duplicate or similar API calls
- **Performance improvement**: Cached responses are returned instantly
- **Consistency**: Identical inputs always return identical outputs
- **Resume capability**: Failed runs can resume from cached results

## Cache Backends

DELM supports multiple cache backends, each with different performance characteristics:

### SQLite (Default)
```yaml
semantic_cache:
  backend: "sqlite"
  path: ".delm_cache"
  max_size_mb: 512
  synchronous: "normal"  # or "full" for better durability
```

**Best for**: Most use cases, good balance of performance and reliability

### LMDB
```yaml
semantic_cache:
  backend: "lmdb"
  path: ".delm_cache"
  max_size_mb: 1024
```

**Best for**: High-performance scenarios with large datasets

### Filesystem
```yaml
semantic_cache:
  backend: "filesystem"
  path: ".delm_cache"
  max_size_mb: 256
```

**Best for**: Simple deployments or when other backends aren't available

## Configuration Options

### Basic Configuration

```yaml
semantic_cache:
  backend: "sqlite"
  path: ".delm_cache"
  max_size_mb: 512
```

### Advanced Configuration

```yaml
semantic_cache:
  backend: "sqlite"
  path: "/path/to/custom/cache"
  max_size_mb: 1024
  synchronous: "full"  # SQLite only: "normal" or "full"
```

### Disable Caching

```yaml
# Omit semantic_cache section entirely, or set backend to null
semantic_cache:
  backend: null
```

## When to Use Caching

### Ideal Scenarios

1. **Reprocessing data**: Running the same extraction multiple times
2. **Incremental updates**: Adding new data to existing datasets
3. **Development/testing**: Iterating on schemas with the same data
4. **Resume scenarios**: Continuing failed or interrupted runs
5. **Similar content**: Processing documents with overlapping content

### When Not to Use Caching

1. **One-time processing**: Single runs with unique data
2. **Memory constraints**: Limited disk space for cache storage
3. **Security requirements**: Sensitive data that shouldn't be cached
4. **Frequently changing schemas**: When schema changes invalidate cache

## Cache Management

### Cache Size Management

```yaml
semantic_cache:
  max_size_mb: 512  # Maximum cache size in megabytes
```

When the cache exceeds this size, DELM automatically prunes old entries to make room for new ones.

### Cache Location

```yaml
semantic_cache:
  path: ".delm_cache"  # Relative to experiment directory
```

Or use an absolute path:

```yaml
semantic_cache:
  path: "/shared/cache/delm_cache"
```

### Cache Sharing

You can share caches between experiments by using the same path:

```python
# Experiment 1
pipeline1 = DELM.from_yaml(
    config_path="config1.yaml",
    experiment_name="experiment_1",
    experiment_directory=Path("experiments"),
)

# Experiment 2 (shares cache with experiment 1)
pipeline2 = DELM.from_yaml(
    config_path="config2.yaml", 
    experiment_name="experiment_2",
    experiment_directory=Path("experiments"),
)
```

## Monitoring Cache Performance

### Cache Hit Rates

```python
# Get cache statistics
cost_summary = pipeline.get_cost_summary()
print(f"Cache hits: {cost_summary.get('total_cached_tokens', 0):,}")
print(f"Total tokens: {cost_summary.get('total_input_tokens', 0):,}")

# Calculate hit rate
hit_rate = cost_summary.get('total_cached_tokens', 0) / cost_summary.get('total_input_tokens', 1)
print(f"Cache hit rate: {hit_rate:.1%}")
```

### Cache Size Monitoring

```python
import os
from pathlib import Path

cache_path = Path(".delm_cache")
if cache_path.exists():
    cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    print(f"Cache size: {cache_size / (1024*1024):.1f} MB")
```

## Best Practices

### 1. Choose the Right Backend

- **SQLite**: Good default choice for most scenarios
- **LMDB**: Use for high-performance, large-scale processing
- **Filesystem**: Use when other backends aren't available

### 2. Set Appropriate Cache Size

```yaml
# For small datasets (< 1GB)
max_size_mb: 256

# For medium datasets (1-10GB)  
max_size_mb: 512

# For large datasets (> 10GB)
max_size_mb: 1024
```

### 3. Use Consistent Cache Paths

```yaml
# Good: Consistent relative path
semantic_cache:
  path: ".delm_cache"

# Avoid: Different paths for similar experiments
semantic_cache:
  path: "experiment_1_cache"  # Won't share with other experiments
```

### 4. Monitor Cache Performance

- Check cache hit rates regularly
- Monitor disk usage
- Clean up old caches when needed

### 5. Handle Cache Invalidation

When you change your schema or configuration, consider clearing the cache:

```bash
# Remove cache directory
rm -rf .delm_cache

# Or use a new cache path
semantic_cache:
  path: ".delm_cache_v2"
```

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.


## Next Steps

- [Cost Tracking](cost-tracking.md) - Monitor costs and budget limits
- [Batch Processing](batch-processing.md) - Optimize performance with batching
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
