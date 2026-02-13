# Caching

Learn how to use DELM's caching to reduce costs and improve performance by avoiding duplicate API calls.

**Recommendation**: Caching is **nearly always recommended**. It saves money by avoiding duplicate API calls, and the performance overhead is minimal. Enable caching unless you have a specific reason not to.

**Default Behavior**: Caching is **enabled by default** using the SQLite backend. The cache is stored at `.delm/cache` (relative to your current working directory). You don't need to configure anything to start using it.

## What is Caching?

DELM caches LLM responses using an **exact-match** key system. When you process text with the same prompt, model, and request settings, DELM returns the cached result instead of making a new API call.

### How It Works

The cache key is computed from:
- The rendered prompt text (including chunk content and template variables)
- The system prompt
- The model name (e.g., `gpt-4o-mini`)
- The temperature setting
- The Instructor `mode` (when set)
- `max_completion_tokens` (when not the default `4096`)

If all of these match exactly, the cached response is returned. This means **identical inputs always return identical outputs**, even across different runs.

For backward compatibility, older cache entries (created before `mode`/`max_completion_tokens` were included) are still reused when running with `mode=None` and `max_completion_tokens=4096`.

### Benefits

- **Cost reduction**: Avoid paying for duplicate API calls
- **Performance improvement**: Cached responses are returned instantly
- **Consistency**: Identical inputs always return identical outputs
- **Resume capability**: Failed runs can resume from cached results

### Key Use Case: Scoring Strategies and Filters

Caching is particularly valuable when using **relevance scoring and filtering**. Even when you filter chunks based on scores, the underlying text chunks can overlap across different filter thresholds or scoring strategies. Caching ensures you only pay once for processing the same chunk, regardless of which filter settings you use.

For example:
- Run 1: Filter with `score >= 0.8` (processes chunks A, B, C)
- Run 2: Filter with `score >= 0.5` (processes chunks A, B, C, D, E)

With caching enabled, chunks A, B, and C from Run 1 are cached, so Run 2 only pays to process chunks D and E.

### Important: Temperature and Caching

**Even with a non-zero temperature**, if you use the same prompt, model, and temperature value, DELM will return the cached result (which was generated with that temperature). The cache does **not** re-generate responses to get different outputs.

**To get variable results**: If you want different outputs for the same input (e.g., for testing or exploration), you must **disable caching** for that run.

```python
# Same inputs = same cached result (even with temperature > 0)
delm = DELM(
    # ...
    temperature=0.7,
    cache_backend="sqlite"  # Enabled
)
results1 = delm.extract("data.txt")  # First call: generates response
results2 = delm.extract("data.txt")  # Second call: returns cached result

# To get different results, disable cache
delm_no_cache = DELM(
    # ...
    temperature=0.7,
    cache_backend=None  # Disabled
)
```

## Cache Backends

DELM supports multiple cache backends, each with different performance characteristics:

### SQLite (Default)

```python
delm = DELM(
    # ...
    cache_backend="sqlite",
    cache_path=".delm/cache",
    cache_max_size_mb=512,
    cache_synchronous="normal"  # or "full" for better durability
)
```

**Best for**: Most use cases, good balance of performance and reliability

### LMDB

```python
delm = DELM(
    # ...
    cache_backend="lmdb",
    cache_path=".delm/cache",
    cache_max_size_mb=1024
)
```

**Best for**: High-performance scenarios with large datasets

*Note: Requires `pip install lmdb`*

### Filesystem

```python
delm = DELM(
    # ...
    cache_backend="filesystem",
    cache_path=".delm/cache",
    cache_max_size_mb=256
)
```

**Best for**: Simple deployments or when other backends aren't available

## Configuration

### Basic Configuration

```python
delm = DELM(
    # ...
    cache_backend="sqlite",
    cache_path=".delm/cache",
    cache_max_size_mb=512
)
```

### Disable Caching

```python
delm = DELM(
    # ...
    cache_backend=None  # Disable caching
)
```

## When Not to Use Caching

Caching is recommended in almost all cases. Only disable it if:

1. **Variable outputs desired**: When you want different results for the same input (e.g., exploring different outputs with the same prompt and temperature). Even then, you can disable caching only for specific runs.

2. **Very tight memory constraints**: If you have extremely limited disk space. However, you can control cache size with `cache_max_size_mb`, so this is rarely necessary.

## Cache Management

### Cache Size Management

The cache automatically prunes old entries when it exceeds `cache_max_size_mb`:

```python
delm = DELM(
    # ...
    cache_max_size_mb=512  # Maximum cache size in megabytes
)
```

### Cache Location

```python
# Relative path (default)
delm = DELM(
    # ...
    cache_path=".delm/cache"
)

# Absolute path
delm = DELM(
    # ...
    cache_path="/shared/cache/delm_cache"
)
```

### Cache Sharing

You can share caches between experiments by using the same path:

```python
# Experiment 1
delm1 = DELM(
    # ...
    cache_path=".delm/shared_cache"
)

# Experiment 2 (shares cache with experiment 1)
delm2 = DELM(
    # ...
    cache_path=".delm/shared_cache"  # Same path = shared cache
)
```

## Monitoring Cache Performance

### Programmatic Access

You can inspect cache statistics programmatically:

```python
# Access cache stats through the extraction manager
# (Note: This requires accessing internal components)
cache_stats = delm.semantic_cache.stats()
print(f"Cache entries: {cache_stats.get('entries', 0)}")
print(f"Cache size: {cache_stats.get('bytes', 0) / (1024*1024):.1f} MB")
print(f"Cache hits: {cache_stats.get('hit', 0)}")
print(f"Cache misses: {cache_stats.get('miss', 0)}")
```

### Command-Line Interface

DELM includes a CLI tool for inspecting and managing caches. After installing DELM (`pip install delm`), you can use it directly:

```bash
# View cache statistics (SQLite backend - default)
python -m delm.utils.semantic_cache .delm/cache --stats

# Prune cache to a specific size (e.g., 256 MB)
python -m delm.utils.semantic_cache .delm/cache --prune 256

# For other backends, specify --backend
python -m delm.utils.semantic_cache .delm/cache --backend lmdb --stats
```

**Options**:
- `cache_dir`: Path to your cache directory
- `--backend`: Cache backend (`sqlite` default, `lmdb`, or `filesystem`) - only needed if not using SQLite
- `--stats`: Show cache statistics and exit
- `--prune MEGABYTES`: Prune cache to the specified size in megabytes
