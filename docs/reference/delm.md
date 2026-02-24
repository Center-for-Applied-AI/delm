# DELM

Main extraction pipeline class.

## Class

```python
from delm import DELM

delm = DELM(
    schema,
    provider="openai",
    model="gpt-4o-mini",
    **kwargs
)
```

## Constructor Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `Schema \| str \| Path \| dict` | Extraction schema (Schema object, path to YAML, or dict) |

### LLM Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider (via Instructor) |
| `model` | `str` | `"gpt-4o-mini"` | Model name |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `batch_size` | `int` | `10` | Number of chunks per batch |
| `max_workers` | `int` | `1` | Concurrent workers per batch |
| `max_retries` | `int` | `3` | Retry attempts for failed requests |
| `base_delay` | `float` | `1.0` | Exponential backoff base delay (seconds) |
| `rate_limit_tokens` | `int` | `null` | Maximum tokens per rate limit period |
| `rate_limit_requests` | `int` | `null` | Maximum requests per rate limit period |
| `rate_limit_period_seconds` | `float` | `60.0` | Rate limit window in seconds |
| `max_completion_tokens` | `int` | `4096` | Max completion tokens per request |
| `api_kwargs` | `dict \| None` | `None` | Extra kwargs passed through to the LLM API call |


### Cost Tracking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_cost` | `bool` | `True` | Enable cost tracking |
| `max_budget` | `float \| None` | `None` | Stop extraction if budget exceeded (USD) |
| `model_input_cost_per_1M_tokens` | `float \| None` | `None` | Custom input token cost |
| `model_output_cost_per_1M_tokens` | `float \| None` | `None` | Custom output token cost |

### Preprocessing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_column` | `str` | `"text"` | Column containing text to extract from |
| `drop_target_column` | `bool` | `False` | Remove target column from output |
| `splitting_strategy` | `SplitStrategy \| dict \| None` | `None` | Text chunking strategy |
| `relevance_scorer` | `RelevanceScorer \| dict \| None` | `None` | Chunk relevance scoring |
| `score_filter` | `str \| None` | `None` | Pandas query to filter chunks (e.g., `"delm_score > 0.5"`) |

### Prompts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_template` | `str \| None` | `"Extract the following..."` | User prompt template |
| `system_prompt` | `str \| None` | `"You are a precise..."` | System prompt |

### Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_backend` | `str` | `"sqlite"` | Cache backend (`"sqlite"`, `"lmdb"`, `"filesystem"`) |
| `cache_path` | `str \| Path` | `".delm/cache"` | Cache directory |
| `cache_max_size_mb` | `int` | `512` | Max cache size (MB) |
| `cache_synchronous` | `str` | `"normal"` | SQLite sync mode |

### Experiment Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_disk_storage` | `bool` | `False` | Enable disk-based checkpointing |
| `experiment_path` | `str \| Path \| None` | `None` | Experiment directory (required if `use_disk_storage=True`) |
| `overwrite_experiment` | `bool` | `False` | Overwrite existing experiment |
| `auto_checkpoint_and_resume_experiment` | `bool` | `True` | Automatic checkpoint/resume |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_log_file` | `bool` | `False` | Save logs to disk |
| `log_dir` | `str \| Path \| None` | `".delm/logs"` | Log directory |
| `log_file_prefix` | `str` | `""` | Log filename prefix |
| `console_log_level` | `str` | `"INFO"` | Console log level |
| `file_log_level` | `str` | `"DEBUG"` | File log level |
| `override_logging` | `bool` | `True` | Override existing logging config |

## Methods

### extract()

Extract structured data from input.

```python
results_df = delm.extract(
    data: str | Path | pd.DataFrame,
    sample_size: int = -1
) -> pd.DataFrame
```

**Parameters:**
- `data`: Input data (file path, directory, or DataFrame)
- `sample_size`: Number of records to sample (`-1` = all)

**Returns:** DataFrame with original columns + DELM system columns

---

### prep_data()

Preprocess input data (loading, splitting, scoring).

```python
preprocessed_df = delm.prep_data(
    data: str | Path | pd.DataFrame,
    sample_size: int = -1
) -> pd.DataFrame
```

**Parameters:**
- `data`: Input data (file path, directory, or DataFrame)
- `sample_size`: Number of records to sample (`-1` = all)

**Returns:** Preprocessed DataFrame with text chunks

---

### process_via_llm()

Run LLM extraction on preprocessed data.

```python
results_df = delm.process_via_llm(
    preprocessed_file_path: Path | None = None
) -> pd.DataFrame
```

**Parameters:**
- `preprocessed_file_path`: Path to preprocessed data (optional, uses internal data if None)

**Returns:** DataFrame with extraction results

---

### get_extraction_results()

Get core extraction results (without metadata columns).

```python
results_df = delm.get_extraction_results() -> pd.DataFrame
```

**Returns:** DataFrame with only extraction columns (`delm_file_name`, `delm_raw_data`, `delm_text_chunk`, `delm_chunk_id`, `delm_batch_id`, `delm_errors`, `delm_extracted_data_json`)

---

### get_cost_summary()

Get cost tracking summary.

```python
cost_summary = delm.get_cost_summary() -> dict
```

**Returns:** Dictionary with keys:
- `input_tokens` (int)
- `output_tokens` (int)
- `total_cost` (float)
- `cached_tokens` (int)
- `cached_cost` (float)

**Raises:** `ValueError` if `track_cost=False`

---

### preview_prompt()

Preview the user prompt (without system prompt or Instructor wrapper).

```python
prompt = delm.preview_prompt(text: str | None = None) -> str
```

**Parameters:**
- `text`: Example text (uses placeholder if None)

**Returns:** Formatted user prompt string

---

### from_config()

Create DELM instance from configuration object or file.

```python
delm = DELM.from_config(
    config: str | Path | DELMConfig,
    **overrides
) -> DELM
```

**Parameters:**
- `config`: Config object or path to YAML
- `**overrides`: Override specific parameters (same as constructor)

**Returns:** DELM instance

