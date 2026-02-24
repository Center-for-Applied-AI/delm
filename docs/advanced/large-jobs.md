# Large Jobs & Checkpointing

Process large datasets reliably with automatic progress saving and resumption.

## Overview

When processing thousands of documents, failures happen (network timeouts, rate limits, crashes). DELM handles this through:

1. **Automatic Checkpointing**: Saves progress after each batch completes
2. **Automatic Resumption**: Detects existing progress and continues from where it left off
3. **Experiment Management**: Organizes all data, configs, and checkpoints in a structured directory

**Note**: For moderate datasets, **caching alone is often sufficient** - interrupted jobs can simply be re-run with cached results returned instantly at no cost. Use disk storage/checkpointing for very large datasets (100K+ chunks) or when you want to save configs and/or reload results later.

## Enabling Disk Storage & Checkpointing

Checkpointing requires disk storage to be enabled:

```python
from delm import DELM, Schema, ExtractionVariable

schema = Schema.simple([
    ExtractionVariable(name="company", data_type="string"),
    ExtractionVariable(name="revenue", data_type="number")
])

delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    
    # Enable disk storage for checkpointing
    use_disk_storage=True,
    experiment_path="experiments/annual_reports_2024",
    auto_checkpoint_and_resume_experiment=True  # Default is True
)

# Run extraction - progress saved automatically after each batch
results_df = delm.extract("data/reports.csv")
```

## How Checkpointing Works

### Automatic Progress Saving

DELM processes data in batches. After each batch completes, it saves:
1. **Batch results** - Extracted data for that batch
2. **State** - Cost tracker and progress information

If your job crashes after processing 5,000 of 10,000 chunks, those 5,000 are already saved.

### Automatic Resumption

Simply **re-run the same code**. DELM will:
1. Check the experiment directory for existing checkpoints
2. Load already-processed batch IDs
3. Skip completed batches and process only remaining chunks

```python
# First run - crashes at 50%
results_df = delm.extract("data/reports.csv")

# ... Fix issue (internet, rate limit, etc.) ...

# Second run - automatically resumes from 50%
results_df = delm.extract("data/reports.csv")
```

**Important**: The schema and config must match exactly between runs. If they don't match, DELM will raise an error to prevent data inconsistency.

## Configuration Options

### Batch Size

Batch size determines how many chunks are processed before a checkpoint:

```python
delm = DELM(
    schema=schema,
    batch_size=10,  # Checkpoint every 10 chunks
    use_disk_storage=True,
    experiment_path="experiments/my_experiment"
)
```

**Trade-offs:**
- **Smaller batches (1-10)**: More frequent checkpoints, less work lost on failure, but more checkpoint overhead
- **Larger batches (50-100)**: Less checkpoint overhead, but more work lost if a batch fails

**Note**: If a batch fails midway (e.g., chunk 7 of 10), the entire batch is retried. Smaller batches mean less wasted work on retries.

### Concurrent Workers

Process chunks within a batch concurrently for speed:

```python
delm = DELM(
    schema=schema,
    batch_size=10,
    max_workers=4,  # Process 4 chunks in parallel within each batch
    use_disk_storage=True,
    experiment_path="experiments/my_experiment"
)
```

**How it works**: All workers process chunks from the **same batch** in parallel. Once all chunks in a batch complete, the checkpoint is saved, and they move to the next batch.

**Best Practice**: Set `max_workers` ≤ `batch_size`. Having more workers than chunks in a batch just wastes resources. For example, if `batch_size=10` and `max_workers=20`, you'll have 10 idle workers.

**Warning**: More workers = more concurrent API calls = higher rate limit usage. If you hit "429 Too Many Requests" errors, you may need to reduce `max_workers` or increase `base_delay`. A better solution is to set `rate_limit_tokens` and `rate_limit_requests` to match your provider's limits (with `rate_limit_period_seconds` to set the time window).

### Overwrite vs Resume

```python
# Resume from existing checkpoints (default)
delm = DELM(
    schema=schema,
    use_disk_storage=True,
    experiment_path="experiments/my_experiment",
    auto_checkpoint_and_resume_experiment=True  # Resume if possible
)

# Start fresh, delete existing experiment (waits 3 seconds as safety)
delm = DELM(
    schema=schema,
    use_disk_storage=True,
    experiment_path="experiments/my_experiment",
    overwrite_experiment=True  # Delete and start over
)
```

## Experiment Directory Structure

When `use_disk_storage=True`, DELM creates this structure:

```
experiments/
└── my_experiment/
    ├── config/
    │   └── config.yaml              # Saved config for verification
    ├── delm_data/
    │   ├── preprocessed.feather     # Preprocessed data (chunks, scores)
    │   └── extraction_result.feather # Final consolidated results
    └── delm_llm_processing/
        ├── batch_000000.feather     # Batch checkpoint files
        ├── batch_000001.feather
        ├── batch_000002.feather
        ├── ...
        └── state.json               # Cost tracker state
```

**File purposes:**
- **`config.yaml`**: Snapshot of your config for verification on resume
- **`preprocessed.feather`**: Processed text chunks ready for LLM extraction
- **`extraction_result.feather`**: Final results after all batches complete
- **`batch_*.feather`**: Individual batch checkpoints (deleted after consolidation)
- **`state.json`**: Tracks costs and progress

## Retrieving Results

### During Extraction

The `extract()` method returns results directly:

```python
results_df = delm.extract("data/reports.csv")
# Results available immediately
print(results_df.head())
```

### After Completion

If you've already run extraction and want to reload results later:

```python
# Create DELM instance pointing to the same experiment
delm = DELM(
    schema=schema,
    use_disk_storage=True,
    experiment_path="experiments/my_experiment"
)

# Get saved results
results_df = delm.get_extraction_results()
```

**Note**: `get_extraction_results()` only works after extraction has completed. It reads from `extraction_result.feather`.

## Troubleshooting

### "Experiment directory already exists"

If you see this error and want to continue from checkpoints:

```python
delm = DELM(
    schema=schema,
    use_disk_storage=True,
    experiment_path="experiments/my_experiment",
    auto_checkpoint_and_resume_experiment=True  # Enable resumption
)
```

If you want to start fresh:

```python
delm = DELM(
    schema=schema,
    use_disk_storage=True,
    experiment_path="experiments/my_experiment",
    overwrite_experiment=True  # Delete existing experiment
)
```

### "Config mismatch" Error

This happens when trying to resume with a different schema or config. Either:
- Use the exact same config as the original run, or
- Use `overwrite_experiment=True` to start fresh
