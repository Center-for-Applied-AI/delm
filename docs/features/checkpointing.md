# Checkpointing and Resuming

Learn how DELM automatically saves progress and allows you to resume failed extractions.

## How Checkpointing Works

DELM automatically saves your progress during extraction. If your experiment fails or is interrupted, you can simply rerun the same code to resume from where it left off.

### Automatic Checkpointing

```yaml
llm_extraction:
  batch_size: 10        # Progress saved after each batch
  max_workers: 2        # Concurrent processing
```

DELM saves progress after each batch completes, so you never lose more than one batch of work.

## Resuming Failed Experiments

### Simple Resume

If your experiment fails mid-run, just rerun the exact same code:

```python
from pathlib import Path
from delm import DELM

# Same code as before - DELM will automatically resume
pipeline = DELM.from_yaml(
    config_path="config.yaml",
    experiment_name="my_extraction",  # Same experiment name
    experiment_directory=Path("experiments")
)

pipeline.prep_data("data/input.csv")
pipeline.process_via_llm()  # Resumes from last checkpoint
```

### What Gets Resumed

- **Completed batches**: Already processed and saved
- **Failed batches**: Will be retried from the beginning
- **Progress tracking**: Cost and processing statistics continue from where they left off

## Checkpoint Management

### Checkpoint Location

Checkpoints are stored in your experiment directory:

```
experiments/
└── my_extraction/
    ├── checkpoints/          # Batch progress
    ├── logs/                 # Processing logs
    └── results/             # Final results
```

### Manual Checkpoint Control

```python
# Check current progress
results = pipeline.get_extraction_results()
total_chunks = len(results)
processed = len(results.dropna(subset=['delm_extracted_data_json']))
print(f"Progress: {processed}/{total_chunks} chunks completed")

# Force checkpoint (if needed)
pipeline.save_checkpoint()
```

## Error Recovery

### Common Scenarios

**Network timeout**: Rerun the same code - DELM will retry the failed batch
**Out of memory**: Reduce batch size in config, then rerun
**API rate limits**: DELM automatically handles retries with exponential backoff

### Resume After Configuration Changes

If you need to change configuration (like reducing batch size), use a new experiment name:

```python
# New experiment with smaller batch size
pipeline = DELM.from_yaml(
    config_path="config_smaller_batch.yaml",  # Updated config
    experiment_name="my_extraction_v2",        # New experiment name
    experiment_directory=Path("experiments")
)
```

## Best Practices

### 1. Use Descriptive Experiment Names

```python
# Good: Descriptive with version info
experiment_name = "financial_reports_v2"

# Avoid: Generic names that might conflict
experiment_name = "test"
```

### 2. Monitor Progress

```python
# Check progress during long runs
results = pipeline.get_extraction_results()
cost_summary = pipeline.get_cost_summary()
print(f"Processed: {len(results)} chunks, Cost: ${cost_summary['total_cost']:.2f}")
```

### 3. Handle Large Datasets

For very large datasets, consider:

- **Smaller batch sizes**: More frequent checkpoints
- **Lower concurrency**: Reduces memory pressure
- **Regular monitoring**: Check progress periodically

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.

## Next Steps

- [Batch Processing](batch-processing.md) - Optimize performance with batching
- [Cost Tracking](cost-tracking.md) - Monitor costs and budget limits
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
