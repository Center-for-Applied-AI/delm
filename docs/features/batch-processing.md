# Batch Processing

Learn how to optimize DELM performance with batching, concurrent processing, checkpointing, and experiment management.

## Batch Processing Overview

DELM processes data in batches to optimize API usage and improve performance. You can configure batch size, concurrent workers, and checkpointing to match your needs.

### Basic Configuration

```yaml
llm_extraction:
  batch_size: 10        # Records per batch
  max_workers: 1        # Concurrent workers
```

## Batch Size Optimization

### Choosing Batch Size

**Small batches** (5-10 records):
- ✅ Lower memory usage
- ✅ Better error isolation
- ✅ More frequent checkpointing
- ❌ Higher API overhead
- ❌ Slower overall processing

**Large batches** (20-50 records):
- ✅ Lower API overhead
- ✅ Faster overall processing
- ✅ Better throughput
- ❌ Higher memory usage
- ❌ Less frequent checkpointing


## Concurrent Processing

### Worker Configuration

```yaml
llm_extraction:
  max_workers: 2        # Number of concurrent workers
```

**Single worker** (max_workers: 1):
- ✅ Predictable processing order
- ✅ Lower resource usage
- ✅ Easier debugging
- ❌ Slower processing

**Multiple workers** (max_workers: 2-4):
- ✅ Faster processing
- ✅ Better resource utilization
- ❌ Higher memory usage
- ❌ More complex error handling

### Rate Limit Considerations

```yaml
llm_extraction:
  batch_size: 10
  max_workers: 2         # Stay within provider rate limits
  max_retries: 3         # Handle rate limit errors
  base_delay: 1.0        # Delay between retries
```

## Checkpointing and Resume

### Automatic Checkpointing

DELM automatically saves progress during processing:

```python
pipeline = DELM.from_yaml(
    config_path="config.yaml",
    experiment_name="my_experiment",
    experiment_directory=Path("experiments"),
    auto_checkpoint_and_resume_experiment=True  # Default: True
)
```


## Experiment Management

### Experiment Storage

DELM creates organized experiment directories:

```
experiments/
└── my_experiment/
    ├── delm_data/
    │   ├── preprocessed_data.feather
    │   └── extracted_data.feather
    ├── delm_logs/
    │   └── delm_my_experiment_2024-01-15_14-30-00.log
    └── cost_summary.json
```

### Experiment Lifecycle

#### 1. Create Experiment
```python
pipeline = DELM.from_yaml(
    config_path="config.yaml",
    experiment_name="production_run_v1",
    experiment_directory=Path("experiments"),
    overwrite_experiment=True  # Start fresh
)
```

#### 2. Process Data
```python
pipeline.prep_data("data/input.csv")
pipeline.process_via_llm()
```

#### 3. Save Results
```python
results = pipeline.get_extraction_results()
cost_summary = pipeline.get_cost_summary()

# Save to custom location
results.to_csv("results/final_extractions.csv", index=False)
```



## Performance Monitoring

Monitor your processing progress and costs using the built-in methods:

```python
# Check processing progress
results = pipeline.get_extraction_results()
cost_summary = pipeline.get_cost_summary()
```

## Error Handling and Recovery

DELM automatically retries failed requests and provides checkpointing for recovery:

```yaml
llm_extraction:
  max_retries: 3         # Number of retry attempts
  base_delay: 1.0        # Base delay between retries (seconds)
```


## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.


## Next Steps

- [Checkpointing](checkpointing.md) - Resume failed extractions automatically
- [Caching](caching.md) - Reduce costs with semantic caching
- [Cost Tracking](cost-tracking.md) - Monitor costs and budget limits
- [Text Processing](text-processing.md) - Optimize text splitting and scoring
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
