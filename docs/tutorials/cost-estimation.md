# Cost Estimation Tutorial

Learn how to estimate extraction costs before running large-scale data processing jobs.

## When to Use Cost Estimation

Cost estimation helps you:
- **Budget planning**: Understand costs before committing to large extractions
- **Model selection**: Compare costs between different models and providers
- **Configuration optimization**: Find the most cost-effective settings
- **Risk management**: Avoid unexpected charges on large datasets

## Input Token Cost Estimation

Estimate costs without making API calls using `estimate_input_token_cost`:

```python
from delm.utils.cost_estimation import estimate_input_token_cost

# Estimate input token costs for your dataset
input_cost = estimate_input_token_cost(
    config="config.yaml",
    data_source="data.csv"
)

print(f"Estimated input token cost: ${input_cost:.2f}")
```

### How It Works

This method:
1. Loads your configuration and data
2. Processes text through splitting and scoring (if configured)
3. Counts input tokens using the same prompts that would be sent to the LLM
4. Calculates cost based on your model's input token pricing

### Example Output

```
Estimated input token cost: $12.45
```

**Note**: This only estimates input tokens. Total costs will be higher due to output tokens.

## Total Cost Estimation

Get more accurate estimates using `estimate_total_cost` with actual API calls:

```python
from delm.utils.cost_estimation import estimate_total_cost

# Estimate total costs using a sample of your data
total_cost = estimate_total_cost(
    config="config.yaml",
    data_source="data.csv",
    sample_size=100  # Process 100 records for estimation
)

print(f"Estimated total cost: ${total_cost:.2f}")
```

### How It Works

This method:
1. Samples a subset of your data (default: 10 records)
2. Runs the full extraction pipeline on the sample
3. Tracks actual input and output token usage
4. Scales the sample cost to estimate full dataset cost

### Example Output

```
Estimated total cost: $156.78
```

**Warning**: This method makes actual API calls and will charge you for the sample data.

## Interpreting Results

### Cost Breakdown

After running either estimation method, you can get detailed cost information:

```python
# If you ran estimate_total_cost, you can access the pipeline's cost tracker
from delm import DELM

pipeline = DELM.from_yaml(
    config_path="config.yaml",
    experiment_name="cost_estimation",
    experiment_directory=Path("experiments"),
)

# Run your estimation
pipeline.prep_data("data.csv")
pipeline.process_via_llm()

# Get detailed cost summary
cost_summary = pipeline.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost']:.4f}")
print(f"Input tokens: {cost_summary['total_input_tokens']:,}")
print(f"Output tokens: {cost_summary['total_output_tokens']:,}")
print(f"Cached tokens: {cost_summary.get('total_cached_tokens', 0):,}")
```

### Cost Optimization Strategies

Based on your estimates, consider these optimizations:

#### 1. Reduce Input Tokens
```yaml
data_preprocessing:
  splitting:
    type: "FixedWindowSplit"
    window: 3  # Smaller chunks = fewer tokens
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast"]
  pandas_score_filter: "delm_score >= 0.8"  # Filter irrelevant chunks
```

#### 2. Use Caching
```yaml
semantic_cache:
  backend: "sqlite"
  path: ".delm_cache"
  max_size_mb: 512
```

#### 3. Choose Cost-Effective Models
```yaml
llm_extraction:
  provider: "openai"
  name: "gpt-3.5-turbo"  # Cheaper than gpt-4o-mini
  # or
  provider: "anthropic"
  name: "claude-3-haiku"  # Anthropic's most cost-effective model
```

#### 4. Optimize Batch Size
```yaml
llm_extraction:
  batch_size: 20  # Larger batches can reduce overhead
  max_workers: 2  # Parallel processing
```

## Best Practices

### 1. Start with Input Token Estimation
Always begin with `estimate_input_token_cost` since it's free and gives you a baseline.

### 2. Use Representative Samples
For `estimate_total_cost`, use a sample size that represents your full dataset:
- **Small datasets** (< 1000 records): Use 10-20% of your data
- **Medium datasets** (1000-10000 records): Use 5-10% of your data  
- **Large datasets** (> 10000 records): Use 1-5% of your data

### 3. Account for Caching
If you plan to use caching, your actual costs will be lower than estimates since repeated chunks won't be re-processed.

### 4. Set Budget Limits
```yaml
llm_extraction:
  track_cost: true
  max_budget: 100.0  # Stop processing if cost exceeds $100
```

### 5. Monitor During Processing
```python
# Check costs during long-running jobs
cost_summary = pipeline.get_cost_summary()
if cost_summary['total_cost'] > 50.0:
    print("Warning: Approaching budget limit")
```

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.

## Next Steps

- [Performance Evaluation Tutorial](performance-evaluation.md) - Learn to measure extraction quality
- [Cost Tracking](../features/cost-tracking.md) - Advanced cost monitoring and budget management
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference

