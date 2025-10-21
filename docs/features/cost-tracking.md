# Cost Tracking

Learn how to monitor API costs, set budget limits, and optimize spending with DELM's cost tracking features.

## Enabling Cost Tracking

Cost tracking is enabled by default, but you can configure it explicitly:

```yaml
llm_extraction:
  track_cost: true        # Enable cost tracking
  max_budget: 100.0       # Optional: Set budget limit
```

## Budget Management

### Setting Budget Limits

```yaml
llm_extraction:
  track_cost: true
  max_budget: 50.0        # Stop processing if cost exceeds $50
```

**Important**: When `max_budget` is reached, processing stops immediately. You can resume from checkpoints after increasing the budget.

### Budget Monitoring

```python
# Check current costs during processing
cost_summary = pipeline.get_cost_summary()
current_cost = cost_summary['total_cost']
budget_limit = pipeline.config.llm_extraction.max_budget

if current_cost > budget_limit * 0.8:  # 80% of budget
    print(f"Warning: Approaching budget limit ({current_cost:.2f}/{budget_limit})")
```

### Dynamic Budget Adjustment

```python
# Increase budget during processing
pipeline.config.llm_extraction.max_budget = 200.0
print("Budget increased to $200")
```

## Cost Analysis

### Basic Cost Summary

```python
cost_summary = pipeline.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost']:.4f}")
print(f"Input tokens: {cost_summary['total_input_tokens']:,}")
print(f"Output tokens: {cost_summary['total_output_tokens']:,}")
```

### Detailed Cost Breakdown

```python
# Get detailed cost information
cost_summary = pipeline.get_cost_summary()

# Cost by token type
print(f"Input cost: ${cost_summary.get('input_cost', 0):.4f}")
print(f"Output cost: ${cost_summary.get('output_cost', 0):.4f}")

# Token usage
print(f"Input tokens: {cost_summary.get('total_input_tokens', 0):,}")
print(f"Output tokens: {cost_summary.get('total_output_tokens', 0):,}")

# Caching impact
cached_tokens = cost_summary.get('total_cached_tokens', 0)
if cached_tokens > 0:
    print(f"Cached tokens: {cached_tokens:,}")
    print(f"Cache savings: ${cost_summary.get('cached_cost', 0):.4f}")
```

### Cost by Provider

```python
# Analyze costs across different providers
cost_df = pipeline.get_cost_summary_df()
print(cost_df)

# Group by provider
provider_costs = cost_df.groupby('provider')['cost'].sum()
print("Cost by provider:")
print(provider_costs)
```

## Custom Pricing

### Override Model Pricing

```yaml
llm_extraction:
  provider: "openai"
  name: "gpt-4o-mini"
  model_input_cost_per_1M_tokens: 0.15    # Custom input pricing
  model_output_cost_per_1M_tokens: 0.60   # Custom output pricing
```

### Custom Provider Pricing

```yaml
llm_extraction:
  provider: "custom_provider"
  name: "custom_model"
  model_input_cost_per_1M_tokens: 0.10
  model_output_cost_per_1M_tokens: 0.40
  track_cost: true
```

**Note**: When using custom pricing, ensure `track_cost: true` is set.

## Cost Optimization Strategies

### 1. Use Caching

```yaml
semantic_cache:
  backend: "sqlite"
  path: ".delm_cache"
  max_size_mb: 512
```

Caching reduces costs by avoiding duplicate API calls for similar content.

### 2. Optimize Text Processing

```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance"]
  pandas_score_filter: "delm_score >= 0.7"  # Filter irrelevant chunks
```

Filtering reduces the number of chunks processed, lowering costs.

### 3. Choose Cost-Effective Models

```yaml
# Cheaper models for initial testing
llm_extraction:
  provider: "openai"
  name: "gpt-3.5-turbo"  # Cheaper than gpt-4o-mini

# Or use Anthropic's cost-effective model
llm_extraction:
  provider: "anthropic"
  name: "claude-3-haiku"  # Most cost-effective Anthropic model
```

### 4. Optimize Batch Processing

```yaml
llm_extraction:
  batch_size: 20        # Larger batches reduce API overhead
  max_workers: 2        # Parallel processing
```

### 5. Use Cost Estimation

```python
from delm.utils.cost_estimation import estimate_input_token_cost

# Estimate costs before running
estimated_cost = estimate_input_token_cost(
    config="config.yaml",
    data_source="data.csv"
)
print(f"Estimated cost: ${estimated_cost:.2f}")
```

## Cost Monitoring

Monitor costs during processing:

```python
# Check current costs
cost_summary = pipeline.get_cost_summary()
print(f"Current cost: ${cost_summary['total_cost']:.4f}")
```

## Cost Analysis

Get detailed cost breakdowns:

```python
# Get cost summary DataFrame
cost_df = pipeline.get_cost_summary_df()
print(cost_df)
```


## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.


## Next Steps

- [Cost Estimation Tutorial](../tutorials/cost-estimation.md) - Learn to estimate costs before running extractions
- [Caching](caching.md) - Reduce costs with semantic caching
- [Batch Processing](batch-processing.md) - Optimize performance with batching
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
