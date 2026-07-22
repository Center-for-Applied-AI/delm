# Cost Management

DELM provides tools to estimate costs before running a job, track costs during execution, and enforce budget limits to prevent overspending.

## Cost Estimation

Before running a large extraction job, you should estimate the potential cost. DELM offers two methods for this: a free input-only estimate and a more accurate sample-based estimate.

**Note on Pricing**: DELM uses the [tokencost](https://github.com/AgentOps-AI/tokencost) package to look up model prices automatically. If a model is missing from the bundled database (e.g. a model released after your tokencost version), DELM fetches tokencost's live price feed once per process, so newly released models (such as `claude-fable-5` or `gpt-5.6-sol`) resolve without an upgrade. If the model still isn't found or prices have changed, override with `model_input_cost_per_1M_tokens` and `model_output_cost_per_1M_tokens`.

### 1. Input Token Estimation (Free)

This method calculates the cost of **input tokens only** by tokenizing your dataset locally. It does **not** make any API calls.

**Best for**: Getting a lower-bound baseline cost quickly and for free.

```python
from delm import DELM
from delm.utils.cost_estimation import estimate_input_token_cost

# 1. Configure your pipeline
delm = DELM(
    # ... provider/model ...
    model_input_cost_per_1M_tokens=0.15,  # Custom pricing
    splitting_strategy={"type": "ParagraphSplit"}
)

# 2. Estimate cost using this configuration
input_cost = estimate_input_token_cost(
    config=delm,
    data_source="data/financial_reports.csv"
)

print(f"Estimated minimum cost (input only): ${input_cost:.2f}")
```

### 2. Upper-Bound Estimation (Free)

This method computes a **worst-case total cost** without API calls. Output
tokens per chunk are bounded by
`min(max_completion_tokens, context_window - input_tokens)`, using the model's
limits from the tokencost database.

**Best for**: Knowing the maximum you could possibly spend before launching a job.

```python
from delm.utils.cost_estimation import estimate_max_total_cost

max_cost = estimate_max_total_cost(
    config=delm,
    data_source="data/financial_reports.csv"
)

print(f"Worst-case total cost: ${max_cost:.2f}")
```

### 3. Total Cost Estimation (Sampled)

This method runs the full extraction pipeline on a small sample of your data to measure both **input and output** tokens. It then extrapolates the total cost.

**Best for**: Getting a realistic estimate of total spend, including the LLM's generation cost.

**Warning**: This will incur a small cost for processing the sample rows.

```python
from delm import DELM
from delm.utils.cost_estimation import estimate_total_cost

# 1. Configure your pipeline with custom pricing
delm = DELM(
    # ... provider/model ...
    model_input_cost_per_1M_tokens=0.15,
    model_output_cost_per_1M_tokens=0.60
)

# 2. Run estimation on a sample
total_cost = estimate_total_cost(
    config=delm,
    data_source="data/financial_reports.csv",
    sample_size=20  # Process 20 records
)

print(f"Estimated total cost: ${total_cost:.2f}")
```

**Important Note on Caching**: During cost estimation, requests available in the cache are **counted toward the token cost** (estimates assume you'll pay for all tokens). However, in the actual cost report after extraction, **cache hit requests are FREE** and not counted in the cost. This means your actual costs may be lower than estimates if you have cache hits.

## Budget Limits

You can set a hard budget limit to ensure you never accidentally overspend. If the limit is reached, DELM stops processing immediately but preserves all results extracted up to that point.

```python
from delm import DELM

delm = DELM(
    # ... other args ...
    track_cost=True,
    max_budget=50.0  # Stop processing if cost exceeds $50.00
)

results_df = delm.extract("data/")
```

## Cost Tracking

DELM automatically tracks token usage and costs for every run. You can access a summary report after execution.

**Important**: Cache hits are **FREE** - they don't count toward your cost. Only actual API calls are charged.

```python
# Run your extraction
results_df = delm.extract("data/")

# Get the cost report
summary = delm.get_cost_summary()

print("--- Cost Report ---")
print(f"Total Cost:      ${summary['total_cost']:.4f}")
print(f"Input Tokens:    {summary['input_tokens']:,}")
print(f"Output Tokens:   {summary['output_tokens']:,}")
```

The `total_cost` reflects only actual API charges. If you processed 1000 chunks but 300 were cache hits, you only pay for the 700 that required API calls.

---

**Disclaimer**: DELM's cost estimation and tracking features are provided as-is for informational purposes. DELM is not responsible for any errors, inaccuracies, or discrepancies in cost estimates or reported costs. Actual costs may vary due to model pricing changes, API rate fluctuations, or other factors. Users are responsible for verifying costs with their LLM provider and managing their own spending.
