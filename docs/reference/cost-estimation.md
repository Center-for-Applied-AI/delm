# Cost Estimation

Utilities for estimating API costs before running extractions.

## estimate_input_token_cost()

Estimate cost based on input tokens only (free, no API calls).

```python
from delm.utils.cost_estimation import estimate_input_token_cost

input_cost = estimate_input_token_cost(
    config: DELM | DELMConfig | str | Path,
    data_source: str | Path | pd.DataFrame,
    save_file_log: bool = False,
    log_dir: str | Path | None = ".delm/logs/cost_estimation",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG"
) -> float
```

**Parameters:**
- `config`: DELM instance, DELMConfig, or path to config YAML
- `data_source`: Input data (file path, directory, or DataFrame)
- `save_file_log`: Save log file
- `log_dir`: Log directory
- `console_log_level`: Console verbosity
- `file_log_level`: File verbosity

**Returns:** Estimated dollar cost (float) of input tokens for all chunks.

**Note:** Counts cached requests toward token cost (they would be cached on first run).

## estimate_max_total_cost()

Estimate an upper bound on total cost (free, no API calls). For each chunk the
output tokens are bounded by
`min(max_completion_tokens, context_window - input_tokens)`, so the bound is

```
input_price * input_tokens
+ output_price * min(max_completion_tokens, context_window - input_tokens)
```

summed over all chunks. Context window and max output tokens are looked up
from the tokencost database; when unavailable (custom models with manual price
overrides), only `max_completion_tokens` bounds the output.

```python
from delm.utils.cost_estimation import estimate_max_total_cost

max_cost = estimate_max_total_cost(
    config: DELM | DELMConfig | str | Path,
    data_source: str | Path | pd.DataFrame,
    save_file_log: bool = False,
    log_dir: str | Path | None = ".delm/logs/cost_estimation",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG"
) -> float
```

**Parameters:** Same as `estimate_input_token_cost()`.

**Returns:** Upper-bound dollar cost (float) for processing all chunks.

## estimate_total_cost()

Estimate total cost (input + output tokens) using sample API calls.

```python
from delm.utils.cost_estimation import estimate_total_cost

total_cost = estimate_total_cost(
    config: DELM | DELMConfig | str | Path,
    data_source: str | Path | pd.DataFrame,
    sample_size: int = 10,
    save_file_log: bool = False,
    log_dir: str | Path | None = ".delm/logs/cost_estimation",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG"
) -> float
```

**Parameters:**
- `config`: DELM instance, DELMConfig, or path to config YAML
- `data_source`: Input data
- `sample_size`: Number of records to sample for estimation
- `save_file_log`, `log_dir`, `console_log_level`, `file_log_level`: Logging settings

**Returns:** Estimated dollar cost (float) for processing the entire dataset,
extrapolated from the sample by input-token share.

**Warning:** Makes real API calls (costs apply).

## Example

```python
from delm import DELM, Schema, ExtractionVariable
from delm.utils.cost_estimation import estimate_input_token_cost, estimate_total_cost

schema = Schema.simple(
    ExtractionVariable("price", "Price value", "number")
)

delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    model_input_cost_per_1M_tokens=0.15,  # Custom pricing
    model_output_cost_per_1M_tokens=0.60
)

# Free estimate (input tokens only)
input_cost = estimate_input_token_cost(delm, "data.csv")
print(f"Input cost: ${input_cost:.4f}")

# Sample-based estimate (costs ~$0.01)
total_cost = estimate_total_cost(delm, "data.csv", sample_size=10)
print(f"Total estimated cost: ${total_cost:.2f}")
```

