# Cost Estimation

Utilities for estimating API costs before running extractions.

## estimate_input_token_cost()

Estimate cost based on input tokens only (free, no API calls).

```python
from delm.utils.cost_estimation import estimate_input_token_cost

cost_report = estimate_input_token_cost(
    config: DELM | DELMConfig | str | Path,
    data_source: str | Path | pd.DataFrame,
    save_file_log: bool = False,
    log_dir: str | Path | None = ".delm/logs/cost_estimation",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG"
) -> dict
```

**Parameters:**
- `config`: DELM instance, DELMConfig, or path to config YAML
- `data_source`: Input data (file path, directory, or DataFrame)
- `save_file_log`: Save log file
- `log_dir`: Log directory
- `console_log_level`: Console verbosity
- `file_log_level`: File verbosity

**Returns:** Dictionary with:
- `estimated_input_tokens` (int)
- `estimated_input_cost` (float)
- `num_records` (int)
- `num_chunks` (int)

**Note:** Counts cached requests toward token cost (they would be cached on first run).

## estimate_total_cost()

Estimate total cost (input + output tokens) using sample API calls.

```python
from delm.utils.cost_estimation import estimate_total_cost

cost_report = estimate_total_cost(
    config: DELM | DELMConfig | str | Path,
    data_source: str | Path | pd.DataFrame,
    sample_size: int = 10,
    save_file_log: bool = False,
    log_dir: str | Path | None = ".delm/logs/cost_estimation",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG"
) -> dict
```

**Parameters:**
- `config`: DELM instance, DELMConfig, or path to config YAML
- `data_source`: Input data
- `sample_size`: Number of chunks to sample for estimation
- `save_file_log`, `log_dir`, `console_log_level`, `file_log_level`: Logging settings

**Returns:** Dictionary with:
- `estimated_total_cost` (float)
- `estimated_input_tokens` (int)
- `estimated_output_tokens` (int)
- `estimated_input_cost` (float)
- `estimated_output_cost` (float)
- `sample_size` (int)
- `total_chunks` (int)

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
print(f"Input cost: ${input_cost['estimated_input_cost']:.4f}")

# Sample-based estimate (costs ~$0.01)
total_cost = estimate_total_cost(delm, "data.csv", sample_size=10)
print(f"Total estimated cost: ${total_cost['estimated_total_cost']:.2f}")
```

