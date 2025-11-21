# Logging & Debugging

Control logging output to troubleshoot issues and monitor extraction progress.

## DELM Extraction

```python
from delm import DELM, Schema, ExtractionVariable

schema = Schema.simple([
    ExtractionVariable(name="price", data_type="number")
])

delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    console_log_level="INFO",      # "DEBUG", "INFO", "WARNING", "ERROR"
    save_log_file=True,             # Save logs to disk
    log_dir=".delm/logs",           # Log directory
    log_file_prefix="extraction_",  # Log filename prefix
    file_log_level="DEBUG"          # File verbosity (usually more detailed)
)

results = delm.extract("data.csv")
# Log file: .delm/logs/extraction_YYYY-MM-DD_HH-MM-SS.log
```

## Cost Estimation

```python
from delm.utils.cost_estimation import estimate_input_token_cost, estimate_total_cost

# Both functions support the same logging parameters
input_cost = estimate_input_token_cost(
    config=delm,
    data_source="data.csv",
    save_file_log=True,
    log_dir=".delm/logs/cost_estimation",  # Default
    console_log_level="INFO",
    file_log_level="DEBUG"
)
# Log file: delm_cost_estimation_YYYY-MM-DD_HH-MM-SS.log
```

## Performance Evaluation

```python
from delm.utils.performance_estimation import estimate_performance

metrics, comparison_df = estimate_performance(
    config=delm,
    data_source="data.csv",
    expected_extraction_output_df=expected_df,
    true_json_column="expected_extraction",
    matching_id_column="id",
    save_file_log=True,
    log_dir=".delm/logs/performance_estimation",  # Default
    console_log_level="INFO",
    file_log_level="DEBUG"
)
# Log file: delm_performance_estimation_YYYY-MM-DD_HH-MM-SS.log
```
