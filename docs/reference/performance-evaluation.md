# Performance Evaluation

Evaluate extraction quality against ground truth data.

## estimate_performance()

Calculate precision, recall, and F1 scores for extraction results.

```python
from delm.utils.performance_estimation import estimate_performance

metrics, comparison_df = estimate_performance(
    config: DELM | DELMConfig | str | Path,
    data_source: str | Path | pd.DataFrame,
    expected_extraction_output_df: pd.DataFrame,
    true_json_column: str,
    matching_id_column: str,
    record_sample_size: int = -1,
    save_file_log: bool = False,
    log_dir: str | Path | None = ".delm/logs/performance_estimation",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG"
) -> tuple[dict, pd.DataFrame]
```

**Parameters:**
- `config`: DELM instance, DELMConfig, or path to config YAML
- `data_source`: Input data
- `expected_extraction_output_df`: DataFrame with ground truth extractions
- `true_json_column`: Column in expected_df containing ground truth JSON
- `matching_id_column`: Column to match records (e.g., `"id"` or `"delm_file_name"`)
- `record_sample_size`: Number of records to evaluate (`-1` = all)
- `save_file_log`, `log_dir`, `console_log_level`, `file_log_level`: Logging settings

**Returns:** Tuple of:
1. **Metrics dictionary** - Field-level metrics:
   ```python
   {
       "price": {
           "precision": 0.95,
           "recall": 0.90,
           "f1": 0.92,
           "tp": 38, "fp": 2, "fn": 4
       },
       "company": {...}
   }
   ```

2. **Comparison DataFrame** - Row-by-row comparisons:
   ```python
   columns: [matching_id_column, "expected_dict", "extracted_dict"]
   ```

**Warning:** Makes real API calls (costs apply).

## Example

```python
from delm import DELM, Schema, ExtractionVariable
from delm.utils.performance_estimation import estimate_performance
import pandas as pd

# Prepare ground truth data
expected_df = pd.DataFrame({
    "id": [1, 2, 3],
    "expected_extraction": [
        {"price": 10.5, "company": "Apple"},
        {"price": 20.0, "company": "Microsoft"},
        {"price": 15.0, "company": "Google"}
    ]
})

# Run performance estimation
delm = DELM(schema=schema, provider="openai", model="gpt-4o-mini")

metrics, comparison = estimate_performance(
    config=delm,
    data_source="data.csv",
    expected_extraction_output_df=expected_df,
    true_json_column="expected_extraction",
    matching_id_column="id"
)

# Analyze results
for field, scores in metrics.items():
    print(f"{field}: Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}")

# Inspect individual failures
comparison[comparison["extracted_dict"] != comparison["expected_dict"]]
```

## Matching Records

### For DataFrames with IDs

Use any existing ID column:

```python
matching_id_column="record_id"
```

### For Directories of Files

Use `delm_file_name` (automatically generated):

```python
expected_df = pd.DataFrame({
    "delm_file_name": ["doc1.pdf", "doc2.pdf"],
    "expected_extraction": [...]
})

estimate_performance(
    config=delm,
    data_source="pdfs/",  # Directory
    expected_extraction_output_df=expected_df,
    true_json_column="expected_extraction",
    matching_id_column="delm_file_name"  # Match by filename
)
```

