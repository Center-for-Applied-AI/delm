# Performance Evaluation

Measure extraction quality by comparing results against human-labeled data. Get precision, recall, and F1 scores for each field in your schema.

## What is Performance Evaluation?

`estimate_performance()` runs your extraction pipeline on sample data and compares the results to expected outputs. It calculates field-level metrics to show you where your extraction is accurate and where it needs improvement.

**Warning**: This function makes API calls and will incur costs based on the sample size you specify.

## Data Requirements

You need two datasets:

1. **Input data**: The raw text data (CSV, parquet, DataFrame, directory of files, etc.)
2. **Expected output data**: A DataFrame with human-labeled extraction results

Both datasets must have a **matching ID column** to link input records to their expected outputs.

### Matching ID for Different Input Types

- **DataFrame/CSV/Parquet**: Use any existing ID column in your data (e.g., `id`, `report_id`, `doc_id`)
- **Directory of files (PDFs, text files, etc.)**: Use `delm_file_name` - this column is automatically created with the filename for each file loaded
- **Single file**: Not typically used for evaluation (since you'd only have one record)

### Expected Output Format

Your expected output DataFrame needs:
- **Matching ID column**: Links to input data
  - For DataFrames/CSV: any ID column (e.g., `id`, `report_id`)
  - For file directories: a column with filenames that will match `delm_file_name`
- **Expected results column**: Contains the correct extraction as a dict or JSON string

The expected results must match your schema structure:

```python
# For a simple schema
expected_dict = {
    "price": 75.0,
    "currency": "USD",
    "commodity": "oil"
}

# For a nested schema
expected_dict = {
    "prices": [
        {"price": 75.0, "currency": "USD"},
        {"price": 80.0, "currency": "EUR"}
    ]
}
```

## Complete Examples

### Example 1: DataFrame Input

```python
import pandas as pd
from delm import DELM, Schema, ExtractionVariable
from delm.utils.performance_estimation import estimate_performance

# 1. Load input data (raw text)
input_df = pd.read_csv("data/raw_texts.csv")
# Columns: id, text

# 2. Load expected output data (human labels)
expected_df = pd.read_csv("data/human_labels.csv")
# Columns: id, expected_extraction (as JSON string or dict)

# 3. Define your schema
schema = Schema.simple([
    ExtractionVariable(
        name="price",
        description="Price value mentioned",
        data_type="number"
    ),
    ExtractionVariable(
        name="currency",
        description="Currency (USD, EUR, etc.)",
        data_type="string"
    ),
    ExtractionVariable(
        name="commodity",
        description="Commodity type (oil, gold, etc.)",
        data_type="string"
    )
])

# 4. Create DELM configuration
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0
)

# 5. Run performance evaluation
metrics, comparison_df = estimate_performance(
    config=delm,  # Can also pass DELMConfig, dict, or YAML path
    data_source=input_df,
    expected_extraction_output_df=expected_df,
    true_json_column="expected_extraction",
    matching_id_column="id",
    record_sample_size=50  # Process 50 records (-1 for all)
)

# 6. Display results
print(f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)
for field, m in metrics.items():
    print(f"{field:<20} {m['precision']:10.3f} {m['recall']:10.3f} {m['f1']:10.3f}")
```

### Example 2: Directory of PDFs

```python
import pandas as pd
from delm import DELM, Schema, ExtractionVariable
from delm.utils.performance_estimation import estimate_performance

# 1. Prepare expected output data with filenames
# Your expected_df must have a column with the filename for matching
expected_df = pd.DataFrame({
    'filename': ['report1.pdf', 'report2.pdf', 'report3.pdf'],
    'expected_extraction': [
        {"price": 75.0, "currency": "USD", "commodity": "oil"},
        {"price": 1950.0, "currency": "USD", "commodity": "gold"},
        {"price": 3.50, "currency": "USD", "commodity": "gas"}
    ]
})

# 2. Define schema and DELM config
schema = Schema.simple([
    ExtractionVariable(name="price", description="Price value", data_type="number"),
    ExtractionVariable(name="currency", description="Currency", data_type="string"),
    ExtractionVariable(name="commodity", description="Commodity type", data_type="string")
])

delm = DELM(schema=schema, provider="openai", model="gpt-4o-mini", temperature=0.0)

# 3. Run evaluation on directory
# The system creates a 'delm_file_name' column automatically for each PDF
metrics, comparison_df = estimate_performance(
    config=delm,
    data_source="data/pdfs/",  # Directory of PDF files
    expected_extraction_output_df=expected_df,
    true_json_column="expected_extraction",
    matching_id_column="delm_file_name",  # Use the auto-generated filename column
    record_sample_size=-1  # Process all files
)

# 4. Display results
for field, m in metrics.items():
    print(f"{field:<20} {m['precision']:10.3f} {m['recall']:10.3f} {m['f1']:10.3f}")
```

**Important**: When using a directory of files, your `expected_extraction_output_df` must have a column that matches the filenames in the directory. Use `delm_file_name` as the `matching_id_column`.

### Output Example

```
Field                Precision     Recall         F1
----------------------------------------------------
price                    0.950      0.900      0.924
currency                 0.875      0.933      0.903
commodity                1.000      0.850      0.919
```

## Understanding Metrics

**Precision**: Of the items your pipeline extracted, what percentage were correct?
- Formula: `TP / (TP + FP)`
- High precision = few false positives (didn't extract things that shouldn't be there)

**Recall**: Of the correct items that should be extracted, what percentage did your pipeline find?
- Formula: `TP / (TP + FN)`
- High recall = few false negatives (didn't miss things that should be extracted)

**F1 Score**: Harmonic mean of precision and recall
- Formula: `2 * (Precision × Recall) / (Precision + Recall)`
- Balanced measure of overall extraction quality

## Analyzing Results

### Inspect the Comparison DataFrame

The `comparison_df` returned by `estimate_performance()` contains the expected vs. extracted results for each record:

```python
# View columns
print(comparison_df.columns)
# Output: ['id', 'expected_dict', 'extracted_dict']

# Examine first few results
print(comparison_df.head())

# Find discrepancies
for idx, row in comparison_df.iterrows():
    if row['expected_dict'] != row['extracted_dict']:
        print(f"Record {row['id']}:")
        print(f"  Expected: {row['expected_dict']}")
        print(f"  Extracted: {row['extracted_dict']}")
```

### Metrics Dictionary Structure

Each field in your schema has its own metrics:

```python
# Example metrics structure
{
    "price": {
        "precision": 0.95,
        "recall": 0.90,
        "f1": 0.924,
        "tp": 45,   # True positives
        "fp": 3,    # False positives
        "fn": 5     # False negatives
    },
    "currency": {
        "precision": 0.875,
        "recall": 0.933,
        "f1": 0.903,
        "tp": 42,
        "fp": 6,
        "fn": 3
    }
}
```
