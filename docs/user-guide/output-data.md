# Output Data

Understand the structure of DELM's extraction results and how to transform them for analysis.

## Output Columns

### From `delm.extract()`

When you call `delm.extract()`, you get a DataFrame with:

1. **Your original columns** (from input data)
2. **DELM system columns** (added during processing):

| Column | Description |
|--------|-------------|
| `delm_chunk_id` | Unique ID for each text chunk processed |
| `delm_record_id` | Links chunks back to original records |
| `delm_text_chunk` | The actual text chunk sent to the LLM |
| `delm_score` | Relevance score (if scorer was used) |
| `delm_batch_id` | Batch number for processing |
| `delm_errors` | Error messages (if extraction failed) |
| `delm_extracted_data_json` | JSON string of extracted data |

**Example:**

```python
results_df = delm.extract("data.csv")
print(results_df.columns)
# Output: ['id', 'company', 'text', 'delm_chunk_id', 'delm_record_id', 
#          'delm_text_chunk', 'delm_score', 'delm_batch_id', 'delm_errors',
#          'delm_extracted_data_json']
```

### From `delm.get_extraction_results()`

This method only returns the **core extraction columns** (no original data, no `delm_record_id`, no `delm_score`):

- `delm_chunk_id`
- `delm_batch_id`
- `delm_text_chunk`
- `delm_errors`
- `delm_extracted_data_json`

**Use case**: When you've saved results to disk with `use_disk_storage=True` and want to reload just the extraction data later.

**Note**: `delm_record_id` and `delm_score` are metadata that are merged in after loading, so they're only available from `extract()`, not from `get_extraction_results()`.

```python
delm = DELM(
    schema=schema,
    use_disk_storage=True,
    experiment_path="experiments/my_run"
)

# Run extraction
results_df = delm.extract("data.csv")  # Returns all columns

# Later, reload just extraction data
extraction_only = delm.get_extraction_results()  # Returns only DELM columns
```

## Transforming Results with `explode_json_results()`

The `explode_json_results()` function converts nested JSON into flat, tabular format for analysis. How it works depends on your schema type.

### Simple Schema

For simple schemas, each row represents one chunk with all extracted fields as columns.

```python
from delm import DELM, Schema, ExtractionVariable
from delm.utils.post_processing import explode_json_results

# Define simple schema
schema = Schema.simple([
    ExtractionVariable(name="company", data_type="string"),
    ExtractionVariable(name="price", data_type="number"),
    ExtractionVariable(name="currency", data_type="string")
])

delm = DELM(schema=schema, provider="openai", model="gpt-4o-mini")
results = delm.extract("data.csv")

# Explode JSON
exploded = explode_json_results(results, schema)
```

**Input JSON** (in `delm_extracted_data_json`):
```json
{"company": "Apple", "price": 150, "currency": "USD"}
```

**Output Table:**

| delm_chunk_id | company | price | currency |
|---------------|---------|-------|----------|
| 0             | Apple   | 150   | USD      |
| 1             | Google  | 2800  | USD      |

### Nested Schema

For nested schemas, each item in the list becomes its own row. Multiple items from the same chunk will create multiple rows.

```python
schema = Schema.nested(
    container_name="commodities",
    variables_list=[
        ExtractionVariable(name="commodity", data_type="string"),
        ExtractionVariable(name="price", data_type="number"),
        ExtractionVariable(name="unit", data_type="string")
    ]
)

delm = DELM(schema=schema, provider="openai", model="gpt-4o-mini")
results = delm.extract("data.csv")

# Explode JSON
exploded = explode_json_results(results, schema)
```

**Input JSON** (in `delm_extracted_data_json`):
```json
{
  "commodities": [
    {"commodity": "oil", "price": 75, "unit": "barrel"},
    {"commodity": "gold", "price": 1950, "unit": "ounce"}
  ]
}
```

**Output Table:**

| delm_chunk_id | commodity | price | unit   |
|---------------|-----------|-------|--------|
| 0             | oil       | 75    | barrel |
| 0             | gold      | 1950  | ounce  |
| 1             | silver    | 24    | ounce  |

**Note**: Both "oil" and "gold" have the same `delm_chunk_id` (0) because they came from the same chunk.

### Multiple Schema

For multiple schemas, each sub-schema is exploded separately, and a `schema_name` column identifies which schema each row belongs to.

```python
schema = Schema.multiple({
    "commodities": Schema.nested(
        container_name="items",
        variables_list=[
            ExtractionVariable(name="name", data_type="string"),
            ExtractionVariable(name="price", data_type="number")
        ]
    ),
    "companies": Schema.nested(
        container_name="items",
        variables_list=[
            ExtractionVariable(name="name", data_type="string"),
            ExtractionVariable(name="sector", data_type="string")
        ]
    )
})

delm = DELM(schema=schema, provider="openai", model="gpt-4o-mini")
results = delm.extract("data.csv")

# Explode JSON
exploded = explode_json_results(results, schema)
```

**Input JSON** (in `delm_extracted_data_json`):
```json
{
  "commodities": [{"name": "oil", "price": 75}],
  "companies": [{"name": "Exxon", "sector": "energy"}]
}
```

**Output Table:**

| delm_chunk_id | schema_name  | name  | price | sector |
|---------------|--------------|-------|-------|--------|
| 0             | commodities  | oil   | 75    | None   |
| 0             | companies    | Exxon | None  | energy |
| 1             | commodities  | gold  | 1950  | None   |
| 1             | companies    | Shell | None  | energy |

**Note**: Fields that don't exist in a schema are filled with `None` (e.g., "sector" is `None` for commodities rows).
