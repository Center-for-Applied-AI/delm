# Post-Processing

Transform and flatten extraction results.

## explode_json_results()

Flatten nested JSON extraction results into tabular format.

```python
from delm.utils.post_processing import explode_json_results

exploded_df = explode_json_results(
    input_df: pd.DataFrame,
    schema: Schema,
    json_column: str = "delm_extracted_data_json"
) -> pd.DataFrame
```

**Parameters:**
- `input_df`: DataFrame with JSON extraction results
- `schema`: Schema used for extraction
- `json_column`: Column containing JSON data

**Returns:** Exploded DataFrame where each extracted item gets its own row

## Behavior by Schema Type

### Simple Schema

Each record becomes one row with columns for each variable.

```python
# Input
delm_file_name | delm_extracted_data_json
"doc1.txt"     | {"company": "Apple", "price": 150.0}

# Output
delm_file_name | company  | price
"doc1.txt"     | "Apple"  | 150.0
```

### Nested Schema

Each item in the container list becomes its own row.

```python
# Input
delm_file_name | delm_extracted_data_json
"doc1.txt"     | {"products": [{"name": "Widget", "price": 10.0}, {"name": "Gadget", "price": 20.0}]}

# Output
delm_file_name | name      | price
"doc1.txt"     | "Widget"  | 10.0
"doc1.txt"     | "Gadget"  | 20.0
```

### Multiple Schema

Each sub-schema is exploded separately with `schema_name` column.

```python
# Input
delm_file_name | delm_extracted_data_json
"doc1.txt"     | {"products": [...], "companies": {...}}

# Output
delm_file_name | schema_name | name      | price
"doc1.txt"     | "products"  | "Widget"  | 10.0
"doc1.txt"     | "products"  | "Gadget"  | 20.0
"doc1.txt"     | "companies" | "Apple"   | None
```

## Missing Fields

Missing fields appear as `None`:

```python
# Input JSON: {"company": "Apple"}  (price missing)
# Output row: {"company": "Apple", "price": None}
```

## Example

```python
from delm import DELM, Schema, ExtractionVariable
from delm.utils.post_processing import explode_json_results

schema = Schema.nested(
    "products",
    ExtractionVariable("name", "Product name", "string"),
    ExtractionVariable("price", "Product price", "number")
)

delm = DELM(schema=schema, provider="openai", model="gpt-4o-mini")
results = delm.extract("data.csv")

# Flatten nested results
exploded = explode_json_results(results, schema)

# Now each product is a separate row
print(exploded[["delm_file_name", "name", "price"]])
```

## merge_jsons_for_record()

Merge multiple JSON extractions for the same record (used internally).

```python
from delm.utils.post_processing import merge_jsons_for_record

merged_json = merge_jsons_for_record(
    json_list: List[dict],
    schema: ExtractionSchema
) -> dict
```

**Merging rules:**
- **Scalars**: Majority vote (ties → first value)
- **Lists**: Concatenate all values

**Example:**

```python
jsons = [
    {"price": 10.0, "tags": ["tech"]},
    {"price": 10.0, "tags": ["gadgets"]},
    {"price": 15.0, "tags": ["home"]}
]

merged = merge_jsons_for_record(jsons, schema)
# {"price": 10.0, "tags": ["tech", "gadgets", "home"]}
# price: 10.0 wins (2 votes vs 1)
# tags: all concatenated
```

