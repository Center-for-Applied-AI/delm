# Post-Processing

Learn how to transform DELM extraction results into tabular format using the `explode_json_results` utility.

## Overview

DELM extracts structured JSON data, but you often need tabular format for analysis. The `explode_json_results` utility converts nested JSON into flat, analyzable tables.

## Basic Usage

### Simple Explosion

```python
from delm.utils.post_processing import explode_json_results

# Convert JSON results to tabular format
exploded_df = explode_json_results(
    final_df,
    schema="schema_spec.yaml"  # Path to your schema file
)

print(exploded_df.head())
```

### Using Schema Object

```python
from delm import DELM

# Load schema from pipeline
pipeline = DELM.from_yaml("config.yaml", "experiment", Path("experiments"))
schema = pipeline.schema_manager.get_extraction_schema()

# Use schema object instead of file path
exploded_df = explode_json_results(
    final_df,
    schema=schema
)
```

## Schema Type Handling

### Simple Schema Explosion

For simple schemas (key-value pairs):

**Input JSON**:
```json
{"price": 100, "company": "Apple"}
```

**Output Table**:
```
| delm_chunk_id | price | company |
|---------------|-------|---------|
| chunk_1       | 100   | Apple   |
```

### Nested Schema Explosion

For nested schemas (list of objects):

**Input JSON**:
```json
{
  "commodities": [
    {"type": "oil", "price": 75},
    {"type": "gold", "price": 1950}
  ]
}
```

**Output Table**:
```
| delm_chunk_id | commodity_type | commodity_price |
|---------------|----------------|-----------------|
| chunk_1       | oil            | 75              |
| chunk_1       | gold           | 1950            |
```

### Multiple Schema Explosion

For multiple schemas (multiple independent lists):

**Input JSON**:
```json
{
  "commodities": [{"type": "oil", "price": 75}],
  "companies": [{"name": "Exxon", "sector": "energy"}]
}
```

**Output Tables**:
```
# commodities table
| delm_chunk_id | commodity_type | commodity_price |
|---------------|----------------|-----------------|
| chunk_1       | oil            | 75              |

# companies table  
| delm_chunk_id | company_name | company_sector |
|---------------|--------------|----------------|
| chunk_1       | Exxon        | energy         |
```

## Advanced Configuration

### Custom Column Names

```python
# Specify custom column name mappings
exploded_df = explode_json_results(
    final_df,
    schema="schema_spec.yaml",
    column_mapping={
        "commodity_type": "type",
        "price_value": "price"
    }
)
```

### Filtering Results

```python
# Filter out null values
exploded_df = explode_json_results(
    final_df,
    schema="schema_spec.yaml",
    drop_null=True  # Remove rows with null values
)
```

### Handling Missing Data

```python
# Keep null values but mark them
exploded_df = explode_json_results(
    final_df,
    schema="schema_spec.yaml",
    null_value="MISSING"  # Replace null with custom value
)
```

## Data Analysis Examples

### Basic Analysis

```python
# Load and explode results
results = pipeline.get_extraction_results()
exploded_df = explode_json_results(results, schema="schema_spec.yaml")

# Basic statistics
print(f"Total extractions: {len(exploded_df)}")
print(f"Unique commodities: {exploded_df['commodity_type'].nunique()}")
print(f"Average price: ${exploded_df['price_value'].mean():.2f}")
```

### Grouped Analysis

```python
# Group by commodity type
commodity_stats = exploded_df.groupby('commodity_type').agg({
    'price_value': ['count', 'mean', 'std'],
    'delm_chunk_id': 'nunique'
}).round(2)

print("Commodity Statistics:")
print(commodity_stats)
```

### Time Series Analysis

```python
# If you have timestamp data
exploded_df['extraction_date'] = pd.to_datetime(exploded_df['delm_chunk_id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])

# Daily price trends
daily_prices = exploded_df.groupby(['extraction_date', 'commodity_type'])['price_value'].mean().unstack()
daily_prices.plot(kind='line', title='Daily Commodity Prices')
```

## Integration with Analysis Tools

### Pandas Integration

```python
import pandas as pd

# Convert to pandas for analysis
df = exploded_df.copy()

# Advanced filtering
oil_prices = df[df['commodity_type'] == 'oil']['price_value']
print(f"Oil price range: ${oil_prices.min():.2f} - ${oil_prices.max():.2f}")

# Statistical analysis
price_correlation = df[['price_value', 'delm_chunk_id']].corr()
print("Price correlation matrix:")
print(price_correlation)
```

### Export to Other Formats

```python
# Export to CSV
exploded_df.to_csv("extracted_data.csv", index=False)

# Export to Excel with multiple sheets
with pd.ExcelWriter("extracted_data.xlsx") as writer:
    exploded_df.to_excel(writer, sheet_name="All Data", index=False)
    
    # Create summary sheet
    summary = exploded_df.groupby('commodity_type')['price_value'].agg(['count', 'mean', 'std'])
    summary.to_excel(writer, sheet_name="Summary")
```

### Database Integration

```python
import sqlite3

# Save to SQLite database
conn = sqlite3.connect("extractions.db")
exploded_df.to_sql("extractions", conn, if_exists="replace", index=False)

# Query the database
query = """
SELECT commodity_type, AVG(price_value) as avg_price, COUNT(*) as count
FROM extractions 
GROUP BY commodity_type
"""
summary_df = pd.read_sql_query(query, conn)
print(summary_df)
```

## Schema-Specific Examples

### Financial Data Extraction

```python
# Schema: financial metrics
schema = {
    "schema_type": "nested",
    "container_name": "metrics",
    "variables": [
        {"name": "metric_name", "data_type": "string"},
        {"name": "value", "data_type": "number"},
        {"name": "currency", "data_type": "string"}
    ]
}

# Explode and analyze
exploded_df = explode_json_results(results, schema=schema)

# Financial analysis
revenue_metrics = exploded_df[exploded_df['metric_name'].str.contains('revenue', case=False)]
print(f"Revenue metrics found: {len(revenue_metrics)}")

# Currency breakdown
currency_dist = exploded_df['currency'].value_counts()
print("Currency distribution:")
print(currency_dist)
```

### Sentiment Analysis

```python
# Schema: sentiment analysis
schema = {
    "schema_type": "nested", 
    "container_name": "sentiments",
    "variables": [
        {"name": "aspect", "data_type": "string"},
        {"name": "sentiment", "data_type": "string"},
        {"name": "intensity", "data_type": "string"}
    ]
}

# Explode and analyze sentiment
exploded_df = explode_json_results(results, schema=schema)

# Sentiment distribution
sentiment_dist = exploded_df['sentiment'].value_counts()
print("Sentiment distribution:")
print(sentiment_dist)

# Aspect analysis
aspect_sentiment = exploded_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
print("Aspect-Sentiment Matrix:")
print(aspect_sentiment)
```

## Performance Optimization

### Large Dataset Handling

```python
# Process large datasets in chunks
def process_large_dataset(pipeline, chunk_size=1000):
    results = pipeline.get_extraction_results()
    exploded_chunks = []
    
    for i in range(0, len(results), chunk_size):
        chunk = results.iloc[i:i+chunk_size]
        exploded_chunk = explode_json_results(chunk, schema="schema_spec.yaml")
        exploded_chunks.append(exploded_chunk)
    
    return pd.concat(exploded_chunks, ignore_index=True)
```

### Memory Optimization

```python
# Optimize memory usage
def memory_efficient_explosion(df, schema):
    # Process in smaller batches
    batch_size = 100
    exploded_batches = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        exploded_batch = explode_json_results(batch, schema=schema)
        exploded_batches.append(exploded_batch)
        
        # Clear memory
        del batch, exploded_batch
    
    return pd.concat(exploded_batches, ignore_index=True)
```

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.


## Next Steps

- [Schema Design](../configuration/schema-design.md) - Learn advanced schema patterns
- [Batch Processing](batch-processing.md) - Optimize performance with batching
- [Cost Tracking](cost-tracking.md) - Monitor costs and budget limits
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
