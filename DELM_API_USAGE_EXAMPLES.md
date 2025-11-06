# DELM API Design: Usage Examples

**Purpose:** Define the desired API through concrete usage examples.

**Approach:** Show exactly how users should interact with DELM in various scenarios.

---

## 1. Simplest Possible Usage

### Example 1.1: Minimal Extraction

```python
from delm import DELM, variable, simple_schema

# Define what to extract
schema = simple_schema(
    variable("company", "Company name mentioned in text", "string"),
    variable("price", "Price value in USD", "number"),
)

# Initialize with just essentials
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
)

# Extract from DataFrame
import pandas as pd
df = pd.DataFrame({
    "text": [
        "Apple stock is $150.",
        "Microsoft revenue is $50B."
    ]
})

result = delm.extract(df)

# Access results
print(f"Extracted {result.num_records} records")
print(f"Cost: ${result.cost['total_cost']:.4f}")
print(result.data[["company", "price"]])
```

### Example 1.2: From Config File

```yaml
# config.yaml
provider: openai
model: gpt-4o-mini

schema:
  type: simple
  variables:
    - name: company
      description: "Company name mentioned"
      data_type: string
    - name: price
      description: "Price value in USD"
      data_type: number
```

```python
from delm import DELM

# One line to initialize
delm = DELM.from_config("config.yaml")

# One line to extract
result = delm.extract(df)

print(result)  # ExtractionResult(records=2, cost=$0.0023)
```

---

## 2. Schema Definitions

### Example 2.1: Simple Schema (Flat Key-Value)

```python
from delm import variable, simple_schema

# Helper function style
schema = simple_schema(
    variable("company", "Company name", "string"),
    variable("revenue", "Annual revenue", "number"),
    variable("sector", "Business sector", "string",
             allowed_values=["tech", "finance", "healthcare"]),
)
```

### Example 2.2: Nested Schema (List of Items)

```python
from delm import variable, nested_schema

# Extract multiple companies per text chunk
schema = nested_schema(
    "companies",  # Container name
    variable("name", "Company name", "string", required=True),
    variable("revenue", "Annual revenue in USD", "number"),
    variable("products", "List of products", "[string]"),
    variable("growth_rate", "Annual growth rate percentage", "number",
             validate_in_text=True),
)
```

### Example 2.3: Multiple Schemas (Different Types)

```python
from delm import variable, nested_schema, multiple_schema

# Define separate schemas
products_schema = nested_schema(
    "products",
    variable("name", "Product name", "string", required=True),
    variable("price", "Price in USD", "number"),
)

companies_schema = nested_schema(
    "companies",
    variable("name", "Company name", "string", required=True),
    variable("revenue", "Revenue", "number"),
)

# Combine them
schema = multiple_schema(
    products=products_schema,
    companies=companies_schema,
)
```

### Example 2.4: Pydantic Class (Full Control)

```python
from delm import Variable, NestedSchema

# Direct Pydantic usage for full type safety
schema = NestedSchema(
    type="nested",
    container_name="commodities",
    variables=[
        Variable(
            name="commodity_type",
            description="Type of commodity (oil, gas, gold, etc.)",
            data_type="string",
            required=True,
            allowed_values=["oil", "gas", "gold", "silver", "copper"],
            validate_in_text=True,
        ),
        Variable(
            name="price_value",
            description="Numeric price value if mentioned",
            data_type="number",
            required=False,
        ),
    ],
)
```

### Example 2.5: Schema from Dict

```python
# Quick prototyping with dicts
schema_dict = {
    "type": "simple",
    "variables": [
        {
            "name": "company",
            "description": "Company name",
            "data_type": "string",
        },
        {
            "name": "price",
            "description": "Price value",
            "data_type": "number",
        },
    ]
}

# Pass dict directly to DELM
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema_dict,  # Dict works too
)
```

### Example 2.6: Schema from YAML File

```python
# Option 1: Pass path to DELM
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema="schemas/my_schema.yaml",  # Path to file
)

# Option 2: Load explicitly first
from delm import SimpleSchema
schema = SimpleSchema.parse_file("schemas/my_schema.yaml")
delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
```

---

## 3. Initialization Patterns

### Example 3.1: With Common Parameters

```python
from delm import DELM, simple_schema, variable

schema = simple_schema(variable("name", "Name", "string"))

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=10,
    max_workers=4,
    max_budget=5.0,
    schema=schema,
    experiment="my_experiment",
)
```

### Example 3.2: With Text Processing

```python
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    
    # Text splitting (string shortcut)
    splitting="paragraph",  # or "sentence", "fixed-window"
    
    # Relevance scoring (keyword list)
    scoring=["price", "revenue", "forecast"],
    
    # Filter by score
    score_filter="delm_score > 0.5",
    
    # Which column to process
    target_column="text",
)
```

### Example 3.3: With Advanced Strategies

```python
from delm import DELM
from delm.strategies import FixedWindowSplit, FuzzyScorer

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    
    # Custom splitting strategy
    splitting=FixedWindowSplit(window=5, stride=2),
    
    # Custom scoring strategy
    scoring=FuzzyScorer(
        keywords=["price", "revenue"],
        threshold=0.8
    ),
)
```

### Example 3.4: From Config with Overrides

```python
# Load base config but override specific values
delm = DELM.from_config(
    "config.yaml",
    temperature=0.5,  # Override config value
    experiment="experiment_v2",  # Override experiment name
    max_budget=10.0,  # Override budget
)
```

---

## 4. Extraction Methods

### Example 4.1: Single-Step Extraction (Recommended)

```python
from delm import DELM
import pandas as pd

delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)

# Extract from DataFrame
df = pd.DataFrame({"text": ["Apple stock", "Microsoft revenue"]})
result = delm.extract(df)

# Result object has everything
print(result.data)           # DataFrame with extracted data
print(result.cost)           # Cost dictionary
print(result.num_records)    # Number of records
print(result.num_chunks)     # Number of chunks
print(result.num_errors)     # Number of errors
```

### Example 4.2: Extract from File

```python
# From parquet file
result = delm.extract("data/earnings_reports.parquet")

# From CSV file
result = delm.extract("data/articles.csv")

# From directory of files
from pathlib import Path
result = delm.extract(Path("data/text_files/"))
```

### Example 4.3: Extract with Sampling

```python
# Process only first 10 records (for testing)
result = delm.extract(df, sample_size=10)

print(f"Processed {result.num_records} records (sampled from {len(df)})")
```

### Example 4.4: Two-Step Processing (Advanced)

```python
# Step 1: Prep data (split, score, chunk)
prepped = delm.prep_data(df)

# Inspect before processing
print(prepped[["delm_chunk_id", "delm_score"]].head())
print(f"High score chunks: {len(prepped[prepped['delm_score'] > 0.7])}")

# Optionally filter
prepped_filtered = prepped[prepped["delm_score"] > 0.5]

# Step 2: Process through LLM
result_df = delm.process_via_llm()

# Get cost summary
cost = delm.get_cost_summary()
print(f"Total cost: ${cost['total_cost']:.4f}")
```

---

## 5. Working with Results

### Example 5.1: Accessing Result Data

```python
result = delm.extract(df)

# Result properties
print(result)  # ExtractionResult(records=10, chunks=25, errors=0, cost=$0.0045)

# Access DataFrame
extracted_df = result.data
print(extracted_df[["company", "price", "delm_record_id"]])

# Access cost details
if result.cost:
    print(f"Total: ${result.cost['total_cost']:.4f}")
    print(f"Input tokens: {result.cost['input_tokens']}")
    print(f"Output tokens: {result.cost['output_tokens']}")

# Check for errors
if result.num_errors > 0:
    errors = result.data[result.data["delm_errors"].notna()]
    print(f"Found {len(errors)} errors")
```

### Example 5.2: Working with Nested Results

```python
# Nested schema returns structured data
schema = nested_schema(
    "companies",
    variable("name", "Company name", "string"),
    variable("revenue", "Revenue", "number"),
)

delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
result = delm.extract(df)

# Access extracted data
# (Format depends on how nested data is stored - likely JSON column)
for _, row in result.data.iterrows():
    extracted = row["delm_extracted_data"]  # or JSON parsed version
    print(f"Found {len(extracted['companies'])} companies")
```

### Example 5.3: Saving Results

```python
result = delm.extract(df)

# Save to file
result.data.to_csv("results.csv", index=False)
result.data.to_parquet("results.parquet")

# Save cost report
import json
with open("cost_report.json", "w") as f:
    json.dump(result.cost, f, indent=2)
```

---

## 6. Complete Workflows

### Example 6.1: Basic Data Extraction Pipeline

```python
from delm import DELM, variable, simple_schema
import pandas as pd

# 1. Define schema
schema = simple_schema(
    variable("company", "Company name", "string"),
    variable("price", "Stock price", "number"),
    variable("date", "Date mentioned", "date"),
)

# 2. Initialize DELM
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    experiment="stock_extraction",
)

# 3. Load data
df = pd.read_parquet("earnings_calls.parquet")

# 4. Extract
result = delm.extract(df)

# 5. Analyze results
print(f"Extracted {result.num_records} records")
print(f"Cost: ${result.cost['total_cost']:.4f}")

# 6. Save
result.data.to_csv("extracted_stock_data.csv")
```

### Example 6.2: Multi-Schema Extraction

```python
from delm import DELM, variable, nested_schema, multiple_schema

# Define schemas for different entity types
products = nested_schema(
    "products",
    variable("name", "Product name", "string", required=True),
    variable("price", "Price", "number"),
    variable("category", "Category", "string"),
)

companies = nested_schema(
    "companies",
    variable("name", "Company name", "string", required=True),
    variable("revenue", "Revenue", "number"),
)

# Combine
schema = multiple_schema(products=products, companies=companies)

# Extract
delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
result = delm.extract(df)

# Access different schemas
print(result.data["products"])
print(result.data["companies"])
```

### Example 6.3: Relevance Filtering Workflow

```python
from delm import DELM, variable, simple_schema

schema = simple_schema(
    variable("price", "Price mentioned", "number"),
    variable("commodity", "Commodity type", "string"),
)

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    splitting="paragraph",
    scoring=["price", "cost", "value", "commodity"],
    score_filter="delm_score > 0.5",  # Only process relevant chunks
)

result = delm.extract(df)
print(f"Processed only high-relevance chunks")
```

### Example 6.4: Budget-Constrained Extraction

```python
from delm import DELM, variable, simple_schema

schema = simple_schema(variable("summary", "Summary", "string"))

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    max_budget=1.0,  # Stop at $1
    batch_size=5,     # Small batches
)

try:
    result = delm.extract(large_df)
except BudgetExceededError:
    print("Budget exceeded, partial results available")
    result = delm.get_extraction_results()
```

### Example 6.5: Team Configuration Workflow

```yaml
# team_config.yaml (in version control)
provider: openai
model: gpt-4o-mini
temperature: 0.0
batch_size: 10
max_workers: 4

splitting: paragraph
scoring: [price, revenue, forecast, guidance]

schema:
  type: nested
  container_name: financial_metrics
  variables:
    - name: metric_name
      description: "Name of financial metric"
      data_type: string
      required: true
    - name: value
      description: "Numeric value"
      data_type: number
    - name: period
      description: "Time period"
      data_type: string

experiment:
  name: financial_extraction
  directory: ./experiments
```

```python
# Everyone on team uses same config
from delm import DELM

delm = DELM.from_config("team_config.yaml")
result = delm.extract(df)

# Override for specific experiments
delm_test = DELM.from_config(
    "team_config.yaml",
    experiment="test_run_v2",
    temperature=0.5,
    sample_size=10,
)
```

---

## 7. Advanced Usage

### Example 7.1: Custom Prompt Templates

```python
from delm import DELM, variable, simple_schema

schema = simple_schema(variable("sentiment", "Sentiment", "string"))

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    prompt_template="""
    Analyze the sentiment of this text carefully.
    
    Extract: {variables}
    
    Text to analyze:
    {text}
    """,
    system_prompt="You are a sentiment analysis expert.",
)

result = delm.extract(df)
```

### Example 7.2: Multiple Providers

```python
# Test with different providers
providers_config = [
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "anthropic", "model": "claude-3-sonnet"},
    {"provider": "google", "model": "gemini-1.5-flash"},
]

results = {}
for config in providers_config:
    delm = DELM(schema=schema, **config)
    results[config["provider"]] = delm.extract(df)
    
# Compare results and costs
for provider, result in results.items():
    print(f"{provider}: {result.num_records} records, ${result.cost['total_cost']:.4f}")
```

### Example 7.3: Programmatic Schema Generation

```python
from delm import Variable, SimpleSchema, DELM

def create_schema_for_fields(field_specs):
    """Generate schema from field specifications."""
    variables = [
        Variable(
            name=name,
            description=spec["description"],
            data_type=spec.get("type", "string"),
            required=spec.get("required", False),
        )
        for name, spec in field_specs.items()
    ]
    return SimpleSchema(type="simple", variables=variables)

# Define fields programmatically
fields = {
    "product": {"description": "Product name", "type": "string", "required": True},
    "price": {"description": "Price in USD", "type": "number"},
    "quantity": {"description": "Quantity", "type": "integer"},
}

schema = create_schema_for_fields(fields)
delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
```

---

## 8. Backward Compatibility

### Example 8.1: Old API Still Works

```python
from delm import DELM, DELMConfig
from pathlib import Path

# Old way with config object
config = DELMConfig.from_yaml("old_config.yaml")

delm = DELM(
    config=config,
    experiment="test",
    experiment_dir=Path("./experiments"),
    overwrite=True,
    auto_checkpoint=True,
    use_disk_storage=True,
)

# Old two-step workflow
delm.prep_data(df)
result_df = delm.process_via_llm()
cost = delm.get_cost_summary()
```

### Example 8.2: Mixed Old and New

```python
# Use old config with new methods
config = DELMConfig.from_yaml("old_config.yaml")
delm = DELM(config=config, experiment="test")

# But use new extract() method
result = delm.extract(df)  # New method works!
```

---

## 9. Common Patterns

### Pattern 1: Quick Test with Sample

```python
# Quick test on sample data
delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
test_result = delm.extract(df, sample_size=5)

print(f"Test cost: ${test_result.cost['total_cost']:.4f}")
print("Looks good, running on full dataset...")

# Full run
full_result = delm.extract(df)
```

### Pattern 2: Inspect and Filter

```python
# Prep and inspect
delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema, scoring=["price"])
prepped = delm.prep_data(df)

# Check scores
print(prepped["delm_score"].describe())

# Filter low scores
high_quality = prepped[prepped["delm_score"] > 0.7]
print(f"Processing {len(high_quality)} high-quality chunks")

# Process
result_df = delm.process_via_llm()
```

### Pattern 3: Iterative Development

```python
# Start simple
simple = simple_schema(variable("company", "Company name", "string"))
delm = DELM(provider="openai", model="gpt-4o-mini", schema=simple)
result = delm.extract(df.head(10))  # Test on small sample

# Add complexity
detailed = simple_schema(
    variable("company", "Company name", "string"),
    variable("revenue", "Revenue", "number"),
    variable("sector", "Sector", "string", allowed_values=["tech", "finance"]),
)
delm = DELM(provider="openai", model="gpt-4o-mini", schema=detailed)
result = delm.extract(df.head(10))  # Test again

# Scale up
result = delm.extract(df)  # Full dataset
```

---

## Summary: The API We Want

**Simple things are simple:**
```python
delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
result = delm.extract(df)
```

**Complex things are possible:**
```python
delm = DELM(
    provider="anthropic",
    model="claude-3-sonnet",
    temperature=0.1,
    max_budget=10.0,
    schema=complex_schema,
    splitting=FixedWindowSplit(window=5, stride=2),
    scoring=FuzzyScorer(keywords=["key"], threshold=0.8),
    score_filter="delm_score > 0.6",
)
result = delm.extract(df)
```

**Everything is type-safe and validated:**
```python
# IDE autocomplete works
# Type errors caught before runtime
# Clear validation errors
```

**Old code keeps working:**
```python
config = DELMConfig.from_yaml("old_config.yaml")
delm = DELM(config=config, experiment="test")
delm.prep_data(df)
result_df = delm.process_via_llm()
```

This is the API we're building toward! 🎯
