# Schema Reference

Schemas define the structured outputs that DELM extracts from your documents. The schema system supports progressive complexity levels, from simple key‑value extraction to complex nested structures.

## Table of Contents

- [Simple Schema (Level 1)](#simple-schema-level-1)
- [Nested Schema (Level 2)](#nested-schema-level-2)
- [Multiple Schemas (Level 3)](#multiple-schemas-level-3)
- [Variable Configuration](#variable-configuration)

## Imports

All schema classes are available directly from the main package:

```python
from delm import Schema, ExtractionVariable
```

## Schema Types

DELM supports three levels of schema complexity, each building on the previous level.

### Simple Schema (Level 1)

The simplest form of extraction: individual key‑value pairs found once per chunk.

```python
schema = Schema.simple(
    ExtractionVariable(
        name="company_names",
        description="Company names mentioned in the text",
        data_type="[string]",
        required=False
    ),
    ExtractionVariable(
        name="revenue_numbers",
        description="Revenue figures mentioned",
        data_type="[number]",
        required=False
    ),
    ExtractionVariable(
        name="forecast_year",
        description="Year for which forecast is made",
        data_type="integer",
        required=True,
        validate_in_text=True
    )
)
```

Output Format:
```json
{
  "company_names": ["Apple", "Microsoft"],
  "revenue_numbers": [1500000000, 2000000000],
  "forecast_year": 2024
}
```

### Nested Schema (Level 2)

Extract structured objects with multiple related fields (a list of dictionaries).

```python
schema = Schema.nested(
    container_name="companies",
    variables_list=[
        ExtractionVariable(
            name="name",
            description="Company name",
            data_type="string",
            required=True
        ),
        ExtractionVariable(
            name="revenue",
            description="Revenue figure in USD",
            data_type="number",
            required=False
        ),
        ExtractionVariable(
            name="sector",
            description="Business sector",
            data_type="string",
            required=False,
            allowed_values=["technology", "finance", "healthcare", "energy", "retail"]
        ),
        ExtractionVariable(
            name="growth_rate",
            description="Annual growth rate percentage",
            data_type="number",
            required=False,
            validate_in_text=True  # Only extract if explicitly mentioned
        ),
        ExtractionVariable(
            name="products",
            description="List of products offered by the company",
            data_type="[string]",
            required=False
        )
    ]
)
```

Output Format:
```json
{
  "companies": [
    {
      "name": "Apple",
      "revenue": 1500000000,
      "sector": "technology",
      "growth_rate": 12.5,
      "products": ["iPhone", "MacBook", "iPad"]
    },
    {
      "name": "Microsoft",
      "revenue": 2000000000,
      "sector": "technology",
      "growth_rate": null,
      "products": ["Windows", "Office", "Azure"]
    }
  ]
}
```

### Multiple Schemas (Level 3)

Extract multiple independent structured objects simultaneously. These can be simple, nested, or even deep multi‑schemas.

```python
# Define sub-schemas first
companies_schema = Schema.nested(
    container_name="companies",
    variables_list=[
        ExtractionVariable(name="name", description="Company name", data_type="string", required=True),
        ExtractionVariable(name="revenue", description="Revenue figure", data_type="number", required=False)
    ]
)

products_schema = Schema.nested(
    container_name="products",
    variables_list=[
        ExtractionVariable(name="name", description="Product name", data_type="string", required=True),
        ExtractionVariable(name="price", description="Product price in USD", data_type="number", required=False),
        ExtractionVariable(
            name="category", 
            description="Product category", 
            data_type="string", 
            allowed_values=["software", "hardware", "service", "consulting"]
        )
    ]
)

trends_schema = Schema.nested(
    container_name="trends",
    variables_list=[
        ExtractionVariable(name="trend_name", description="Market trend description", data_type="string", required=True),
        ExtractionVariable(
            name="impact", 
            description="Expected impact", 
            data_type="string", 
            allowed_values=["positive", "negative", "neutral"]
        )
    ]
)

# Combine into multiple schema
schema = Schema.multiple(
    companies=companies_schema,
    products=products_schema,
    market_trends=trends_schema
)
```

Output Format:
```json
{
  "companies": [
    { "name": "Apple", "revenue": 1500000000 }
  ],
  "products": [
    { "name": "iPhone 15", "price": 999, "category": "hardware" }
  ],
  "trends": [
    { "trend_name": "AI adoption acceleration", "impact": "positive" }
  ]
}
```

## Variable Configuration

Each `ExtractionVariable` can be configured with these arguments.

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `name` | string | Variable name (used as JSON key) |
| `description` | string | Human‑readable description for LLM |
| `data_type` | string | Data type (see supported types below) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `required` | boolean | `False` | Whether field must be present |
| `allowed_values` | list | `None` | List of valid string values (enums) |
| `validate_in_text` | boolean | `False` | Only extract if value literally appears in text |

### Supported Data Types

| Type String | Description | Example Values |
|-------------|-------------|----------------|
| `"string"` | Text values | "Apple", "technology" |
| `"number"` | Floating‑point numbers | 1500000000, 12.5 |
| `"integer"` | Whole numbers | 2024, 100 |
| `"boolean"` | True/false values | `True`, `False` |
| `"date"` | Date strings | "2025-09-15" |
| `"[string]"` | List of strings | ["Apple", "Google"] |
| `"[number]"` | List of numbers | [12.5, 42, 100] |
| `"[integer]"` | List of integers | [2024, 100, 7] |
| `"[boolean]"` | List of booleans | [True, False, True] |