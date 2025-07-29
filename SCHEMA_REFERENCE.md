# DELM Schema Reference

This document provides a comprehensive guide to defining extraction schemas in DELM. The schema system supports progressive complexity levels, from simple key-value extraction to complex nested structures.

## Table of Contents

- [Schema Types](#schema-types)
  - [Simple Schema (Level 1)](#simple-schema-level-1)
  - [Nested Schema (Level 2)](#nested-schema-level-2)
  - [Multiple Schemas (Level 3)](#multiple-schemas-level-3)
- [Variable Configuration](#variable-configuration)
- [Prompt Customization](#prompt-customization)
- [Schema Examples](#schema-examples)

## Schema Types

DELM supports three levels of schema complexity, each building on the previous level.

### Simple Schema (Level 1)

The simplest form of extraction - individual key-value pairs.

```yaml
variables:
  - name: "company_names"
    description: "Company names mentioned in the text"
    data_type: "[string]"
    required: false
  
  - name: "revenue_numbers"
    description: "Revenue figures mentioned"
    data_type: "[number]"
    required: false
  
  - name: "forecast_year"
    description: "Year for which forecast is made"
    data_type: "integer"
    required: true
    validate_in_text: true
```

**Output Format:**
```json
{
  "company_names": ["Apple", "Microsoft"],
  "revenue_numbers": [1500000000, 2000000000],
  "forecast_year": 2024
}
```

### Nested Schema (Level 2)

Extract structured objects with multiple related fields.

```yaml
schema_type: "nested"
container_name: "companies"
variables:
  - name: "name"
    description: "Company name"
    data_type: "string"
    required: true
  
  - name: "revenue"
    description: "Revenue figure in USD"
    data_type: "number"
    required: false
  
  - name: "sector"
    description: "Business sector"
    data_type: "string"
    required: false
    allowed_values: ["technology", "finance", "healthcare", "energy", "retail"]
  
  - name: "growth_rate"
    description: "Annual growth rate percentage"
    data_type: "number"
    required: false
    validate_in_text: true  # Only extract if explicitly mentioned
  
  - name: "products"
    description: "List of products offered by the company"
    data_type: "[string]"
    required: false
```

**Output Format:**
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

Extract multiple independent structured objects simultaneously. These can be simple, nested, or even deep mutli-schemas.

```yaml
schema_type: "multiple"

# Companies schema
companies:
  schema_type: "nested"
  container_name: "companies"
  variables:
    - name: "name"
      description: "Company name"
      data_type: "string"
      required: true
    - name: "revenue"
      description: "Revenue figure"
      data_type: "number"
      required: false

# Products schema
products:
  schema_type: "nested"
  container_name: "products"
  variables:
    - name: "name"
      description: "Product name"
      data_type: "string"
      required: true
    - name: "price"
      description: "Product price in USD"
      data_type: "number"
      required: false
    - name: "category"
      description: "Product category"
      data_type: "string"
      allowed_values: ["software", "hardware", "service", "consulting"]
      required: false

# Market trends schema
market_trends:
  schema_type: "nested"
  container_name: "trends"
  variables:
    - name: "trend_name"
      description: "Market trend description"
      data_type: "string"
      required: true
    - name: "impact"
      description: "Expected impact (positive/negative/neutral)"
      data_type: "string"
      allowed_values: ["positive", "negative", "neutral"]
      required: false
```

**Output Format:**
```json
{
  "companies": [
    {
      "name": "Apple",
      "revenue": 1500000000
    }
  ],
  "products": [
    {
      "name": "iPhone 15",
      "price": 999,
      "category": "hardware"
    }
  ],
  "trends": [
    {
      "trend_name": "AI adoption acceleration",
      "impact": "positive"
    }
  ]
}
```

## Variable Configuration

Each variable in your schema can be configured with these options:

### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Variable name (used as JSON key) |
| `description` | string | Yes | Human-readable description for LLM |
| `data_type` | string | Yes | Data type (see supported types below) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `required` | boolean | false | Whether field must be present |
| `allowed_values` | array | null | List of valid values |
| `validate_in_text` | boolean | false | Only extract if explicitly mentioned |

### Supported Data Types

| Type           | Description                        | Example Values                |
|----------------|------------------------------------|-------------------------------|
| `string`       | Text values                        | "Apple", "technology"         |
| `number`       | Floating point numbers             | 1500000000, 12.5              |
| `integer`      | Whole numbers                      | 2024, 100                     |
| `boolean`      | True/false values                  | true, false                   |
| `[string]` | List of strings | '["Apple", "Google"]'         |
| `[number]` | List of numbers | '[12.5, 42, 100]'             |
| `[integer]`| List of integers     | '[2024, 100, 7]'              |
| `[boolean]`| List of booleans     | '[true, false, true]'         |

**Note:** List datatypes must be surrounded by quotes in `.yaml` files. For example `"[string]"`, not `[string]`


### Variable Examples

```yaml
# Simple string field
- name: "company_name"
  description: "Name of the company"
  data_type: "string"
  required: true

# Number with validation
- name: "revenue"
  description: "Revenue in USD"
  data_type: "number"
  required: false
  validate_in_text: true

# String field with allowed values (essentially an enum)
- name: "sector"
  description: "Business sector"
  data_type: "string"
  allowed_values: ["technology", "finance", "healthcare"]
  required: false

# Boolean field
- name: "is_public"
  description: "Whether company is publicly traded"
  data_type: "boolean"
  required: false

# List of numbers with allowed values
- name: "quarterly_growth_rates"
  description: "Quarterly revenue growth rates in percent"
  data_type: "[number]"
  allowed_values: [0, 5, 10, 15, 20, 25, 30]
  required: false
```

### Validation Features

#### Text Validation
```yaml
- name: "commodity_type"
  description: "Type of commodity mentioned"
  data_type: "string"
  validate_in_text: true  # Only extract if explicitly mentioned in text
```

#### Allowed Values
```yaml
- name: "sentiment"
  description: "Overall sentiment"
  data_type: "string"
  allowed_values: ["positive", "negative", "neutral"]
```

## Schema Examples

### Financial Report Analysis
```yaml
schema_type: "nested"
container_name: "financial_metrics"
variables:
  - name: "metric_name"
    description: "Name of the financial metric"
    data_type: "string"
    required: true
  - name: "value"
    description: "Numeric value of the metric"
    data_type: "number"
    required: true
  - name: "currency"
    description: "Currency of the value"
    data_type: "string"
    allowed_values: ["USD", "EUR", "GBP"]
    required: false
  - name: "period"
    description: "Time period for the metric"
    data_type: "string"
    required: false
```

### Commodity Price Extraction
```yaml
variables:
  - name: "commodity_type"
    description: "Type of commodity mentioned"
    data_type: "string"
    allowed_values: ["oil", "gas", "gold", "silver", "copper"]
    validate_in_text: true
  - name: "price_value"
    description: "Price value mentioned"
    data_type: "number"
    required: false
  - name: "price_mention"
    description: "Whether a price is mentioned"
    data_type: "boolean"
    required: false
  - name: "forecast_period"
    description: "Time period for price forecast"
    data_type: "string"
    required: false
```

### Customer Feedback Analysis
```yaml
schema_type: "multiple"

sentiment:
  schema_type: "nested"
  container_name: "sentiments"
  variables:
    - name: "aspect"
      description: "Product/service aspect mentioned"
      data_type: "string"
      required: true
    - name: "sentiment"
      description: "Sentiment toward the aspect"
      data_type: "string"
      allowed_values: ["positive", "negative", "neutral"]
      required: true
    - name: "intensity"
      description: "Intensity of the sentiment"
      data_type: "string"
      allowed_values: ["low", "medium", "high"]
      required: false

suggestions:
  schema_type: "nested"
  container_name: "suggestions"
  variables:
    - name: "suggestion"
      description: "Improvement suggestion"
      data_type: "string"
      required: true
    - name: "category"
      description: "Category of suggestion"
      data_type: "string"
      allowed_values: ["feature", "bug", "ui", "performance"]
      required: false
```

---

For more help, see the main README.md or open an issue on GitHub. 