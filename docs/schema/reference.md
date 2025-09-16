# Schema Reference

Schemas define the structured outputs that DELM extracts from your documents. This reference walks through each schema type, configuration option, and validation rule so you can design prompts that return clean, predictable data.

## Schema Types

DELM ships with a progressive schema system. Start with simple key/value pairs and grow to nested or multi-schema extractions as your use case evolves.

### Simple Schema (Level 1)

Ideal for flat, per-chunk fields.

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

Output example:

```json
{
  "company_names": ["Apple", "Microsoft"],
  "revenue_numbers": [1500000000, 2000000000],
  "forecast_year": 2024
}
```

### Nested Schema (Level 2)

Group related fields into structured objects.

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
    validate_in_text: true
  - name: "products"
    description: "List of products offered by the company"
    data_type: "[string]"
    required: false
```

Output example:

```json
{
  "companies": [
    {
      "name": "Apple",
      "revenue": 1500000000,
      "sector": "technology",
      "growth_rate": 12.5,
      "products": ["iPhone", "MacBook", "iPad"]
    }
  ]
}
```

### Multiple Schemas (Level 3)

Extract several schemas in one request. Each entry under the root YAML object defines a sub-schema.

```yaml
schema_type: "multiple"

companies:
  schema_type: "nested"
  container_name: "companies"
  variables:
    - { name: "name", data_type: "string", required: true }
    - { name: "revenue", data_type: "number" }

products:
  schema_type: "nested"
  container_name: "products"
  variables:
    - { name: "name", data_type: "string", required: true }
    - { name: "price", data_type: "number" }
    - { name: "category", data_type: "string", allowed_values: ["software", "hardware", "service", "consulting"] }

market_trends:
  schema_type: "nested"
  container_name: "trends"
  variables:
    - { name: "trend_name", data_type: "string", required: true }
    - { name: "impact", data_type: "string", allowed_values: ["positive", "negative", "neutral"] }
```

Output example:

```json
{
  "companies": [ { "name": "Apple", "revenue": 1500000000 } ],
  "products": [ { "name": "iPhone 15", "price": 999, "category": "hardware" } ],
  "trends": [ { "trend_name": "AI adoption acceleration", "impact": "positive" } ]
}
```

## Variable Configuration

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Key used in the output JSON |
| `description` | string | Natural-language instructions passed to the LLM |
| `data_type` | string | Expected value type (see below) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `required` | boolean | `false` | Drop the response when no value is produced |
| `allowed_values` | list | `null` | Restrict outputs to predefined options |
| `validate_in_text` | boolean | `false` | Require the value to appear verbatim in the source text |

### Supported Data Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Free-form text | `"Apple"` |
| `number` | Floating-point numbers | `12.5` |
| `integer` | Whole numbers | `2024` |
| `boolean` | `true` / `false` | `false` |
| `date` | ISO-like date strings | `"2025-09-15"` |
| `[string]` | List of strings | `["oil", "gas"]` |
| `[number]` | List of numbers | `[0.12, 0.34]` |
| `[integer]` | List of integers | `[1, 2, 3]` |
| `[boolean]` | List of booleans | `[true, false]` |

> Wrap list types in quotes inside YAML files (e.g. `"[string]"`).

## Prompt Customization

DELM renders prompts using two strings from your pipeline configuration:

- `schema.system_prompt` becomes the system-role message.
- `schema.prompt_template` is formatted for each chunk with `{variables}`, `{text}`, and optional `{context}` placeholders.

For multi-schema configurations, prompts for each child schema are combined into a single message so that the model receives all required instructions at once.

## Validation Semantics

- Required fields without valid outputs cause the item (or entire response for simple schemas) to be discarded.
- Null-like strings in string fields (`"none"`, `"unknown"`, etc.) are filtered unless included in `allowed_values`.
- `validate_in_text: true` keeps only strings that appear verbatim in the source text (case-insensitive).
- For nested schemas, `container_name` defaults to `"instances"` when omitted.
- In multiple schemas, child outputs are unwrapped so top-level keys match the child schema names.

## Extended Examples

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
  - name: "period"
    description: "Time period for the metric"
    data_type: "string"
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
  - name: "price_mention"
    description: "Whether a price is mentioned"
    data_type: "boolean"
  - name: "forecast_period"
    description: "Time period for price forecast"
    data_type: "string"
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
```
