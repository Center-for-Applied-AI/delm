# ExtractionVariable

Defines a single field to extract from text.

## Constructor

```python
from delm import ExtractionVariable

variable = ExtractionVariable(
    name="price",
    description="The price mentioned in the text",
    data_type="number",
    required=False,
    allowed_values=None,
    validate_in_text=False
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Field name (JSON key) |
| `description` | `str` | **required** | Field description (used in prompt) |
| `data_type` | `str` | **required** | Data type (see below) |
| `required` | `bool` | `False` | If `True`, extraction fails if field is missing |
| `allowed_values` | `List[str] \| None` | `None` | Restrict to specific values |
| `validate_in_text` | `bool` | `False` | If `True`, extracted value must appear in text |

## Data Types

### Scalars

| Type | Description | Example |
|------|-------------|---------|
| `"string"` | Text value | `"Apple Inc."` |
| `"number"` | Float | `123.45` |
| `"integer"` | Whole number | `42` |
| `"boolean"` | True/False | `true` |
| `"date"` | Date string | `"2024-01-15"` |

### Lists

Prefix with brackets: `"[string]"`, `"[number]"`, etc.

```python
ExtractionVariable("tags", "Product tags", "[string]")
# Output: {"tags": ["electronics", "gadgets", "home"]}
```

## Examples

### Basic Variable

```python
ExtractionVariable(
    name="company",
    description="The company name mentioned in the text",
    data_type="string"
)
```

### Required Variable

```python
ExtractionVariable(
    name="transaction_id",
    description="Unique transaction identifier",
    data_type="string",
    required=True  # Extraction fails if missing
)
```

### Restricted Values

```python
ExtractionVariable(
    name="commodity",
    description="Type of commodity mentioned",
    data_type="string",
    allowed_values=["oil", "gas", "gold", "silver"]
)
```

### Text Validation

```python
ExtractionVariable(
    name="ceo_name",
    description="CEO name",
    data_type="string",
    validate_in_text=True  # Value must appear exactly in text
)
```

### List Variable

```python
ExtractionVariable(
    name="prices",
    description="All prices mentioned",
    data_type="[number]"
)
# Output: {"prices": [10.5, 20.0, 15.75]}
```

## Methods

### from_dict()

Create ExtractionVariable from dictionary.

```python
variable = ExtractionVariable.from_dict({
    "name": "price",
    "description": "Price value",
    "data_type": "number",
    "required": False
})
```

---

### to_dict()

Convert to dictionary.

```python
variable_dict = variable.to_dict()
# Returns: {"name": "price", "description": "...", ...}
```

---

### is_list()

Check if variable represents a list.

```python
variable.is_list()  # True if data_type is "[...]"
```

