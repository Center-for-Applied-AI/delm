# Schema

Schema factory for defining extraction structures.

## Simple Schema

Extract flat key-value pairs.

```python
from delm import Schema, ExtractionVariable

schema = Schema.simple(
    ExtractionVariable("company", "Company name", "string"),
    ExtractionVariable("revenue", "Revenue amount", "number", required=True)
)

# Or with list:
variables = [...]
schema = Schema.simple(variables_list=variables)
```

## Nested Schema

Extract a list of objects with the same structure.

```python
schema = Schema.nested(
    "products",  # Container name
    ExtractionVariable("name", "Product name", "string", required=True),
    ExtractionVariable("price", "Product price", "number")
)
```

**JSON output:**

```json
{
  "products": [
    {"name": "Widget", "price": 10.0},
    {"name": "Gadget", "price": 20.0}
  ]
}
```

## Multiple Schema

Extract multiple independent structures from the same text.

```python
products_schema = Schema.nested("products", ...)
companies_schema = Schema.simple(...)

schema = Schema.multiple(
    products=products_schema,
    companies=companies_schema
)
```

**JSON output:**

```json
{
  "products": [...],
  "companies": {"name": "...", "industry": "..."}
}
```

## Methods

### Schema.simple()

```python
Schema.simple(
    *variables: ExtractionVariable,
    variables_list: List[ExtractionVariable] | None = None
) -> Schema
```

Create a simple (flat) schema.

---

### Schema.nested()

```python
Schema.nested(
    container_name: str,
    *variables: ExtractionVariable,
    variables_list: List[ExtractionVariable] | None = None
) -> Schema
```

Create a nested (list) schema.

---

### Schema.multiple()

```python
Schema.multiple(**schemas: Schema) -> Schema
```

Create a multiple schema from named sub-schemas.

---

### Schema.from_dict()

```python
Schema.from_dict(data: dict) -> Schema
```

Create schema from dictionary.

**Example:**

```python
schema = Schema.from_dict({
    "schema_type": "simple",
    "variables": [
        {"name": "price", "description": "Price", "data_type": "number"}
    ]
})
```

---

### Schema.from_yaml()

```python
Schema.from_yaml(path: str | Path) -> Schema
```

Load schema from YAML file.

---

### schema.to_dict()

```python
schema.to_dict() -> dict
```

Convert schema to dictionary representation.

