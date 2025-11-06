# IDE Experience: Type Hints & Autocomplete Mockup

This document shows what the developer experience would look like with the proposed Python configuration API.

---

## VSCode / Pylance Experience

### Scenario 1: Creating a Basic Configuration

**User Types:**
```python
from delm import DELMConfig, LLMExtraction

config = DELMConfig(
    llm_extraction=LLM█
```

**IDE Shows Autocomplete:**
```
┌─────────────────────────────────────────┐
│ ▼ Suggestions                           │
├─────────────────────────────────────────┤
│ ★ LLMExtraction                         │
│   class LLMExtraction(BaseModel)        │
│   LLM API configuration                 │
│                                         │
│   LLMExtractionConfig (deprecated)     │
│   LLMProvider                           │
└─────────────────────────────────────────┘
```

**User Continues:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider=█
```

**IDE Shows Parameter Hint:**
```
provider: Literal["openai", "anthropic", "google", "groq", "together", "fireworks"]
    LLM provider
```

**IDE Shows Autocomplete:**
```
┌─────────────────────────────────────────┐
│ ▼ Valid values for provider             │
├─────────────────────────────────────────┤
│ ★ "openai"                              │
│   "anthropic"                           │
│   "google"                              │
│   "groq"                                │
│   "together"                            │
│   "fireworks"                           │
└─────────────────────────────────────────┘
```

**User Hovers Over `temperature`:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=█  # User hovers here
```

**IDE Shows Tooltip:**
```
┌──────────────────────────────────────────────────────┐
│ (parameter) temperature: float = 0.0                 │
│                                                      │
│ Sampling temperature                                 │
│                                                      │
│ 0.0 = deterministic (recommended for extraction)    │
│ 2.0 = very random (creative generation)             │
│                                                      │
│ Default: 0.0                                         │
│ Range: 0.0 to 2.0                                    │
└──────────────────────────────────────────────────────┘
```

---

### Scenario 2: Schema Definition

**User Types:**
```python
from delm import Variable

var = Variable(
    name="price",
    description="Price mentioned in text",
    data_type=█
```

**IDE Shows Autocomplete with Type Info:**
```
┌─────────────────────────────────────────────────┐
│ ▼ Valid values for data_type                    │
├─────────────────────────────────────────────────┤
│ ★ "string"        Text values                   │
│   "number"        Floating-point numbers        │
│   "integer"       Whole numbers                 │
│   "boolean"       True/False values             │
│   "date"          Date strings (YYYY-MM-DD)     │
│   "[string]"      List of text values           │
│   "[number]"      List of numbers               │
│   "[integer]"     List of integers              │
└─────────────────────────────────────────────────┘
```

**User Adds More Parameters:**
```python
var = Variable(
    name="price",
    description="Price mentioned in text",
    data_type="number",
    █  # Cursor here, user presses Ctrl+Space
```

**IDE Shows All Available Parameters:**
```
┌─────────────────────────────────────────────────────────┐
│ ▼ Available parameters                                  │
├─────────────────────────────────────────────────────────┤
│   required: bool = False                                │
│       Whether this field must be present                │
│                                                         │
│   allowed_values: Optional[List[Union[str, int, ...]]] │
│       List of valid values for this field               │
│                                                         │
│   validate_in_text: bool = False                        │
│       Whether to validate extracted value appears in... │
└─────────────────────────────────────────────────────────┘
```

---

### Scenario 3: Type Error Detection

**User Writes Invalid Code:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=3.0,  # ❌ Squiggly underline
        max_budget=-5.0   # ❌ Squiggly underline
    )
)
```

**IDE Shows Error on Hover:**
```
┌──────────────────────────────────────────────────────┐
│ ❌ Argument of type "Literal[3.0]" cannot be        │
│    assigned to parameter "temperature" of type      │
│    "float"                                          │
│                                                     │
│ Expected: 0.0 <= value <= 2.0                      │
│ Got: 3.0                                           │
└──────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────┐
│ ❌ Argument of type "Literal[-5.0]" cannot be       │
│    assigned to parameter "max_budget" of type       │
│    "Optional[float]"                                │
│                                                     │
│ Expected: value >= 0.0                             │
│ Got: -5.0                                          │
└──────────────────────────────────────────────────────┘
```

**Problems Panel Shows:**
```
Problems (2)
┌───────────────────────────────────────────────────────────────┐
│ ⚠ temperature must be <= 2.0            config.py:5  [Pylance]│
│ ⚠ max_budget must be >= 0.0            config.py:6  [Pylance]│
└───────────────────────────────────────────────────────────────┘
```

---

### Scenario 4: Runtime Validation Errors

**User Runs Code:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        max_budget=10.0,
        track_cost=False  # ❌ Invalid: budget requires cost tracking
    )
)
```

**Runtime Error Message:**
```
ValidationError: 1 validation error for DELMConfig

llm_extraction.max_budget
  max_budget requires track_cost=True
  
  Current values:
    max_budget: 10.0
    track_cost: False
  
  To fix, either:
    1. Set track_cost=True, or
    2. Remove max_budget
  
  (type=value_error)
```

---

### Scenario 5: Builder Pattern Experience

**User Types:**
```python
from delm import SchemaBuilder

schema = SchemaBuilder()█
```

**IDE Shows Available Methods:**
```
┌─────────────────────────────────────────────────────────┐
│ ▼ Methods of SchemaBuilder                              │
├─────────────────────────────────────────────────────────┤
│ ⚙ add_string(name, description, ...) → SchemaBuilder   │
│       Add a string variable                             │
│                                                         │
│ ⚙ add_number(name, description, ...) → SchemaBuilder   │
│       Add a number variable                             │
│                                                         │
│ ⚙ add_integer(name, description, ...) → SchemaBuilder  │
│       Add an integer variable                           │
│                                                         │
│ ⚙ add_boolean(name, description, ...) → SchemaBuilder  │
│       Add a boolean variable                            │
│                                                         │
│ ⚙ add_date(name, description, ...) → SchemaBuilder     │
│       Add a date variable                               │
│                                                         │
│ ⚙ build_simple() → SimpleSchema                        │
│       Build a simple (key-value) schema                 │
│                                                         │
│ ⚙ build_nested(container_name) → NestedSchema          │
│       Build a nested (list) schema                      │
└─────────────────────────────────────────────────────────┘
```

**User Chains Methods:**
```python
schema = (
    SchemaBuilder()
    .add_string("commodity", "Type of commodity", required=True)█
```

**After `.`, IDE Shows Same Methods (Chainable):**
```
┌─────────────────────────────────────────────────────────┐
│ ▼ Methods of SchemaBuilder                              │
├─────────────────────────────────────────────────────────┤
│ ⚙ add_string(...)                                       │
│ ⚙ add_number(...)                                       │
│ ⚙ add_integer(...)                                      │
│ ⚙ build_simple()                                        │
│ ⚙ build_nested(container_name)                         │
└─────────────────────────────────────────────────────────┘
```

**User Completes:**
```python
schema = (
    SchemaBuilder()
    .add_string("commodity", "Type of commodity", required=True)
    .add_number("price", "Price value")
    .add_string("unit", "Unit of measurement")
    .build_nested(█  # IDE shows parameter hint
```

**IDE Shows:**
```
container_name: str
    The key name for the list of extracted items
    
    Example: "products", "commodities", "transactions"
```

---

### Scenario 6: Documentation on Hover

**User Hovers Over Class:**
```python
from delm import SimpleSchema

schema = SimpleSchema(█)
```

**IDE Shows Full Documentation:**
```
┌────────────────────────────────────────────────────────────┐
│ class SimpleSchema(BaseModel)                              │
│                                                            │
│ Simple key-value extraction schema.                        │
│                                                            │
│ Use when extracting a single set of properties per chunk. │
│                                                            │
│ Example:                                                   │
│     >>> SimpleSchema(                                      │
│     ...     variables=[                                    │
│     ...         Variable(name="company", data_type="str"), │
│     ...         Variable(name="price", data_type="number") │
│     ...     ]                                              │
│     ... )                                                  │
│                                                            │
│ Expected Output:                                           │
│     {"company": "Apple", "price": 150.5}                   │
│                                                            │
│ Parameters:                                                │
│     type: Literal["simple"] = "simple"                     │
│         Schema type                                        │
│                                                            │
│     variables: List[Variable]                              │
│         List of fields to extract (min length: 1)          │
└────────────────────────────────────────────────────────────┘
```

---

### Scenario 7: Quick Fix Suggestions

**Code with Error:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini"
    )
    # ❌ Missing required parameter: schema
)
```

**IDE Shows:**
```
┌────────────────────────────────────────────────────────────┐
│ ❌ Missing required argument "schema" for "DELMConfig"    │
│                                                            │
│ Quick Fixes:                                               │
│   💡 Add missing parameter "schema"                        │
│   💡 Generate example schema                               │
│   💡 Load schema from file                                 │
└────────────────────────────────────────────────────────────┘
```

**User Selects "Generate example schema":**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini"
    ),
    schema=SchemaConfig(  # ✨ Auto-generated
        schema=SimpleSchema(
            variables=[
                Variable(
                    name="field_name",
                    description="Field description",
                    data_type="string"
                )
            ]
        )
    )
)
```

---

## PyCharm Experience

### Scenario 1: Parameter Info Popup

**User Types:**
```python
from delm import Variable

Variable(█)  # User presses Ctrl+P (Parameter Info)
```

**PyCharm Shows:**
```
┌────────────────────────────────────────────────────────────┐
│ Parameter Info: Variable.__init__                          │
├────────────────────────────────────────────────────────────┤
│ Required:                                                  │
│   name: str                                                │
│   description: str                                         │
│   data_type: Literal["string", "number", ...]              │
│                                                            │
│ Optional:                                                  │
│   required: bool = False                                   │
│   allowed_values: Optional[List[...]] = None               │
│   validate_in_text: bool = False                           │
└────────────────────────────────────────────────────────────┘
```

---

### Scenario 2: Structure View

**User Opens Structure View (Alt+7):**
```
Structure: config.py
┌────────────────────────────────────────┐
│ 📄 config.py                           │
├────────────────────────────────────────┤
│ ├─ 📦 Imports                          │
│ │  ├─ DELMConfig                       │
│ │  ├─ LLMExtraction                    │
│ │  └─ SchemaConfig                     │
│ ├─ 🔧 config: DELMConfig               │
│ │  ├─ llm_extraction: LLMExtraction    │
│ │  │  ├─ provider: "openai"            │
│ │  │  ├─ name: "gpt-4o-mini"           │
│ │  │  └─ temperature: 0.0              │
│ │  └─ schema: SchemaConfig             │
│ │     └─ schema: SimpleSchema          │
│ │        └─ variables: List[Variable]  │
│ │           ├─ [0]: Variable           │
│ │           │  ├─ name: "price"        │
│ │           │  └─ data_type: "number"  │
│ │           └─ [1]: Variable           │
│ │              ├─ name: "company"      │
│ │              └─ data_type: "string"  │
└────────────────────────────────────────┘
```

---

### Scenario 3: Smart Refactoring

**User Selects `provider="openai"` and Presses Ctrl+Alt+V (Extract Variable):**

**Before:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini"
    )
)
```

**After Refactoring:**
```python
llm_provider: Provider = "openai"  # ✨ Correct type inferred

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider=llm_provider,
        name="gpt-4o-mini"
    )
)
```

---

### Scenario 4: Type Hierarchy

**User Presses Ctrl+H on `SchemaConfig`:**
```
Type Hierarchy: SchemaConfig
┌────────────────────────────────────────┐
│ 📦 SchemaConfig                        │
├────────────────────────────────────────┤
│ ⬆️ Superclasses                        │
│   └─ pydantic.BaseModel                │
│      └─ pydantic.main.ModelMetaclass   │
│                                        │
│ 👥 Usages (5 locations)                │
│   ├─ config.py:15                      │
│   ├─ config.py:42                      │
│   ├─ delm/core.py:88                   │
│   └─ ...                               │
└────────────────────────────────────────┘
```

---

## Jupyter Notebook Experience

### Scenario 1: Interactive Type Info

**In Notebook Cell:**
```python
from delm import Variable

Variable?  # User runs this
```

**Output:**
```
Type:        type
String form: <class 'delm.config.Variable'>
File:        /path/to/delm/config.py
Docstring:
Schema variable definition.

Defines a field to extract from text.

Parameters
----------
name : str
    Field name for extracted data
description : str
    Description for LLM to understand what to extract
data_type : Literal["string", "number", "integer", "boolean", "date", ...]
    Data type of the field. Default: "string"
required : bool
    Whether this field must be present. Default: False
allowed_values : Optional[List[Union[str, int, float]]]
    List of valid values for this field. Default: None
validate_in_text : bool
    Whether to validate extracted value appears in source text. Default: False

Examples
--------
>>> Variable(
...     name="price",
...     description="Price mentioned in text",
...     data_type="number"
... )

>>> Variable(
...     name="category",
...     description="Product category",
...     data_type="string",
...     allowed_values=["electronics", "clothing", "food"]
... )

See Also
--------
SimpleSchema : For simple key-value extraction
NestedSchema : For extracting lists of items
```

---

### Scenario 2: Tab Completion

**In Notebook Cell:**
```python
from delm import DELMConfig, LLMExtraction, SchemaConfig

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini"
    ),
    schema=SchemaConfig█  # User presses Tab
```

**Notebook Shows:**
```
SchemaConfig(
  schema=...,
  prompt_template=None,
  system_prompt=None
)
```

**After User Presses Tab Again (Signature Help):**
```
Signature: SchemaConfig(
    schema: Union[SimpleSchema, NestedSchema, MultipleSchema],
    prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **data
)

Docstring:
Schema configuration for extraction.

Can be defined inline or loaded from file.
```

---

## Command Line Tool Experience

### Scenario 1: Config Validation

**User Runs:**
```bash
$ delm validate-config config.yaml
```

**Output:**
```
✓ Configuration is valid!

Summary:
  Provider: openai
  Model: gpt-4o-mini
  Schema Type: nested
  Container: commodities
  Variables: 3
    - commodity_type (string, required)
    - price_value (number, optional)
    - unit (string, optional)

Estimated Cost (1000 records):
  Input tokens: ~150,000
  Output tokens: ~30,000
  Total cost: ~$0.75
```

---

### Scenario 2: Config Template Generation

**User Runs:**
```bash
$ delm init-config --schema-type nested --provider openai
```

**Output:**
```
✨ Created config.yaml

Next steps:
  1. Edit config.yaml to add your schema variables
  2. Add your OpenAI API key to .env
  3. Run: delm extract --config config.yaml --input data.csv

Example variable to add:
  - name: your_field_name
    description: Description for the LLM
    data_type: string
    required: false
```

---

### Scenario 3: Python Code Generation

**User Runs:**
```bash
$ delm yaml-to-python config.yaml
```

**Output:**
```python
# Generated from config.yaml
from delm import DELMConfig, LLMExtraction, SchemaConfig, NestedSchema, Variable

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=0.0
    ),
    schema=SchemaConfig(
        schema=NestedSchema(
            container_name="commodities",
            variables=[
                Variable(
                    name="commodity_type",
                    description="Type of commodity mentioned",
                    data_type="string",
                    required=True,
                    allowed_values=["oil", "gas", "copper", "gold"]
                ),
                Variable(
                    name="price_value",
                    description="Numeric price value",
                    data_type="number"
                )
            ]
        )
    )
)

# Save this to a .py file and use it:
# from delm import DELM
# delm = DELM(config)
# result = delm.extract(your_data)
```

---

## Comparison: Current vs Proposed

### Current (Dict-based, No Type Hints)

**Code:**
```python
config_dict = {
    "llm_extraction": {
        "provider": "openai",  # ❌ No autocomplete
        "name": "gpt-4o-mini",
        "temperature": 3.0,  # ❌ No validation until runtime
    },
    "schema": {
        "spec_path": "schema.yaml"  # ❌ No inline editing
    }
}
config = DELMConfig.from_dict(config_dict)
```

**IDE Shows:**
```
config_dict: dict[str, Any]  # ❌ No specific type info
```

**Error at Runtime:**
```python
delm = DELM(config)
# ❌ RuntimeError: temperature must be <= 2.0
```

---

### Proposed (Pydantic-based, Full Type Hints)

**Code:**
```python
from delm import DELMConfig, LLMExtraction, SchemaConfig, SimpleSchema, Variable

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",  # ✅ Autocomplete with valid options
        name="gpt-4o-mini",
        temperature=3.0,  # ✅ IDE shows error immediately
    ),
    schema=SchemaConfig(  # ✅ Define inline with autocomplete
        schema=SimpleSchema(
            variables=[
                Variable(name="price", data_type="number")
            ]
        )
    )
)
```

**IDE Shows:**
```
config: DELMConfig  # ✅ Full type information
  llm_extraction: LLMExtraction
    provider: Literal["openai", "anthropic", ...]
    name: str
    temperature: float (0.0-2.0)
```

**Error Before Running:**
```python
# ✅ IDE shows squiggly line and error immediately
temperature=3.0
          ~~~
Error: Value 3.0 exceeds maximum of 2.0
```

---

## Summary: Developer Experience Improvements

| Feature | Current | With Pydantic | Improvement |
|---------|---------|---------------|-------------|
| **Autocomplete** | ❌ None | ✅ Full | ⭐⭐⭐⭐⭐ |
| **Type Checking** | ❌ Runtime only | ✅ IDE + Runtime | ⭐⭐⭐⭐⭐ |
| **Documentation** | ❌ Separate docs | ✅ On hover | ⭐⭐⭐⭐⭐ |
| **Error Messages** | ❌ Generic | ✅ Specific | ⭐⭐⭐⭐⭐ |
| **Validation** | ❌ Runtime | ✅ IDE + Runtime | ⭐⭐⭐⭐⭐ |
| **Refactoring** | ❌ Limited | ✅ Full IDE support | ⭐⭐⭐⭐ |
| **Discovery** | ❌ Read docs | ✅ IDE suggests | ⭐⭐⭐⭐⭐ |
| **Confidence** | ⚠️ Hope it works | ✅ Know it works | ⭐⭐⭐⭐⭐ |

---

## User Testimonials (Projected)

### Before (Current API)
> "I spent 30 minutes debugging a typo in my config file. The error message just said 'KeyError: provider'. I had to read through all the docs to figure out I misspelled it as 'proivder'." 
> — Frustrated Data Scientist

> "I can never remember which parameters are valid for the LLM config. I have to keep the docs open in another tab."
> — Research Engineer

### After (With Type Hints)
> "The IDE just shows me all the valid options! I can hover over any parameter and see exactly what it does. This is amazing!"
> — Happy Data Scientist

> "I made a typo and the IDE caught it before I even ran the code. This saves so much time!"
> — Productive Research Engineer

---

## Next Steps

1. ✅ Review this mockup with team
2. ✅ Validate assumptions with users
3. ✅ Start implementation with Pydantic
4. ✅ Create migration guide
5. ✅ Update documentation
6. ✅ Add examples for all editors (VSCode, PyCharm, Jupyter)

The future of DELM configuration is type-safe and delightful! 🚀


