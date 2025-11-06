# Flat API Design: Simplicity First

## Philosophy

**Keep it flat unless there's a real reason for nesting.**

Most parameters are unique enough to be top-level:
- ✅ `temperature` - unique
- ✅ `provider` - unique  
- ✅ `model` - unique
- ✅ `batch_size` - unique
- ✅ `max_workers` - unique

Only use objects when:
1. The configuration has sub-parameters (e.g., splitting strategies)
2. It represents a complex entity (e.g., schema with multiple variables)

---

## Proposed Flat API

### Level 1: Simplest Possible (YAML)

```yaml
# delm_config.yaml

# LLM Settings (flat!)
provider: openai
model: gpt-4o-mini
temperature: 0.0
batch_size: 10
max_workers: 4
max_budget: 5.0

# Data Settings (flat where possible)
target_column: text
splitting: paragraph  # Simple string for common cases
scoring_keywords:     # Flat list
  - price
  - revenue
  - forecast

# Schema (complex, so it stays structured)
schema:
  type: simple
  variables:
    - name: price
      data_type: number
    - name: company
      data_type: string
```

```python
from delm import DELM

# One line!
delm = DELM.from_config("delm_config.yaml")
result = delm.extract(df)
```

---

### Level 2: Python with Flat Parameters

```python
from delm import DELM, Schema

# Most things are flat parameters!
delm = DELM(
    # LLM settings (flat)
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=10,
    max_budget=5.0,
    
    # Data settings (flat)
    target_column="text",
    splitting="paragraph",  # String for simple cases
    scoring_keywords=["price", "revenue"],
    
    # Only schema is structured (because it's complex)
    schema=Schema.simple(
        price="number",
        company="string",
        category=("string", ["electronics", "clothing", "food"])  # with allowed values
    )
)

result = delm.extract(df)
```

**Full type hints and autocomplete on all parameters!**

---

### Level 3: Advanced (When You Need Custom Strategies)

```python
from delm import DELM, Schema, FixedWindowSplit, KeywordScorer

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    
    # Custom splitting strategy (has sub-params, so use object)
    splitting=FixedWindowSplit(window=5, stride=2),
    
    # Custom scoring (has sub-params, so use object)
    scoring=KeywordScorer(keywords=["price"], threshold=0.8),
    
    schema=Schema.nested(
        container_name="products",
        variables={
            "name": "string",
            "price": "number"
        }
    )
)
```

---

## Smart String Shortcuts

For common strategies, use simple strings that map to objects:

```python
# Simple string shortcuts
splitting="paragraph"      # → ParagraphSplit()
splitting="sentence"       # → RegexSplit(pattern=r"(?<=[.!?])\s+")
splitting="fixed-window"   # → FixedWindowSplit() with defaults
splitting=None             # → No splitting

# Or full object when you need custom params
splitting=FixedWindowSplit(window=10, stride=5)

# Same for scoring
scoring="keywords:price,revenue,forecast"  # → KeywordScorer(keywords=[...])
scoring=None  # → No scoring

# Or full object
scoring=KeywordScorer(keywords=["price"], fuzzy=True, threshold=0.8)
```

---

## Complete Flat API Design

### DELM Constructor Signature

```python
from typing import Union, List, Optional, Literal
from pathlib import Path

class DELM:
    def __init__(
        self,
        # ============================================================
        # LLM SETTINGS (flat - all unique parameter names)
        # ============================================================
        provider: Literal["openai", "anthropic", "google", "groq", "together", "fireworks"],
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_workers: int = 1,
        max_budget: Optional[float] = None,
        dotenv_path: Union[str, Path] = ".env",
        
        # ============================================================
        # DATA PREPROCESSING (flat where possible)
        # ============================================================
        target_column: str = "delm_raw_data",
        
        # Simple strings for common cases, objects for advanced
        splitting: Union[
            Literal["paragraph", "sentence", "fixed-window"],  # String shortcuts
            SplitStrategy,  # Or full object
            None
        ] = None,
        
        # Simple for common case (keyword list), object for advanced
        scoring: Union[
            List[str],  # Shortcut: just list keywords
            ScoringStrategy,  # Or full object
            None
        ] = None,
        
        score_filter: Optional[str] = None,  # e.g., "score > 0.5"
        
        # ============================================================
        # SCHEMA (stays structured - too complex to flatten)
        # ============================================================
        schema: Union[Schema, str, Path],  # Schema object or path to YAML
        
        # ============================================================
        # EXPERIMENT (flat - optional)
        # ============================================================
        experiment: Optional[str] = None,  # Just the name
        experiment_dir: Path = Path("./experiments"),
        overwrite: bool = False,
        
        # ============================================================
        # CACHE (flat)
        # ============================================================
        cache: bool = True,
        cache_path: Path = Path(".delm_cache"),
    ):
        """
        Initialize DELM for data extraction.
        
        Args:
            provider: LLM provider ("openai", "anthropic", etc.)
            model: Model name (e.g., "gpt-4o-mini")
            temperature: Sampling temperature (0.0-2.0)
            batch_size: Chunks to process per batch
            max_workers: Number of concurrent workers
            max_budget: Optional budget limit in dollars
            
            target_column: Column containing text to process
            
            splitting: Text splitting strategy:
                - "paragraph": Split by paragraphs
                - "sentence": Split by sentences  
                - "fixed-window": Fixed-size windows
                - FixedWindowSplit(window=5, stride=2): Custom
                - None: No splitting
            
            scoring: Relevance scoring:
                - ["price", "revenue"]: Keyword list (simple)
                - KeywordScorer(...): Custom scorer object
                - None: No scoring
            
            schema: Extraction schema (Schema object or YAML path)
            
            experiment: Optional experiment name for tracking
            
        Example:
            >>> delm = DELM(
            ...     provider="openai",
            ...     model="gpt-4o-mini",
            ...     schema=Schema.simple(price="number", company="string")
            ... )
        """
```

---

## Schema API - Also Simplified!

```python
class Schema:
    """Simplified schema creation."""
    
    @staticmethod
    def simple(**variables) -> SimpleSchema:
        """
        Create a simple (key-value) schema.
        
        Args:
            **variables: variable_name=data_type pairs
            
        Examples:
            >>> Schema.simple(
            ...     price="number",
            ...     company="string",
            ...     active="boolean"
            ... )
            
            >>> # With allowed values (use tuple)
            >>> Schema.simple(
            ...     category=("string", ["electronics", "clothing", "food"]),
            ...     price="number"
            ... )
            
            >>> # Mark as required (prefix with !)
            >>> Schema.simple(
            ...     company="!string",  # Required
            ...     price="number"       # Optional
            ... )
        """
    
    @staticmethod
    def nested(container_name: str, **variables) -> NestedSchema:
        """
        Create a nested (list) schema.
        
        Args:
            container_name: Name for the list container
            **variables: variable_name=data_type pairs
            
        Example:
            >>> Schema.nested(
            ...     "products",
            ...     name="!string",  # Required
            ...     price="number",
            ...     quantity="integer"
            ... )
        """
    
    @staticmethod
    def from_yaml(path: Union[str, Path]) -> Union[SimpleSchema, NestedSchema]:
        """Load schema from YAML file."""
    
    @staticmethod
    def from_dict(data: dict) -> Union[SimpleSchema, NestedSchema]:
        """Create schema from dictionary."""
```

---

## Complete Examples

### Example 1: Absolute Simplest

```python
from delm import DELM, Schema

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=Schema.simple(price="number", company="string")
)

result = delm.extract(df)
```

**Lines of code: 6**  
**Type hints: ✅ Full**  
**Autocomplete: ✅ Everything**

---

### Example 2: With Keywords

```python
from delm import DELM, Schema

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    splitting="paragraph",
    scoring=["price", "revenue", "forecast"],  # Just a list!
    schema=Schema.simple(price="number", company="string")
)

result = delm.extract(df)
```

---

### Example 3: Nested Schema with Allowed Values

```python
from delm import DELM, Schema

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=Schema.nested(
        "commodities",
        commodity_type=("!string", ["oil", "gas", "copper", "gold"]),  # Required + allowed values
        price_value="number",
        price_unit="string"
    )
)
```

---

### Example 4: Advanced - Custom Strategies

```python
from delm import DELM, Schema, FixedWindowSplit, FuzzyScorer

delm = DELM(
    provider="anthropic",
    model="claude-3-sonnet",
    temperature=0.1,
    max_budget=10.0,
    
    # Custom splitting with parameters
    splitting=FixedWindowSplit(window=5, stride=2),
    
    # Custom scoring with fuzzy matching
    scoring=FuzzyScorer(
        keywords=["price", "forecast"],
        threshold=0.8,
        fuzzy=True
    ),
    
    schema=Schema.nested(
        "products",
        name="!string",
        price="number"
    ),
    
    experiment="my_extraction_v1"
)
```

---

### Example 5: From YAML (Still Supported)

```yaml
# config.yaml
provider: openai
model: gpt-4o-mini
temperature: 0.0

splitting: paragraph
scoring:
  keywords:
    - price
    - revenue

schema:
  type: simple
  variables:
    price: number
    company: string
```

```python
delm = DELM.from_config("config.yaml")
```

---

## Type Hints in Action

### IDE Experience: Parameter Hints

**User types:**
```python
delm = DELM(
    provider=█
```

**IDE shows:**
```
provider: Literal["openai", "anthropic", "google", "groq", "together", "fireworks"]
    LLM provider

Autocomplete:
  • openai
  • anthropic
  • google
  • groq
  • together
  • fireworks
```

---

**User types:**
```python
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    splitting=█
```

**IDE shows:**
```
splitting: Union[Literal["paragraph", "sentence", "fixed-window"], SplitStrategy, None]
    Text splitting strategy
    
Options:
  • "paragraph" - Split by paragraphs
  • "sentence" - Split by sentences
  • "fixed-window" - Fixed-size windows
  • FixedWindowSplit(...) - Custom configuration
  • None - No splitting
```

---

**User types:**
```python
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    scoring=█
```

**IDE shows:**
```
scoring: Union[List[str], ScoringStrategy, None]
    Relevance scoring
    
Options:
  • ["keyword1", "keyword2"] - Simple keyword list
  • KeywordScorer(...) - Custom keyword scorer
  • FuzzyScorer(...) - Fuzzy keyword matcher
  • None - No scoring
```

---

## Handling the Strategy Problem

You're right that strategies are the tricky part. Here's the elegant solution:

### Option 1: String Shortcuts (Simple Cases)

```python
# 90% of users just want simple splitting
splitting="paragraph"  # → ParagraphSplit()
splitting="sentence"   # → RegexSplit(pattern=r"(?<=[.!?])\s+")

# 90% of users just want keyword scoring
scoring=["price", "revenue"]  # → KeywordScorer(keywords=[...])
```

### Option 2: Objects (Advanced Cases)

```python
# 10% of users need custom parameters
splitting=FixedWindowSplit(window=10, stride=5)
splitting=RegexSplit(pattern=r"\n\n")

scoring=KeywordScorer(keywords=["price"], case_sensitive=False)
scoring=FuzzyScorer(keywords=["price"], threshold=0.8)
```

### Implementation: Union Types

```python
SplittingOption = Union[
    Literal["paragraph", "sentence", "fixed-window"],  # String shortcuts
    SplitStrategy,  # Custom strategy objects
    None  # No splitting
]

ScoringOption = Union[
    List[str],  # Simple: just keywords
    ScoringStrategy,  # Custom: full scorer object
    None  # No scoring
]

def __init__(
    self,
    ...,
    splitting: SplittingOption = None,
    scoring: ScoringOption = None,
    ...
):
    # Normalize to objects internally
    if isinstance(splitting, str):
        splitting = self._parse_splitting_string(splitting)
    
    if isinstance(scoring, list):
        scoring = KeywordScorer(keywords=scoring)
```

---

## Variable Declaration Shortcuts

### For Schema Variables

Instead of verbose:
```python
Variable(
    name="price",
    description="Price value",
    data_type="number",
    required=False
)
```

Use compact syntax:
```python
# Format: name=data_type
price="number"

# Required (prefix with !)
price="!number"

# With allowed values (use tuple)
category=("string", ["electronics", "clothing"])

# Required + allowed values
category=("!string", ["electronics", "clothing"])

# With description (use dict)
price={"type": "number", "description": "Price value in USD"}
```

### Schema Builder Implementation

```python
class Schema:
    @staticmethod
    def simple(**variables) -> SimpleSchema:
        parsed_vars = []
        for name, spec in variables.items():
            parsed_vars.append(_parse_variable(name, spec))
        return SimpleSchema(variables=parsed_vars)

def _parse_variable(name: str, spec: Union[str, tuple, dict]) -> Variable:
    """
    Parse compact variable specification.
    
    Examples:
        "number" → Variable(name=name, data_type="number")
        "!string" → Variable(name=name, data_type="string", required=True)
        ("string", ["a", "b"]) → Variable(name=name, data_type="string", allowed_values=["a", "b"])
    """
    if isinstance(spec, str):
        required = spec.startswith("!")
        data_type = spec.lstrip("!")
        return Variable(
            name=name,
            description=name.replace("_", " ").title(),  # Auto-generate description
            data_type=data_type,
            required=required
        )
    
    elif isinstance(spec, tuple):
        data_type_spec, allowed_values = spec
        required = data_type_spec.startswith("!")
        data_type = data_type_spec.lstrip("!")
        return Variable(
            name=name,
            description=name.replace("_", " ").title(),
            data_type=data_type,
            required=required,
            allowed_values=allowed_values
        )
    
    elif isinstance(spec, dict):
        return Variable(
            name=name,
            description=spec.get("description", name.replace("_", " ").title()),
            data_type=spec.get("type", "string"),
            required=spec.get("required", False),
            allowed_values=spec.get("allowed_values"),
            validate_in_text=spec.get("validate_in_text", False)
        )
```

---

## Comparison: Nested vs Flat

### Nested (Original Pydantic Proposal)

```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=0.0,
        batch_size=10,
        max_workers=4
    ),
    data_preprocessing=DataPreprocessing(
        target_column="text",
        splitting=ParagraphSplit(),
        scoring=KeywordScorer(keywords=["price", "revenue"])
    ),
    schema=SchemaConfig(
        schema=SimpleSchema(
            variables=[
                Variable(
                    name="price",
                    description="Price value",
                    data_type="number"
                ),
                Variable(
                    name="company",
                    description="Company name",
                    data_type="string"
                )
            ]
        )
    )
)

delm = DELM(config)
```

**Lines: 31**  
**Nesting levels: 4**  
**Cognitive load: High**

---

### Flat (New Proposal)

```python
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=10,
    max_workers=4,
    target_column="text",
    splitting="paragraph",
    scoring=["price", "revenue"],
    schema=Schema.simple(
        price="number",
        company="string"
    )
)
```

**Lines: 12**  
**Nesting levels: 1**  
**Cognitive load: Low**

**Improvement: 61% fewer lines, much flatter!**

---

## Migration Path

### Old API (Still Works)
```python
config = DELMConfig.from_yaml("config.yaml")
delm = DELM(config=config, experiment_name="test", ...)
```

### New Flat API
```python
delm = DELM.from_config("config.yaml")  # or
delm = DELM(provider="openai", model="gpt-4o-mini", ...)
```

Both work, with deprecation warnings guiding users to new API.

---

## Best of Both Worlds

### For Beginners: YAML is Simplest
```yaml
provider: openai
model: gpt-4o-mini
scoring: [price, revenue]
schema:
  type: simple
  variables:
    price: number
    company: string
```

### For Python Users: Flat is Natural
```python
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    scoring=["price", "revenue"],
    schema=Schema.simple(price="number", company="string")
)
```

### For Advanced: Objects When Needed
```python
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    splitting=FixedWindowSplit(window=5, stride=2),
    scoring=FuzzyScorer(keywords=["price"], threshold=0.8),
    schema=Schema.from_yaml("complex_schema.yaml")
)
```

---

## Summary: Why Flat is Better

### Pros of Flat API
1. ✅ **Less typing** - No nested objects for simple params
2. ✅ **More Pythonic** - Feels like native Python functions
3. ✅ **Better autocomplete** - All params at top level
4. ✅ **Clearer** - No artificial grouping
5. ✅ **Progressive** - String shortcuts → objects when needed

### Where Nesting Still Makes Sense
1. **Schema** - Complex with multiple variables
2. **Custom strategies** - Have sub-parameters
3. **Config files** - Logical grouping in YAML

### The Rule
- **Flat by default** - Most parameters are unique
- **Nest only when**:
  - Multiple sub-parameters (e.g., `FixedWindowSplit(window=5, stride=2)`)
  - Complex entity (e.g., Schema with variables)
  - String shortcut won't work (e.g., custom regex)

---

## Final Recommendation

**Use the flat API with smart shortcuts:**

```python
# Simple case (90% of users)
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    scoring=["price", "revenue"],
    schema=Schema.simple(price="number", company="string")
)

# Advanced case (10% of users)
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    splitting=FixedWindowSplit(window=5, stride=2),
    scoring=FuzzyScorer(keywords=["price"], threshold=0.8),
    schema=Schema.from_yaml("schema.yaml")
)
```

This gives you:
- ✅ Simplicity for common cases
- ✅ Power for advanced cases
- ✅ Full type safety everywhere
- ✅ Great IDE support
- ✅ Natural Python feel

**This is the cleanest possible API!** 🎯


