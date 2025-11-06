# Python Configuration API: Full Type Safety & IDE Support

## Vision

Enable users to define DELM configurations entirely in Python with:
- ✅ Full IDE autocomplete
- ✅ Type hints with hover documentation
- ✅ Compile-time error checking
- ✅ Programmatic composition
- ✅ Still support YAML for those who prefer it

---

## Design Principles

1. **Type Safety First**: Everything should be typed and validated
2. **Ergonomic**: Should feel natural and Pythonic
3. **Discoverable**: IDE should guide users through options
4. **Composable**: Easy to build configs programmatically
5. **Interoperable**: YAML ↔ Python ↔ dict conversions
6. **Progressive**: Simple things simple, complex things possible

---

## Option 1: Pydantic-Based Configuration (RECOMMENDED)

### Why Pydantic?
- ✅ Runtime validation with great error messages
- ✅ Automatic type coercion (e.g., str → Path)
- ✅ JSON schema generation (for docs)
- ✅ Excellent IDE support
- ✅ Industry standard (FastAPI, LangChain use it)
- ✅ YAML integration via pydantic-yaml
- ✅ Can generate from dict/JSON/YAML

### Implementation Example

```python
# src/delm/config.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Union, List
from pathlib import Path

# =============================================================================
# Schema Configuration Models
# =============================================================================

class Variable(BaseModel):
    """Schema variable definition.
    
    Defines a field to extract from text.
    
    Example:
        >>> Variable(
        ...     name="price",
        ...     description="Price mentioned in text",
        ...     data_type="number"
        ... )
    """
    name: str = Field(..., description="Field name for extracted data")
    description: str = Field(..., description="Description for LLM to understand what to extract")
    data_type: Literal["string", "number", "integer", "boolean", "date", "[string]", "[number]", "[integer]"] = Field(
        default="string",
        description="Data type of the field"
    )
    required: bool = Field(default=False, description="Whether this field must be present")
    allowed_values: Optional[List[Union[str, int, float]]] = Field(
        default=None,
        description="List of valid values for this field"
    )
    validate_in_text: bool = Field(
        default=False,
        description="Whether to validate extracted value appears in source text"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "price",
                    "description": "Price value mentioned",
                    "data_type": "number",
                    "required": False
                }
            ]
        }


class SimpleSchema(BaseModel):
    """Simple key-value extraction schema.
    
    Use when extracting a single set of properties per chunk.
    
    Example:
        >>> SimpleSchema(
        ...     variables=[
        ...         Variable(name="company", data_type="string"),
        ...         Variable(name="price", data_type="number")
        ...     ]
        ... )
    """
    type: Literal["simple"] = "simple"
    variables: List[Variable] = Field(..., min_length=1, description="List of fields to extract")


class NestedSchema(BaseModel):
    """Nested list extraction schema.
    
    Use when extracting multiple items per chunk.
    
    Example:
        >>> NestedSchema(
        ...     container_name="products",
        ...     variables=[
        ...         Variable(name="name", data_type="string", required=True),
        ...         Variable(name="price", data_type="number")
        ...     ]
        ... )
    """
    type: Literal["nested"] = "nested"
    container_name: str = Field(..., description="Key name for the list of extracted items")
    variables: List[Variable] = Field(..., min_length=1, description="List of fields to extract")


class MultipleSchema(BaseModel):
    """Multiple independent schemas.
    
    Use when extracting different types of data simultaneously.
    
    Example:
        >>> MultipleSchema(
        ...     schemas={
        ...         "products": NestedSchema(...),
        ...         "companies": NestedSchema(...)
        ...     }
        ... )
    """
    type: Literal["multiple"] = "multiple"
    schemas: dict[str, Union[SimpleSchema, NestedSchema]] = Field(
        ...,
        description="Dictionary of named sub-schemas"
    )


class SchemaConfig(BaseModel):
    """Schema configuration for extraction.
    
    Can be defined inline or loaded from file.
    
    Example:
        >>> SchemaConfig(
        ...     schema=SimpleSchema(variables=[...]),
        ...     prompt_template="Extract: {variables}\\nText: {text}"
        ... )
    """
    schema: Union[SimpleSchema, NestedSchema, MultipleSchema] = Field(
        ...,
        description="Schema definition"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Custom prompt template. Use {variables} and {text} placeholders."
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for LLM"
    )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SchemaConfig":
        """Load schema from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# =============================================================================
# Data Preprocessing Models
# =============================================================================

class ParagraphSplit(BaseModel):
    """Split text by paragraphs."""
    type: Literal["ParagraphSplit"] = "ParagraphSplit"


class FixedWindowSplit(BaseModel):
    """Split text into fixed-size windows."""
    type: Literal["FixedWindowSplit"] = "FixedWindowSplit"
    window: int = Field(default=5, ge=1, description="Number of sentences per chunk")
    stride: int = Field(default=5, ge=1, description="Number of sentences to overlap")


class RegexSplit(BaseModel):
    """Split text using regex pattern."""
    type: Literal["RegexSplit"] = "RegexSplit"
    pattern: str = Field(..., description="Regex pattern to split on")


SplitStrategy = Union[ParagraphSplit, FixedWindowSplit, RegexSplit, None]


class KeywordScorer(BaseModel):
    """Score chunks by keyword presence."""
    type: Literal["KeywordScorer"] = "KeywordScorer"
    keywords: List[str] = Field(..., min_length=1, description="Keywords to score on")


class FuzzyScorer(BaseModel):
    """Score chunks by fuzzy keyword matching."""
    type: Literal["FuzzyScorer"] = "FuzzyScorer"
    keywords: List[str] = Field(..., min_length=1, description="Keywords to score on")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Fuzzy match threshold")


ScoringStrategy = Union[KeywordScorer, FuzzyScorer, None]


class DataPreprocessing(BaseModel):
    """Data preprocessing configuration."""
    target_column: str = Field(
        default="delm_raw_data",
        description="Column containing text to process"
    )
    drop_target_column: bool = Field(
        default=False,
        description="Whether to drop target column after processing"
    )
    pandas_score_filter: Optional[str] = Field(
        default=None,
        description="Pandas query string to filter by score (e.g., 'delm_score > 0.5')"
    )
    preprocessed_data_path: Optional[Path] = Field(
        default=None,
        description="Path to pre-processed .feather file"
    )
    splitting: Optional[SplitStrategy] = Field(
        default=None,
        description="Text splitting strategy"
    )
    scoring: Optional[ScoringStrategy] = Field(
        default=None,
        description="Chunk scoring strategy"
    )


# =============================================================================
# LLM Configuration Models
# =============================================================================

Provider = Literal["openai", "anthropic", "google", "groq", "together", "fireworks"]


class LLMExtraction(BaseModel):
    """LLM API configuration."""
    provider: Provider = Field(..., description="LLM provider")
    name: str = Field(..., description="Model name (e.g., 'gpt-4o-mini')")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = very random)"
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum API retry attempts")
    base_delay: float = Field(default=1.0, ge=0.0, description="Base delay between retries (seconds)")
    batch_size: int = Field(default=10, ge=1, description="Number of chunks to process per batch")
    max_workers: int = Field(default=1, ge=1, description="Number of concurrent workers")
    dotenv_path: Optional[Path] = Field(default=Path(".env"), description="Path to .env file")
    track_cost: bool = Field(default=True, description="Whether to track API costs")
    max_budget: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum budget in dollars (requires track_cost=True)"
    )
    model_input_cost_per_1M_tokens: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Override input cost per 1M tokens"
    )
    model_output_cost_per_1M_tokens: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Override output cost per 1M tokens"
    )
    
    @validator("max_budget")
    def validate_budget_requires_tracking(cls, v, values):
        if v is not None and not values.get("track_cost", True):
            raise ValueError("max_budget requires track_cost=True")
        return v


# =============================================================================
# Semantic Cache Configuration
# =============================================================================

CacheBackend = Literal["sqlite", "lmdb", "filesystem"]
CacheSynchronous = Literal["normal", "full"]


class SemanticCache(BaseModel):
    """Semantic caching configuration."""
    backend: CacheBackend = Field(default="sqlite", description="Cache backend type")
    path: Path = Field(default=Path(".delm_cache"), description="Cache directory path")
    max_size_mb: int = Field(default=512, ge=1, description="Maximum cache size in MB")
    synchronous: CacheSynchronous = Field(
        default="normal",
        description="SQLite synchronous mode (sqlite backend only)"
    )


# =============================================================================
# Experiment Configuration
# =============================================================================

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Experiment(BaseModel):
    """Experiment tracking configuration."""
    name: Optional[str] = Field(default=None, description="Experiment name")
    directory: Path = Field(default=Path("./experiments"), description="Experiment directory")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing experiment")
    auto_checkpoint: bool = Field(default=True, description="Automatically checkpoint progress")
    use_disk_storage: bool = Field(default=True, description="Use disk storage for checkpoints")
    log_level: LogLevel = Field(default="INFO", description="Logging level")


# =============================================================================
# Main Configuration
# =============================================================================

class DELMConfig(BaseModel):
    """Complete DELM configuration.
    
    Example:
        >>> config = DELMConfig(
        ...     llm_extraction=LLMExtraction(
        ...         provider="openai",
        ...         name="gpt-4o-mini"
        ...     ),
        ...     schema=SchemaConfig(
        ...         schema=SimpleSchema(
        ...             variables=[
        ...                 Variable(name="price", data_type="number")
        ...             ]
        ...         )
        ...     )
        ... )
    """
    llm_extraction: LLMExtraction = Field(..., description="LLM API configuration")
    schema: SchemaConfig = Field(..., description="Extraction schema configuration")
    data_preprocessing: DataPreprocessing = Field(
        default_factory=DataPreprocessing,
        description="Data preprocessing configuration"
    )
    semantic_cache: SemanticCache = Field(
        default_factory=SemanticCache,
        description="Semantic cache configuration"
    )
    experiment: Experiment = Field(
        default_factory=Experiment,
        description="Experiment tracking configuration"
    )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DELMConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Validated DELMConfig instance
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "DELMConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Validated DELMConfig instance
        """
        return cls.model_validate(data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json", exclude_none=True), f)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump(mode="json", exclude_none=True)
    
    class Config:
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Raise error on unknown fields
```

### Usage Examples

#### Example 1: Simple Python Configuration

```python
from delm import DELM, DELMConfig, LLMExtraction, SchemaConfig, SimpleSchema, Variable

# Define configuration in pure Python with full type hints
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=0.0,
        batch_size=10
    ),
    schema=SchemaConfig(
        schema=SimpleSchema(
            variables=[
                Variable(
                    name="price",
                    description="Price mentioned in the text",
                    data_type="number",
                    required=False
                ),
                Variable(
                    name="company",
                    description="Company name",
                    data_type="string",
                    required=True,
                    validate_in_text=True
                )
            ]
        )
    )
)

# Use the config
delm = DELM(config)
result = delm.extract(my_dataframe)
```

**IDE Experience:**
- Hovering over `LLMExtraction` shows all parameters with descriptions
- Autocomplete suggests valid `provider` options
- Type checker catches `temperature=3.0` (out of range)
- Validation happens immediately with clear error messages

---

#### Example 2: Nested Schema with Scoring

```python
from delm import (
    DELM, DELMConfig, LLMExtraction, SchemaConfig, 
    NestedSchema, Variable, DataPreprocessing,
    ParagraphSplit, KeywordScorer
)

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="anthropic",
        name="claude-3-sonnet",
        max_budget=5.0  # Budget limit
    ),
    schema=SchemaConfig(
        schema=NestedSchema(
            container_name="commodities",
            variables=[
                Variable(
                    name="commodity_type",
                    description="Type of commodity",
                    data_type="string",
                    required=True,
                    allowed_values=["oil", "gas", "copper", "gold"]
                ),
                Variable(
                    name="price_value",
                    description="Price value",
                    data_type="number"
                )
            ]
        ),
        prompt_template="""
        Extract commodity information:
        {variables}
        
        Text:
        {text}
        """
    ),
    data_preprocessing=DataPreprocessing(
        target_column="text",
        splitting=ParagraphSplit(),
        scoring=KeywordScorer(keywords=["price", "commodity", "market"])
    )
)

delm = DELM(config)
result = delm.extract(df)
```

---

#### Example 3: Builder Pattern for Fluent API

```python
# src/delm/builders.py
class SchemaBuilder:
    """Fluent builder for schemas."""
    
    def __init__(self):
        self._variables = []
    
    def add_string(
        self,
        name: str,
        description: str,
        required: bool = False,
        allowed_values: Optional[List[str]] = None
    ) -> "SchemaBuilder":
        """Add a string variable.
        
        Args:
            name: Variable name
            description: Description for LLM
            required: Whether field is required
            allowed_values: Optional list of valid values
            
        Returns:
            Self for chaining
        """
        self._variables.append(Variable(
            name=name,
            description=description,
            data_type="string",
            required=required,
            allowed_values=allowed_values
        ))
        return self
    
    def add_number(
        self,
        name: str,
        description: str,
        required: bool = False
    ) -> "SchemaBuilder":
        """Add a number variable."""
        self._variables.append(Variable(
            name=name,
            description=description,
            data_type="number",
            required=required
        ))
        return self
    
    def add_boolean(
        self,
        name: str,
        description: str,
        required: bool = False
    ) -> "SchemaBuilder":
        """Add a boolean variable."""
        self._variables.append(Variable(
            name=name,
            description=description,
            data_type="boolean",
            required=required
        ))
        return self
    
    def build_simple(self) -> SimpleSchema:
        """Build a simple schema."""
        return SimpleSchema(variables=self._variables)
    
    def build_nested(self, container_name: str) -> NestedSchema:
        """Build a nested schema."""
        return NestedSchema(
            container_name=container_name,
            variables=self._variables
        )


class DELMBuilder:
    """Fluent builder for DELM configuration."""
    
    def __init__(self):
        self._config_dict = {}
    
    def with_model(
        self,
        provider: Provider,
        name: str,
        temperature: float = 0.0
    ) -> "DELMBuilder":
        """Configure LLM model."""
        self._config_dict["llm_extraction"] = {
            "provider": provider,
            "name": name,
            "temperature": temperature
        }
        return self
    
    def with_budget(self, max_budget: float) -> "DELMBuilder":
        """Set maximum budget."""
        self._config_dict.setdefault("llm_extraction", {})["max_budget"] = max_budget
        return self
    
    def with_schema(self, schema: Union[SimpleSchema, NestedSchema]) -> "DELMBuilder":
        """Set extraction schema."""
        self._config_dict["schema"] = {"schema": schema}
        return self
    
    def with_keywords(self, keywords: List[str]) -> "DELMBuilder":
        """Configure keyword scoring."""
        self._config_dict.setdefault("data_preprocessing", {})["scoring"] = {
            "type": "KeywordScorer",
            "keywords": keywords
        }
        return self
    
    def build(self) -> DELMConfig:
        """Build the configuration."""
        return DELMConfig.model_validate(self._config_dict)


# Usage with builder pattern
schema = (
    SchemaBuilder()
    .add_string("commodity", "Type of commodity", required=True)
    .add_number("price", "Price value")
    .add_string("unit", "Unit of measurement")
    .build_nested("commodities")
)

config = (
    DELMBuilder()
    .with_model("openai", "gpt-4o-mini", temperature=0.0)
    .with_budget(10.0)
    .with_schema(schema)
    .with_keywords(["price", "commodity"])
    .build()
)

delm = DELM(config)
```

---

#### Example 4: Helper Functions for Common Patterns

```python
# src/delm/shortcuts.py
from typing import List, Optional
from delm.config import *

def quick_extraction(
    provider: Provider,
    model: str,
    variables: List[Variable],
    schema_type: Literal["simple", "nested"] = "simple",
    container_name: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    temperature: float = 0.0,
    max_budget: Optional[float] = None
) -> DELMConfig:
    """Create a quick extraction configuration.
    
    Args:
        provider: LLM provider
        model: Model name
        variables: List of variables to extract
        schema_type: "simple" or "nested"
        container_name: Container name for nested schema
        keywords: Optional keywords for scoring
        temperature: Sampling temperature
        max_budget: Optional budget limit
        
    Returns:
        Complete DELMConfig ready to use
        
    Example:
        >>> config = quick_extraction(
        ...     provider="openai",
        ...     model="gpt-4o-mini",
        ...     variables=[
        ...         Variable(name="price", data_type="number"),
        ...         Variable(name="company", data_type="string")
        ...     ]
        ... )
    """
    # Build schema
    if schema_type == "simple":
        schema = SimpleSchema(variables=variables)
    else:
        if not container_name:
            raise ValueError("container_name required for nested schema")
        schema = NestedSchema(container_name=container_name, variables=variables)
    
    # Build preprocessing
    preprocessing_kwargs = {}
    if keywords:
        preprocessing_kwargs["scoring"] = KeywordScorer(keywords=keywords)
        preprocessing_kwargs["splitting"] = ParagraphSplit()
    
    return DELMConfig(
        llm_extraction=LLMExtraction(
            provider=provider,
            name=model,
            temperature=temperature,
            max_budget=max_budget
        ),
        schema=SchemaConfig(schema=schema),
        data_preprocessing=DataPreprocessing(**preprocessing_kwargs) if preprocessing_kwargs else DataPreprocessing()
    )


# Ultra-simple usage
config = quick_extraction(
    provider="openai",
    model="gpt-4o-mini",
    variables=[
        Variable(name="price", data_type="number"),
        Variable(name="company", data_type="string")
    ],
    keywords=["price", "revenue"]
)

delm = DELM(config)
result = delm.extract(df)
```

---

#### Example 5: YAML ↔ Python Interoperability

```python
# Load from YAML
config = DELMConfig.from_yaml("my_config.yaml")

# Modify in Python with type safety
config.llm_extraction.temperature = 0.5
config.llm_extraction.max_budget = 10.0
config.schema.schema.variables.append(
    Variable(name="new_field", data_type="string")
)

# Save back to YAML
config.to_yaml("modified_config.yaml")

# Or use directly
delm = DELM(config)
result = delm.extract(df)
```

---

#### Example 6: Programmatic Config Generation

```python
def create_commodity_config(
    commodities: List[str],
    model: str = "gpt-4o-mini",
    budget: float = 5.0
) -> DELMConfig:
    """Generate config for commodity extraction.
    
    Args:
        commodities: List of commodity types to extract
        model: LLM model to use
        budget: Maximum budget
        
    Returns:
        Configured DELMConfig
    """
    return DELMConfig(
        llm_extraction=LLMExtraction(
            provider="openai",
            name=model,
            max_budget=budget
        ),
        schema=SchemaConfig(
            schema=NestedSchema(
                container_name="commodities",
                variables=[
                    Variable(
                        name="type",
                        description="Type of commodity",
                        data_type="string",
                        required=True,
                        allowed_values=commodities
                    ),
                    Variable(
                        name="price",
                        description="Price value",
                        data_type="number"
                    ),
                    Variable(
                        name="unit",
                        description="Unit of measurement",
                        data_type="string"
                    )
                ]
            )
        ),
        data_preprocessing=DataPreprocessing(
            splitting=ParagraphSplit(),
            scoring=KeywordScorer(keywords=["price"] + commodities)
        )
    )


# Use it
config = create_commodity_config(
    commodities=["oil", "gas", "copper", "gold"],
    budget=10.0
)
```

---

## Option 2: Direct Constructor with Type Hints

### Implementation

```python
from delm import DELM
from typing import List, Literal, Optional

# Direct initialization with full type hints
delm = DELM(
    # LLM settings
    provider="openai",  # type: Literal["openai", "anthropic", ...]
    model="gpt-4o-mini",  # type: str
    temperature=0.0,  # type: float (0.0-2.0)
    max_budget=5.0,  # type: Optional[float]
    
    # Schema (inline dict with validation)
    schema={
        "type": "nested",
        "container_name": "commodities",
        "variables": [
            {
                "name": "commodity_type",
                "description": "Type of commodity",
                "data_type": "string",
                "required": True,
                "allowed_values": ["oil", "gas", "copper"]
            },
            {
                "name": "price",
                "data_type": "number"
            }
        ]
    },
    
    # Preprocessing
    target_column="text",
    splitting="ParagraphSplit",
    keywords=["price", "commodity"]
)

result = delm.extract(df)
```

**Pros:**
- Very concise
- All in one place
- Still type-safe

**Cons:**
- Nested dicts lose some type safety
- Harder to compose/reuse
- Less structured

---

## Comparison Matrix

| Feature | Pydantic Models | Builder Pattern | Direct Constructor | YAML Only |
|---------|----------------|-----------------|-------------------|-----------|
| Type Safety | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| IDE Autocomplete | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Validation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Composability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Simple Cases | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Complex Cases | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Learning Curve | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| YAML Interop | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Runtime Errors | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Documentation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Recommended Hybrid Approach

Support **all three levels** to serve different user needs:

### Level 1: YAML (Declarative, Beginners)
```yaml
# config.yaml
llm_extraction:
  provider: openai
  name: gpt-4o-mini
schema:
  type: simple
  variables:
    - name: price
      data_type: number
```

```python
delm = DELM.from_yaml("config.yaml")
```

### Level 2: Helper Functions (Quick Python)
```python
from delm import quick_extraction, Variable

config = quick_extraction(
    provider="openai",
    model="gpt-4o-mini",
    variables=[Variable(name="price", data_type="number")]
)
delm = DELM(config)
```

### Level 3: Full Pydantic (Power Users)
```python
from delm import DELMConfig, LLMExtraction, SchemaConfig, SimpleSchema, Variable

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=0.0,
        max_budget=5.0
    ),
    schema=SchemaConfig(
        schema=SimpleSchema(
            variables=[
                Variable(
                    name="price",
                    description="Price mentioned",
                    data_type="number"
                )
            ]
        )
    )
)
delm = DELM(config)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. ✅ Add pydantic as dependency
2. ✅ Create Pydantic models for all config sections
3. ✅ Add `.from_yaml()`, `.to_yaml()`, `.from_dict()`, `.to_dict()`
4. ✅ Maintain backward compatibility with existing YAML parsing

### Phase 2: Builders & Helpers (Week 3)
1. ✅ Create `SchemaBuilder` class
2. ✅ Create `DELMBuilder` class  
3. ✅ Add `quick_extraction()` helper
4. ✅ Add convenience functions for common patterns

### Phase 3: DELM Integration (Week 4)
1. ✅ Update `DELM.__init__()` to accept `DELMConfig`
2. ✅ Update `DELM.from_config()` to handle both YAML and Pydantic
3. ✅ Add validation error handling with helpful messages
4. ✅ Add config serialization methods to DELM

### Phase 4: Documentation & Examples (Week 5)
1. ✅ Add docstrings with examples to all models
2. ✅ Create type stubs for better IDE support
3. ✅ Write tutorial notebooks showing all patterns
4. ✅ Generate API docs from Pydantic schemas

### Phase 5: Advanced Features (Week 6+)
1. ✅ Add config templates (e.g., `DELMConfig.template_commodity_extraction()`)
2. ✅ Add config diff/merge utilities
3. ✅ Add config validation CLI tool
4. ✅ Add JSON Schema export for external tooling

---

## IDE Support Examples

### VSCode with Pylance

**Hovering over `LLMExtraction`:**
```
class LLMExtraction(BaseModel):
    """LLM API configuration."""
    
    provider: Literal["openai", "anthropic", "google", ...]
        LLM provider
    
    name: str
        Model name (e.g., 'gpt-4o-mini')
    
    temperature: float = 0.0
        Sampling temperature (0.0 = deterministic, 2.0 = very random)
        Range: 0.0-2.0
```

**Autocomplete for `provider`:**
```
provider: ▼
  - openai
  - anthropic
  - google
  - groq
  - together
  - fireworks
```

**Type error catching:**
```python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=3.0  # ❌ Error: Value must be <= 2.0
    )
)
```

### PyCharm Support

- ✅ Parameter hints in function calls
- ✅ Quick documentation on hover
- ✅ Type checking in real-time
- ✅ Refactoring support
- ✅ Go-to-definition for all config classes

---

## Migration Examples

### Migrate YAML to Python

```python
# Before: YAML file
"""
llm_extraction:
  provider: openai
  name: gpt-4o-mini
schema:
  spec_path: schema.yaml
"""

# After: Python with type hints
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini"
    ),
    schema=SchemaConfig.from_yaml("schema.yaml")  # Can still load schema from YAML
)

# Or fully in Python
config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini"
    ),
    schema=SchemaConfig(
        schema=SimpleSchema(
            variables=[
                Variable(name="price", data_type="number")
            ]
        )
    )
)
```

### Auto-migration Tool

```python
# tools/migrate_config.py
from delm.config import DELMConfig

# Load old YAML config
config = DELMConfig.from_yaml("old_config.yaml")

# Generate Python code
python_code = config.to_python_code()
print(python_code)

# Output:
"""
from delm import DELMConfig, LLMExtraction, SchemaConfig, SimpleSchema, Variable

config = DELMConfig(
    llm_extraction=LLMExtraction(
        provider="openai",
        name="gpt-4o-mini",
        temperature=0.0
    ),
    schema=SchemaConfig(
        schema=SimpleSchema(
            variables=[
                Variable(
                    name="price",
                    description="Price mentioned",
                    data_type="number"
                )
            ]
        )
    )
)
"""
```

---

## Error Messages: Before vs After

### Before (Dict-based config)
```
KeyError: 'llm_extraction'
  File "delm/config.py", line 123, in __init__
    self.provider = config["llm_extraction"]["provider"]
```

### After (Pydantic config)
```
ValidationError: 2 validation errors for DELMConfig
llm_extraction.temperature
  ensure this value is less than or equal to 2.0 (type=value_error.number.not_le; limit_value=2.0)
llm_extraction.max_budget
  max_budget requires track_cost=True (type=value_error)
```

Much clearer what's wrong and how to fix it!

---

## Best Practices Guide

### For Simple Use Cases: Use YAML
```yaml
# config.yaml
llm_extraction:
  provider: openai
  name: gpt-4o-mini
schema:
  type: simple
  variables:
    - name: price
      data_type: number
```

```python
delm = DELM.from_yaml("config.yaml")
```

### For Programmatic Generation: Use Pydantic
```python
def create_config_for_model(model_name: str) -> DELMConfig:
    return DELMConfig(
        llm_extraction=LLMExtraction(
            provider="openai",
            name=model_name
        ),
        schema=load_standard_schema()
    )
```

### For Quick Experiments: Use Helpers
```python
config = quick_extraction(
    provider="openai",
    model="gpt-4o-mini",
    variables=[Variable(name="price", data_type="number")]
)
```

### For Complex Workflows: Use Builders
```python
config = (
    DELMBuilder()
    .with_model("openai", "gpt-4o-mini")
    .with_budget(10.0)
    .with_schema(my_schema)
    .with_keywords(["price"])
    .build()
)
```

---

## Testing Strategy

### Type Checking Tests
```python
# tests/type_checking/test_config_types.py
from delm import DELMConfig, LLMExtraction

def test_type_hints():
    # This should pass type checking
    config = DELMConfig(
        llm_extraction=LLMExtraction(
            provider="openai",
            name="gpt-4o-mini"
        ),
        schema=...
    )
    
    # This should fail type checking
    config = DELMConfig(
        llm_extraction=LLMExtraction(
            provider="invalid_provider",  # Type error!
            name=123  # Type error!
        )
    )
```

### Validation Tests
```python
def test_validation():
    with pytest.raises(ValidationError) as exc_info:
        DELMConfig(
            llm_extraction=LLMExtraction(
                provider="openai",
                name="gpt-4o-mini",
                temperature=3.0  # Out of range
            )
        )
    
    assert "temperature" in str(exc_info.value)
    assert "2.0" in str(exc_info.value)
```

### Round-trip Tests
```python
def test_yaml_python_roundtrip():
    # Load from YAML
    config1 = DELMConfig.from_yaml("test_config.yaml")
    
    # Convert to Python and back to YAML
    config1.to_yaml("temp_config.yaml")
    config2 = DELMConfig.from_yaml("temp_config.yaml")
    
    # Should be identical
    assert config1 == config2
```

---

## Conclusion

### Recommendation: **Pydantic-Based with Helpers**

**Why:**
1. ✅ Best-in-class type safety and IDE support
2. ✅ Industry standard (FastAPI, LangChain, Prefect use it)
3. ✅ Excellent validation with clear error messages
4. ✅ YAML interoperability built-in
5. ✅ Can add helpers for convenience
6. ✅ Extensible for future features

**Implementation Order:**
1. **Week 1-2**: Core Pydantic models
2. **Week 3**: Helper functions & shortcuts
3. **Week 4**: DELM integration
4. **Week 5**: Documentation & examples
5. **Week 6+**: Advanced features

**User Journey:**
- Beginners: Use YAML
- Intermediate: Use helper functions
- Advanced: Use full Pydantic models
- All users: Get great IDE support and validation

This gives the best of all worlds!


