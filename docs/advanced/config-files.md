# Configuration Files

Use YAML configuration files to define reusable, version-controlled extraction pipelines.

## When to Use Config Files

**Primary API**: We recommend using the **Python API** (`DELM()`) for most use cases - it's more intuitive and has better IDE support.

**Use YAML configs when:**
- You want to version control extraction configurations alongside code
- You need to share configs across different scripts or team members
- You're running experiments with many configuration variations
- You want to separate configuration from code logic

## Python API (Recommended)

```python
from delm import DELM, Schema, ExtractionVariable

schema = Schema.simple([
    ExtractionVariable(name="price", description="Price value", data_type="number")
])

delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0
)

results = delm.extract("data.csv")
```

## YAML Config API

### Loading from YAML

```python
from delm import DELM

# Load config from YAML
delm = DELM.from_config("config.yaml")

results = delm.extract("data.csv")
```

### Config File Structure

DELM config files use a **flat structure** with all parameters at the top level:

```yaml
# config.yaml

# Schema (REQUIRED) - can be inline dict or path to schema file
schema:
  schema_type: "simple"
  variables:
    - name: "price"
      description: "Price value mentioned"
      data_type: "number"

# OR reference external schema file
# schema: "schema.yaml"

# LLM Settings
provider: "openai"              # REQUIRED: "openai", "anthropic", "google", "groq", etc.
model: "gpt-4o-mini"            # REQUIRED: Model identifier
temperature: 0.0                 # Default: 0.0, Range: 0.0-2.0

# Processing Settings
batch_size: 10                   # Default: 10, chunks per batch
max_workers: 1                   # Default: 1, concurrent workers per batch
max_retries: 3                   # Default: 3, API retry attempts
base_delay: 1.0                  # Default: 1.0, seconds between retries
rate_limit_tokens: null          # Default: null, max tokens per rate limit period
rate_limit_requests: null        # Default: null, max requests per rate limit period
rate_limit_period_seconds: 60.0  # Default: 60.0, rate limit window in seconds
max_completion_tokens: 4096      # Default: 4096, max completion tokens per request

# API Pass-through
api_kwargs: {}                   # Default: {}, extra kwargs sent to the LLM API (e.g. {store: false})

# Cost Management
track_cost: true                 # Default: true
max_budget: null                 # Default: null, max spend in dollars (requires track_cost: true)
model_input_cost_per_1M_tokens: null   # Default: auto-detected via tokencost
model_output_cost_per_1M_tokens: null  # Default: auto-detected via tokencost

# Data Preprocessing  
target_column: "text"            # Default: "text", input text column name
drop_target_column: false        # Default: false, whether to drop target column after processing
score_filter: null               # Default: null, pandas query like "delm_score >= 0.7"

# Text Splitting (optional)
splitting_strategy:
  type: "ParagraphSplit"         # Options: "ParagraphSplit", "FixedWindowSplit", "RegexSplit", null
  # window: 5                    # For FixedWindowSplit only
  # stride: 5                    # For FixedWindowSplit only
  # pattern: "\n\n"              # For RegexSplit only

# Relevance Scoring (optional)
relevance_scorer:
  type: "KeywordScorer"          # Options: "KeywordScorer", "FuzzyScorer", null
  keywords: ["price", "forecast"]  # For KeywordScorer/FuzzyScorer

# Prompt Customization (optional)
prompt_template: |
  Extract the following information from the text:
  
  {variables}
  
  Text to analyze:
  {text}

system_prompt: "You are a precise data-extraction assistant."

# Caching Settings
cache_backend: "sqlite"          # Default: "sqlite", Options: "sqlite", "lmdb", "filesystem"
cache_path: ".delm/cache"        # Default: ".delm/cache"
cache_max_size_mb: 512           # Default: 512
cache_synchronous: "normal"      # Default: "normal", Options: "normal", "full" (SQLite only)
```

## Separate Schema Files

You can define schemas in separate YAML files:

**config.yaml:**
```yaml
schema: "schema.yaml"  # Path to schema file
provider: "openai"
model: "gpt-4o-mini"
# ... other settings
```

**schema.yaml:**
```yaml
schema_type: "simple"
variables:
  - name: "price"
    description: "Price value mentioned in text"
    data_type: "number"
    required: false
  
  - name: "company"
    description: "Company name if mentioned"
    data_type: "string"
    required: false
    validate_in_text: true
```

See the [Schemas documentation](../user-guide/schemas.md) for complete schema specification details.

## Complete Example

**config.yaml:**
```yaml
# Schema definition
schema:
  schema_type: "nested"
  container_name: "commodities"
  variables:
    - name: "commodity_type"
      description: "Type of commodity mentioned"
      data_type: "string"
      required: true
      allowed_values: ["oil", "gas", "gold", "copper"]
      validate_in_text: true
    
    - name: "price"
      description: "Price value if mentioned"
      data_type: "number"
      required: false
    
    - name: "unit"
      description: "Unit of measurement (barrel, ounce, ton)"
      data_type: "string"
      required: false

# LLM configuration
provider: "openai"
model: "gpt-4o-mini"
temperature: 0.0
batch_size: 20
max_workers: 4
max_retries: 3
base_delay: 1.0
rate_limit_tokens: 500000
rate_limit_requests: 500
rate_limit_period_seconds: 60.0

# Cost tracking
track_cost: true
max_budget: 50.0

# Preprocessing
target_column: "text"
drop_target_column: false

splitting_strategy:
  type: "ParagraphSplit"

relevance_scorer:
  type: "KeywordScorer"
  keywords: ["price", "forecast", "guidance", "commodity"]

score_filter: "delm_score >= 0.5"

# Custom prompts
prompt_template: |
  Extract commodity price information from the following text.
  
  {variables}
  
  IMPORTANT: Only extract information explicitly mentioned in the text.
  
  Text:
  {text}

system_prompt: "You are a commodity price extraction specialist. Extract only factual information explicitly stated in the text."

# Caching
cache_backend: "sqlite"
cache_path: ".delm/cache"
cache_max_size_mb: 1024
```

**Usage:**
```python
from delm import DELM

# Load and run
delm = DELM.from_config("config.yaml")
results = delm.extract("data/reports.csv")

# Get cost summary
cost_summary = delm.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost']:.2f}")
```

## Configuration Reference

### Schema (Required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | dict or string | Schema definition (inline dict or path to YAML file) |

### LLM Extraction Config

Contains all LLM-related settings including provider, model, prompts, processing, and cost tracking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | string | **REQUIRED** | LLM provider ("openai", "anthropic", "google", "groq", "together", "fireworks") |
| `model` | string | **REQUIRED** | Model identifier (e.g., "gpt-4o-mini", "claude-3-sonnet") |
| `temperature` | float | 0.0 | Sampling temperature (0.0-2.0) |
| `prompt_template` | string | (default) | User prompt template with `{variables}` and `{text}` placeholders |
| `system_prompt` | string | "You are a precise data-extraction assistant." | System prompt sent to LLM |
| `max_retries` | int | 3 | Number of retry attempts on API failure |
| `batch_size` | int | 10 | Number of chunks processed per batch |
| `max_workers` | int | 1 | Concurrent workers (within each batch) |
| `base_delay` | float | 1.0 | Seconds between retry attempts |
| `rate_limit_tokens` | int | null | Max tokens per rate limit period |
| `rate_limit_requests` | int | null | Max requests per rate limit period |
| `rate_limit_period_seconds` | float | 60.0 | Rate limit window in seconds |
| `max_completion_tokens` | int | 4096 | Maximum completion tokens per request |
| `api_kwargs` | dict | {} | Extra kwargs passed through to the LLM API call |
| `track_cost` | bool | true | Enable cost tracking |
| `max_budget` | float | null | Maximum budget in dollars (requires `track_cost: true`) |
| `model_input_cost_per_1M_tokens` | float | null | Custom input token cost (auto-detected via tokencost if null) |
| `model_output_cost_per_1M_tokens` | float | null | Custom output token cost (auto-detected via tokencost if null) |

### Data Preprocessing Config

Controls text splitting, relevance scoring, and filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_column` | string | "text" | Input text column name |
| `drop_target_column` | bool | false | Drop target column after splitting |
| `splitting_strategy` | dict | null | Text splitting configuration (e.g., `{"type": "ParagraphSplit"}`) |
| `relevance_scorer` | dict | null | Relevance scoring configuration (e.g., `{"type": "KeywordScorer", "keywords": [...]}`) |
| `score_filter` | string | null | Pandas query to filter chunks (e.g., "delm_score >= 0.7") |

### Semantic Cache Config

Controls caching of LLM responses.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_backend` | string | "sqlite" | Cache backend ("sqlite", "lmdb", "filesystem") |
| `cache_path` | string | ".delm/cache" | Cache storage path |
| `cache_max_size_mb` | int | 512 | Maximum cache size in MB before pruning |
| `cache_synchronous` | string | "normal" | SQLite synchronous mode ("normal", "full") |
