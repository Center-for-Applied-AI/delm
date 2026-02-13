# DELMConfig

Configuration objects for DELM pipelines.

## DELMConfig

Main configuration object containing all settings.

```python
from delm import DELMConfig

config = DELMConfig(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    **kwargs
)
```

**Parameters:** Same as [`DELM` constructor](delm.md#constructor-parameters)

### Methods

#### from_yaml()

Load configuration from YAML file.

```python
config = DELMConfig.from_yaml("config.yaml")
```

---

#### to_yaml()

Save configuration to YAML file.

```python
config.to_yaml("config.yaml")
```

---

#### to_dict()

Convert to dictionary.

```python
config_dict = config.to_dict()
```

---

#### validate()

Validate configuration (called automatically on construction).

```python
config.validate()  # Raises ValueError if invalid
```

## Sub-Configurations

### LLMExtractionConfig

LLM and extraction settings.

**Attributes:**
- `provider`, `model`, `temperature`
- `batch_size`, `max_workers`, `max_retries`, `base_delay`, `tokens_per_minute`, `requests_per_minute`
- `max_completion_tokens`
- `track_cost`, `max_budget`
- `model_input_cost_per_1M_tokens`, `model_output_cost_per_1M_tokens`
- `prompt_template`, `system_prompt`

---

### DataPreprocessingConfig

Data loading and preprocessing settings.

**Attributes:**
- `target_column`, `drop_target_column`
- `splitting_strategy`, `relevance_scorer`, `score_filter`

---

### SemanticCacheConfig

Caching settings.

**Attributes:**
- `backend` (`"sqlite"`, `"lmdb"`, `"json"`)
- `path`, `max_size_mb`, `synchronous`

## Example

```python
from delm import DELM, DELMConfig

# Create and save config
delm = DELM(schema=schema, model="gpt-4o-mini")
delm.config.to_yaml("experiment_config.yaml")

# Load and reuse config
config = DELMConfig.from_yaml("experiment_config.yaml")
delm2 = DELM.from_config(config, model="claude-3-5-sonnet-20241022")  # Override model
```
