# DELM API Improvement Proposal

## Executive Summary

After reviewing all test examples (excluding unit tests), I've identified several opportunities to simplify and improve the DELM API while maintaining full functionality. This document outlines current patterns, pain points, and concrete recommendations for a cleaner, more user-friendly API.

---

## Current API Analysis

### Current Usage Pattern

The typical DELM workflow currently looks like this:

```python
# 1. Load configuration from TWO separate files
config = DELMConfig.from_yaml("config.yaml")  # References schema_spec.yaml inside

# 2. Initialize DELM with many parameters
delm = DELM(
    config=config,
    experiment_name="my_experiment",
    experiment_directory=Path("./experiments"),
    overwrite_experiment=True,
    auto_checkpoint_and_resume_experiment=True,
    use_disk_storage=True,
    console_log_level="INFO"
)

# 3. Two-step processing
delm.prep_data(data_source)
result_df = delm.process_via_llm()

# 4. Get results
cost_summary = delm.get_cost_summary()
```

### File Structure Issues

**Current**: Two separate files required
```yaml
# config.yaml
llm_extraction:
  provider: openai
  name: gpt-4o-mini
schema:
  spec_path: "schema_spec.yaml"  # <-- Separate file reference

# schema_spec.yaml
schema_type: nested
container_name: commodities
variables:
  - name: commodity_type
    description: "Type of commodity"
    data_type: string
```

**Problem**: Users must manage two files and ensure the path reference is correct. This is unnecessarily complex.

---

## Common Use Cases from Tests

After reviewing all test directories, here are the most common patterns:

### 1. **Simple DataFrame Extraction** (80% of cases)
- Load data from parquet/CSV
- Extract structured information
- Get results as DataFrame

### 2. **Directory-Based Extraction** (15% of cases)
- Process all files in a directory (txt, csv, pdf)
- Each file becomes a record

### 3. **Performance/Cost Estimation** (5% of cases)
- Estimate costs before full run
- Validate extraction quality against labeled data

### 4. **Parameter Experiments** (Rare)
- Test different temperatures
- Compare different models
- Try different prompts

---

## Proposed Improvements

### ✅ **Improvement 1: Merge config.yaml and schema_spec.yaml**

**Current** (2 files):
```yaml
# config.yaml
schema:
  spec_path: "schema_spec.yaml"

# schema_spec.yaml
schema_type: nested
container_name: commodities
variables: [...]
```

**Proposed** (1 file):
```yaml
# delm_config.yaml
llm_extraction:
  provider: openai
  name: gpt-4o-mini

schema:
  type: nested  # Changed from schema_type
  container_name: commodities
  variables:
    - name: commodity_type
      description: "Type of commodity"
      data_type: string
      required: true
```

**Benefits**:
- Single source of truth
- No path resolution issues
- Easier to version control
- Clearer mental model

**Migration Path**:
- Keep `spec_path` as fallback for backward compatibility
- If `spec_path` is null/missing, read schema inline
- Eventually deprecate `spec_path`

---

### ✅ **Improvement 2: Simplify Initialization**

**Current** (too many params):
```python
delm = DELM(
    config=config,
    experiment_name="my_experiment",
    experiment_directory=Path("./experiments"),
    overwrite_experiment=True,
    auto_checkpoint_and_resume_experiment=True,
    use_disk_storage=True,
    console_log_level="INFO"
)
```

**Proposed Option A**: Group experiment settings in config
```yaml
# In config file
experiment:
  name: my_experiment
  directory: ./experiments
  overwrite: true
  auto_checkpoint: true
  use_disk_storage: true
  log_level: INFO
```

```python
# Simpler initialization
delm = DELM.from_config("delm_config.yaml")
# or
delm = DELM(config)
```

**Proposed Option B**: Minimal required params + method chaining
```python
delm = (
    DELM.from_config("delm_config.yaml")
    .with_experiment("my_experiment")
    .with_logging("INFO")
)
```

**Proposed Option C**: Most Pythonic - Sensible Defaults
```python
# Minimal for most cases
delm = DELM.from_config("delm_config.yaml")

# Advanced users can override
delm = DELM.from_config(
    "delm_config.yaml",
    experiment_name="custom",
    overwrite=True
)
```

**Recommendation**: Go with Option C - it's the most Pythonic and maintains simplicity while allowing flexibility.

---

### ✅ **Improvement 3: Single-Step Processing**

**Current** (two methods):
```python
delm.prep_data(data_source)
result_df = delm.process_via_llm()
```

**Proposed**: Add convenience method for common case
```python
# New simple method for most users
result_df = delm.extract(data_source)

# Power users can still use two-step if needed
delm.prep_data(data_source)
delm.visualize_chunks()  # Inspect before processing
result_df = delm.process_via_llm()
```

**Benefits**:
- 80% of users just want results
- Reduce cognitive load
- Still allow inspection for advanced users

---

### ✅ **Improvement 4: Better Return Types**

**Current**: Returns just DataFrame
```python
result_df = delm.process_via_llm()
cost_summary = delm.get_cost_summary()  # Separate call
```

**Proposed**: Return rich result object
```python
result = delm.extract(data_source)

# Result is a DelmResult object with:
result.data          # DataFrame
result.cost          # Cost summary dict
result.metrics       # Processing metrics
result.to_csv(path)  # Convenience methods
result.to_parquet(path)
```

**Benefits**:
- All related data together
- Prevents "forgot to get cost" issues
- More discoverable API
- Enables future additions without breaking changes

---

### ✅ **Improvement 5: Smarter Experiment Management**

**Current**: Always requires experiment setup
```python
delm = DELM(
    config=config,
    experiment_name="test",
    experiment_directory=Path("./experiments"),
    overwrite_experiment=True
)
```

**Proposed**: Make experiments optional for simple cases
```python
# Simple case - no experiment management
delm = DELM.from_config("config.yaml")
result = delm.extract(df)

# When you want experiment management
delm = DELM.from_config(
    "config.yaml",
    experiment="my_experiment_name"  # Auto-creates, auto-resumes
)

# Advanced control
delm = DELM.from_config(
    "config.yaml",
    experiment="my_experiment",
    experiment_dir="./experiments",
    overwrite=True
)
```

**Benefits**:
- Beginners can ignore experiment complexity
- Advanced users get full control
- Auto-resume by default (safer)

---

### ✅ **Improvement 6: Better Defaults**

Many config options should have smarter defaults:

**Current**: Users must specify everything
```yaml
llm_extraction:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.0
  max_retries: 3
  batch_size: 10
  max_workers: 1
  base_delay: 1.0
  dotenv_path: ".env"
  track_cost: true
```

**Proposed**: Only require essentials
```yaml
llm_extraction:
  provider: openai
  name: gpt-4o-mini
  # Everything else uses smart defaults:
  # - temperature: 0.0 (deterministic by default)
  # - max_retries: 3 (reasonable resilience)
  # - batch_size: 10 (good balance)
  # - max_workers: 1 (safe default)
  # - dotenv_path: ".env" (standard location)
  # - track_cost: true (why wouldn't you?)
```

---

### ✅ **Improvement 7: Better Type Hints and Validation**

**Current**: Data source can be many things
```python
delm.prep_data(data_source)  # DataFrame? Path? str? dict?
```

**Proposed**: Clear unions with good error messages
```python
from typing import Union
from pathlib import Path

def extract(
    self, 
    data: Union[pd.DataFrame, Path, str, dict],
    sample_size: Optional[int] = None
) -> DelmResult:
    """
    Extract structured data from text.
    
    Args:
        data: Input data source:
            - pd.DataFrame: Process dataframe directly
            - Path/str: Process files in directory
            - dict: Process single document {"text": "..."}
        sample_size: Number of records to process (None = all)
    
    Returns:
        DelmResult with extracted data, costs, and metrics
    """
```

---

## Proposed New API Examples

### Example 1: Simple Extraction (Most Common)

```python
from delm import DELM

# Minimal setup
delm = DELM.from_config("delm_config.yaml")

# One-line extraction
result = delm.extract(my_dataframe)

# Access results
print(result.data)
print(result.cost)
result.to_csv("output.csv")
```

### Example 2: Directory Processing

```python
from delm import DELM
from pathlib import Path

delm = DELM.from_config("delm_config.yaml")

# Extract from all PDFs in directory
result = delm.extract(Path("./documents"), sample_size=10)
```

### Example 3: Cost Estimation Before Full Run

```python
from delm import DELM

delm = DELM.from_config("delm_config.yaml")

# Estimate cost before full run
estimate = delm.estimate_cost(my_dataframe, sample_size=5)
print(f"Estimated cost: ${estimate.total_cost:.2f}")

# Proceed if acceptable
if estimate.total_cost < 10.0:
    result = delm.extract(my_dataframe)
```

### Example 4: Experiment with Checkpointing

```python
from delm import DELM

# Auto-creates experiment, auto-resumes on crash
delm = DELM.from_config(
    "delm_config.yaml",
    experiment="large_extraction_v1"
)

result = delm.extract(large_dataframe)
# If it crashes, re-run same code - it will resume
```

### Example 5: Advanced - Inspect Before Processing

```python
from delm import DELM

delm = DELM.from_config("delm_config.yaml")

# Prepare data
delm.prep_data(my_dataframe)

# Inspect chunks
print(f"Created {len(delm.prepped_data)} chunks")
print(f"Average score: {delm.prepped_data['delm_score'].mean()}")

# Visualize
delm.plot_score_distribution()

# Process if satisfied
result = delm.process_via_llm()
```

---

## Configuration Schema: Unified Version

Here's what the merged config would look like:

```yaml
# =============================================================================
# delm_config.yaml - Unified Configuration
# =============================================================================

# LLM Settings (REQUIRED)
llm_extraction:
  provider: openai        # REQUIRED
  name: gpt-4o-mini       # REQUIRED
  temperature: 0.0        # Optional, default: 0.0
  batch_size: 10          # Optional, default: 10
  max_workers: 1          # Optional, default: 1
  dotenv_path: .env       # Optional, default: .env
  track_cost: true        # Optional, default: true
  max_budget: null        # Optional, default: null (no limit)

# Data Preprocessing (OPTIONAL)
data_preprocessing:
  target_column: text     # Optional, default: "delm_raw_data"
  splitting:
    type: ParagraphSplit  # Optional, default: null (no splitting)
  scoring:
    type: KeywordScorer   # Optional, default: null (no scoring)
    keywords: 
      - price
      - forecast

# Schema Definition (REQUIRED) - NOW INLINE!
schema:
  type: nested            # Required: simple, nested, or multiple
  container_name: commodities  # Required for nested/multiple
  
  # Optional: Custom prompts
  system_prompt: "You are a data extraction assistant."
  prompt_template: |
    Extract the following information:
    {variables}
    
    Text:
    {text}
  
  # Schema variables
  variables:
    - name: commodity_type
      description: "Type of commodity mentioned"
      data_type: string
      required: true
      allowed_values: ["oil", "gas", "copper", "gold"]
    
    - name: price_value
      description: "Numeric price value"
      data_type: number
      required: false

# Experiment Settings (OPTIONAL) - NEW!
experiment:
  name: null              # Optional, default: null (no experiments)
  directory: ./experiments  # Optional, default: ./experiments
  overwrite: false        # Optional, default: false
  auto_checkpoint: true   # Optional, default: true
  use_disk_storage: true  # Optional, default: true
  log_level: INFO         # Optional, default: INFO
```

---

## Migration Strategy

### Phase 1: Add New Features (Non-Breaking)
1. Support inline schema in config (keep `spec_path` as fallback)
2. Add `extract()` convenience method
3. Add `DelmResult` return type
4. Add experiment settings to config

### Phase 2: Improve Defaults (Non-Breaking)
1. Make more parameters optional with smart defaults
2. Improve type hints and validation
3. Better error messages

### Phase 3: Deprecation (Breaking, but with warnings)
1. Deprecate separate `schema_spec.yaml` (show warning, still works)
2. Deprecate many constructor params in favor of config
3. Provide migration tool/script

### Phase 4: Clean Up (Major Version Bump)
1. Remove deprecated features
2. Simplify internal APIs
3. Update all documentation

---

## Backward Compatibility Approach

Keep old API working with warnings:

```python
# OLD API - Still works but shows deprecation warning
config = DELMConfig.from_yaml("config.yaml")  # config still has spec_path
delm = DELM(
    config=config,
    experiment_name="test",
    experiment_directory=Path("./experiments"),
    # ... many params
)
delm.prep_data(df)
result_df = delm.process_via_llm()

# NEW API - Recommended
delm = DELM.from_config("delm_config.yaml")
result = delm.extract(df)
```

---

## Implementation Priority

### High Priority (Do First)
1. ✅ **Merge schema into config** - Biggest pain point, clear win
2. ✅ **Add `extract()` method** - Makes API much simpler
3. ✅ **Move experiment params to config** - Cleaner initialization

### Medium Priority
4. **Add `DelmResult` return type** - Better organization
5. **Improve defaults** - Less boilerplate
6. **Better type hints** - Better IDE support

### Low Priority (Nice to Have)
7. **Method chaining** - Stylistic preference
8. **Visualization helpers** - Nice for exploration
9. **Config validation tool** - Catch errors early

---

## Testing Implications

### What Needs Testing
1. Inline schema works same as external `spec_path`
2. New `extract()` method produces same results as two-step
3. Experiment settings in config work correctly
4. Backward compatibility maintained
5. Good error messages for common mistakes

### Test Coverage
- Unit tests for new config parsing
- Integration tests for new workflows
- Regression tests for old API
- Performance tests (ensure no slowdown)

---

## Documentation Updates Needed

1. **Getting Started Guide**: Show new simple API first
2. **Migration Guide**: How to move from old to new API
3. **API Reference**: Document new methods and return types
4. **Config Reference**: Update to show inline schema
5. **Examples**: Rewrite to show new patterns
6. **Changelog**: Clear explanation of changes

---

## Questions to Answer

### 1. Should we keep two-step processing?
**Recommendation**: YES - Some users need to inspect prepped data before processing.

### 2. Should experiments be always-on or opt-in?
**Recommendation**: OPT-IN - Most users don't need experiment tracking for simple extractions.

### 3. How aggressive should we be with defaults?
**Recommendation**: MODERATE - Have good defaults but allow override. Don't hide too much magic.

### 4. Should we support both inline and external schemas forever?
**Recommendation**: Support both for 2-3 major versions, then deprecate external. Inline is simpler for 90% of cases.

### 5. What about very complex multi-schema configs?
**Recommendation**: For "multiple" schema type, inline might get verbose. Consider:
- Keep external schema support for complex cases
- Provide schema composition utilities
- Allow mixing: some inline, some external

---

## Conclusion

The proposed improvements make DELM:
- **Simpler**: 1 file instead of 2, fewer params
- **More Pythonic**: Better defaults, clear types, single-step extraction
- **More Flexible**: Can still do complex things when needed
- **More Maintainable**: Clearer mental model, easier to extend

The migration path maintains backward compatibility while guiding users toward the better API.

### Recommended Next Steps

1. **Discuss and refine** this proposal with the team
2. **Start with Improvement #1** (merge configs) - clear win, moderate effort
3. **Prototype Improvement #2** (simple init) and #3 (extract method)
4. **Get user feedback** on prototypes
5. **Iterate** based on feedback
6. **Roll out gradually** with good migration docs

---

## Appendix: Alternative Considered

### Alternative: Configuration as Code (Python)

Instead of YAML, use Python for configuration:

```python
from delm import DELM, Schema, Variable

schema = Schema.nested(
    container_name="commodities",
    variables=[
        Variable.string("commodity_type", required=True),
        Variable.number("price_value"),
    ]
)

delm = DELM(
    model="openai/gpt-4o-mini",
    schema=schema,
    keywords=["price", "forecast"]
)

result = delm.extract(df)
```

**Pros**: Type safety, IDE autocomplete, composable
**Cons**: More verbose, YAML is industry standard for config

**Recommendation**: Keep YAML but add Python config option for power users.

