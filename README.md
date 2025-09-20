<a href="#"><img src="delm_logo.png" align="left" width="160" style="margin-right: 15px;" alt="DELM logo"/></a>

# Data Extraction with Language Models

<br clear="left"/>

DELM is a Python toolkit for extracting structured data from unstructured text using language models. It provides a configurable pipeline with cost tracking, caching, and evaluation capabilities.

## Features

- Supported input formats: TXT, HTML, MD, DOCX, PDF, CSV, Excel, Parquet, Feather
- Progressive schema system: simple → nested → multiple
- Multiple model providers: OpenAI, Anthropic, Google, Groq, Together AI, Fireworks AI
- Configurable processing: text splitting, relevance scoring, filtering
- Cost management: cost tracking, caching, budget limits
- Batch processing: parallel execution with checkpointing and resume
- Evaluation tools: performance metrics and cost analysis

## Installation

```bash
pip install delm
```

Or if you would like to install from source:

```bash
# Clone the repository
git clone https://github.com/Center-for-Applied-AI/delm.git
cd delm

# Install from source
pip install -e .
```

## Quick start

### Basic usage

```python
from pathlib import Path
from delm import DELM

# Initialize DELM from a pipeline config YAML
delm = DELM.from_yaml(
    config_path="example.config.yaml",
    experiment_name="my_experiment",
    experiment_directory=Path("experiments"),
)

# Process data
df = delm.prep_data("data/input.txt")
results = delm.process_via_llm()

# Get results
final_df = delm.get_extraction_results()
cost_summary = delm.get_cost_summary()
```

### Configuration files

DELM uses two configuration files:

1. Pipeline configuration (`config.yaml`)
```yaml
llm_extraction:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.0
  batch_size: 10
  track_cost: true
  max_budget: 50.0

data_preprocessing:
  target_column: "text"
  splitting:
    type: "ParagraphSplit"
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance"]

schema:
  spec_path: "schema_spec.yaml"
```

2. Schema specification (`schema_spec.yaml`)
```yaml
schema_type: "nested"
container_name: "commodities"

variables:
  - name: "commodity_type"
    description: "Type of commodity mentioned"
    data_type: "string"
    required: true
    allowed_values: ["oil", "gas", "copper", "gold"]
  
  - name: "price_value"
    description: "Price mentioned in text"
    data_type: "number"
    required: false
```

Validation notes:
- `validate_in_text: true` applies to string fields only. Values must literally appear (case‑insensitive) in the source text to be kept.

## Schema types

DELM supports three levels of schema complexity:

### Simple schema (level 1)
Extract key-value pairs from each text chunk:
```yaml
schema_type: "simple"
variables:
  - name: "price"
    description: "Price mentioned"
    data_type: "number"
  - name: "company"
    description: "Company name"
    data_type: "string"
```

### Nested schema (level 2)
Extract structured objects with multiple fields:
```yaml
schema_type: "nested"
container_name: "commodities"
variables:
  - name: "type"
    description: "Commodity type"
    data_type: "string"
  - name: "price"
    description: "Price value"
    data_type: "number"
```

### Multiple schemas (level 3)
Extract multiple independent schemas simultaneously:
```yaml
schema_type: "multiple"
commodities:
  schema_type: "nested"
  container_name: "commodities"
  variables: [...]
companies:
  schema_type: "nested"
  container_name: "companies"
  variables: [...]
```

Note: Multiple schema outputs are unwrapped. For a nested sub‑schema named `books` with container `books`, the output key is `books: [...]` (not `books: {books: [...]}`).

## Supported data types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"Apple Inc."` |
| `number` | Floating-point numbers | `150.5` |
| `integer` | Whole numbers | `2024` |
| `boolean` | True/False values | `true` |
| `date` | Date strings | `"2025-09-15"` |
| `[string]` | List of strings | `["oil", "gas"]` |
| `[number]` | List of numbers | `[100, 200, 300]` |
| `[integer]` | List of integers | `[1, 2, 3, 4]` |
| `[boolean]` | List of booleans | `[true, false, true]` |


## Advanced features

### Cost summary
```python
# Get cost summary after extraction
cost_summary = delm.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost']}")
```

### Semantic caching
Caches API responses for identical calls to reduce repeated cost in re‑runs.
```yaml
semantic_cache:
  backend: "sqlite"        # sqlite, lmdb, filesystem
  path: ".delm_cache"
  max_size_mb: 512
  synchronous: "normal"    # sqlite only: "normal" or "full"
```

### Relevance filtering
```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance"]
  pandas_score_filter: "delm_score >= 0.7"
```
If a scorer is configured but no `pandas_score_filter` is provided, all chunks are kept (a warning is logged).

You can also use a fuzzy keyword scorer. This requires the optional dependency `rapidfuzz`.
```yaml
data_preprocessing:
  scoring:
    type: "FuzzyScorer"
    keywords:
      - price
      - forecast
      - guidance
  pandas_score_filter: "delm_score >= 0.5"  # example threshold for fuzzy scores
```

### Text splitting strategies
```yaml
data_preprocessing:
  splitting:
    type: "ParagraphSplit"      # Split by paragraphs
    # type: "FixedWindowSplit"  # Split by sentence count
    # window: 5
    # stride: 2
    # type: "RegexSplit"        # Custom regex pattern
    # pattern: "\n\n"
```

### Post-processing
After extraction, explode the JSON column into tabular rows according to your schema.
```python
from delm.utils.post_processing import explode_json_results

# Use the same schema used for extraction (path to the YAML/JSON or a schema object)
exploded_df = explode_json_results(
    final_df,
    schema=delm.config.schema.spec_path  # or a loaded schema object
)
```

## Performance and evaluation

### Cost estimation
> [!WARNING]
> Cost estimation is provided as‑is. Estimates are not guarantees and may be inaccurate. The authors and maintainers accept no liability for any losses, charges, or damages resulting from use of this feature. Use at your own risk.

Estimate total cost of your current configuration setup before running the full extraction.
```python
from delm.utils.cost_estimation import estimate_input_token_cost, estimate_total_cost

# Estimate input token costs without API calls
input_cost = estimate_input_token_cost(
    config="config.yaml",
    data_source="data.csv"
)
print(f"Input token cost: ${input_cost:.2f}")

# Estimate total costs using API calls on a sample
total_cost = estimate_total_cost(
    config="config.yaml",
    data_source="data.csv",
    sample_size=100
)
print(f"Estimated total cost: ${total_cost:.2f}")
```

### Performance evaluation
Estimate the performance of your current configuration before running the full extraction.
```python
from delm.utils.performance_estimation import estimate_performance

# Evaluate against human-labeled data
metrics, expected_and_extracted_df = estimate_performance(
    config="config.yaml",
    data_source="test_data.csv",
    expected_extraction_output_df=human_labeled_df,
    true_json_column="expected_json",
    matching_id_column="id",
    record_sample_size=50  # Optional: limit sample size
)

# Display performance metrics
for key, value in metrics.items():
    precision = value.get("precision", 0)
    recall = value.get("recall", 0)
    f1 = value.get("f1", 0)
    print(f"{key:<30} Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
```

## Configuration reference

### Required fields
- `llm_extraction.provider`: LLM provider (openai, anthropic, google, etc.)
- `llm_extraction.name`: Model name (gpt-4o-mini, claude-3-sonnet, etc.)
- `schema.spec_path`: Path to schema specification file

### Optional fields with defaults
- `llm_extraction.temperature`: 0.0 (deterministic)
- `llm_extraction.batch_size`: 10 (records per batch)
- `llm_extraction.max_workers`: 1 (concurrent workers)
- `llm_extraction.track_cost`: true (cost tracking)
- `semantic_cache.backend`: "sqlite" (cache backend)

### Additional LLM fields
- `llm_extraction.max_retries`: 3 (retry attempts)
- `llm_extraction.base_delay`: 1.0 (seconds, exponential backoff base)
- `llm_extraction.dotenv_path`: null (path to “.env” for credentials)
- `llm_extraction.model_input_cost_per_1M_tokens`: null (override pricing)
- `llm_extraction.model_output_cost_per_1M_tokens`: null (override pricing)

If using providers not present in the built-in pricing DB, set both `model_input_cost_per_1M_tokens` and `model_output_cost_per_1M_tokens`, or set `track_cost: false`.

### Data preprocessing fields
- `data_preprocessing.drop_target_column`: false
- `data_preprocessing.pandas_score_filter`: null (e.g., "delm_score >= 0.7")
- `data_preprocessing.preprocessed_data_path`: null (path to “.feather” with `delm_text_chunk` and `delm_chunk_id`; when set, omit splitting/scoring/filter fields)

### Semantic cache fields
- `semantic_cache.backend`: "sqlite" | "lmdb" | "filesystem"
- `semantic_cache.path`: ".delm_cache"
- `semantic_cache.max_size_mb`: 512
- `semantic_cache.synchronous`: "normal" | "full" (sqlite only)

## Experiment storage and logging

- Disk storage (default): checkpointing, resume, and results persisted under `delm_experiments/<experiment_name>/`.
- In-memory storage: `use_disk_storage=False` for fast prototyping (no persistence, no resume).
- Logging: by default, rotating file logs under `delm_logs/<experiment_name>/` when `save_file_log=True`.
  - Tunables: `save_file_log`, `log_dir`, `console_log_level`, `file_log_level`, `override_logging`.
  - Or call `delm.logging.configure(...)` directly.

## Architecture

### Core components
1. DataProcessor: Handles loading, splitting, and scoring
2. SchemaManager: Manages schema loading and validation
3. ExtractionManager: Orchestrates LLM extraction
4. ExperimentManager: Handles experiment state and checkpointing
5. CostTracker: Monitors API costs and budgets

### Strategy classes
- SplitStrategy: Text chunking (Paragraph, FixedWindow, Regex)
- RelevanceScorer: Content scoring (Keyword, Fuzzy)
- SchemaRegistry: Schema type management

### Estimation functions
- estimate_input_token_cost: Estimate input token costs without API calls
- estimate_total_cost: Estimate total costs using API calls on a sample
- estimate_performance: Evaluate extraction performance against human-labeled data

## File format support

| Format | Extension | Requirements |
|--------|-----------|--------------|
| Text | `.txt` | Built-in |
| HTML/Markdown | `.html`, `.htm`, `.md` | `beautifulsoup4` |
| Word Documents | `.docx` | `python-docx` |
| PDF | `.pdf` | `marker-pdf` (OCR) |
| CSV | `.csv` | `pandas` |
| Excel | `.xlsx` | `openpyxl` |
| Parquet | `.parquet` | `pyarrow` |
| Feather | `.feather` | `pyarrow` |

## Documentation

### Local MkDocs site
1. Install the documentation dependencies: `pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python`
2. Serve the docs locally to `http://127.0.0.1:8000/`: `mkdocs serve`
3. Use `mkdocs build` to generate a static site in the `site/` directory.

### Reference materials
- [Schema Reference](SCHEMA_REFERENCE.md) - Detailed schema configuration guide
- [Configuration Examples](example.config.yaml) - Complete configuration templates
- [Schema Examples](example.schema_spec.yaml) - Schema specification templates

## Acknowledgments

- Built on [Instructor](https://python.useinstructor.com/) for structured outputs
- Uses [Marker](https://pypi.org/project/marker-pdf/) for PDF processing
- Developed at the Center for Applied AI at Chicago Booth
