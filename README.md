# DELM (Data Extraction with Language Models)

A comprehensive Python toolkit for extracting structured data from unstructured text using language models. DELM provides a configurable, scalable pipeline with built-in cost tracking, caching, and evaluation capabilities.

## Features

- **Multi-format Support**: TXT, HTML, MD, DOCX, PDF, CSV, Excel, Parquet, Feather
- **Progressive Schema System**: Simple → Nested → Multiple schemas for any complexity
- **Multi-Provider Support**: OpenAI, Anthropic, Google, Groq, Together AI, Fireworks AI
- **Smart Processing**: Configurable text splitting, relevance scoring, and filtering
- **Cost Optimization**: Built-in cost tracking, caching, and budget management
- **Batch Processing**: Parallel execution with checkpointing and resume capabilities
- **Comprehensive Evaluation**: Performance metrics and cost analysis tools

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/delm.git
cd delm

# Install from source
pip install -e .
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from delm import DELM, DELMConfig

# Load configuration from YAML
config = DELMConfig.from_yaml("example.config.yaml")

# Initialize DELM
delm = DELM(
    config=config,
    experiment_name="my_experiment",
    experiment_directory=Path("experiments"),
)

# Process data
df = delm.prep_data("data/input.txt")
results = delm.process_via_llm()

# Get results
final_df = delm.get_extraction_results_df()
cost_summary = delm.get_cost_summary()
```

### Configuration Files

DELM uses two configuration files:

**1. Pipeline Configuration (`config.yaml`)**
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

**2. Schema Specification (`schema_spec.yaml`)**
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

## Schema Types

DELM supports three levels of schema complexity:

### Simple Schema (Level 1)
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

### Nested Schema (Level 2)
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

### Multiple Schema (Level 3)
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

## Use Cases

### Financial Data Extraction
- Earnings call transcript analysis
- Commodity price forecasting
- Financial report parsing
- Market sentiment analysis

### Research & Analysis
- Academic paper analysis
- Survey response processing
- Interview transcript coding
- Literature review automation

### Business Intelligence
- Customer feedback analysis
- Product review extraction
- Competitor analysis
- Market research automation

## Supported Data Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"Apple Inc."` |
| `number` | Floating-point numbers | `150.5` |
| `integer` | Whole numbers | `2024` |
| `boolean` | True/False values | `true` |
| `[string]` | List of strings | `["oil", "gas"]` |
| `[number]` | List of numbers | `[100, 200, 300]` |
| `[integer]` | List of integers | `[1, 2, 3, 4]` |
| `[boolean]` | List of booleans | `[true, false, true]` |


## Advanced Features

### Cost Summary
```python
# Get cost summary after extraction
cost_summary = delm.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost_usd']}")
```

### Semantic Caching
Reuses api responses from identical calls. Ensures no wasted api credits for certain experiment re-runs.
```yaml
semantic_cache:
  backend: "sqlite"        # sqlite, lmdb, filesystem
  path: ".delm_cache"
  max_size_mb: 512
```

### Relevance Filtering
```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance"]
  pandas_score_filter: "delm_score >= 0.7"
```

### Text Splitting Strategies
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

## Performance & Evaluation

### Cost Estimation
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

### Performance Evaluation
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

## Examples

### Commodity Price Extraction
```python
# Extract commodity prices from earnings calls
config = DELMConfig.from_yaml("examples/commodity_schema.yaml")
delm = DELM(config=config, experiment_name="commodity_extraction")

# Process earnings call transcripts
df = delm.prep_data("earnings_calls.csv")
extraction_results = delm.process_via_llm()
```

## Configuration Reference

### Required Fields
- `llm_extraction.provider`: LLM provider (openai, anthropic, google, etc.)
- `llm_extraction.name`: Model name (gpt-4o-mini, claude-3-sonnet, etc.)
- `schema.spec_path`: Path to schema specification file

### Optional Fields with Defaults
- `llm_extraction.temperature`: 0.0 (deterministic)
- `llm_extraction.batch_size`: 10 (records per batch)
- `llm_extraction.max_workers`: 1 (concurrent workers)
- `llm_extraction.track_cost`: true (cost tracking)
- `semantic_cache.backend`: "sqlite" (cache backend)

## Architecture

### Core Components
1. **DataProcessor**: Handles loading, splitting, and scoring
2. **SchemaManager**: Manages schema loading and validation
3. **ExtractionManager**: Orchestrates LLM extraction
4. **ExperimentManager**: Handles experiment state and checkpointing
5. **CostTracker**: Monitors API costs and budgets

### Strategy Classes
- **SplitStrategy**: Text chunking (Paragraph, FixedWindow, Regex)
- **RelevanceScorer**: Content scoring (Keyword, Fuzzy)
- **SchemaRegistry**: Schema type management

### Estimation Functions
- **estimate_input_token_cost**: Estimate input token costs without API calls
- **estimate_total_cost**: Estimate total costs using API calls on a sample to
- **estimate_performance**: Evaluate extraction performance against human-labeled data

## File Format Support

| Format | Extension | Requirements |
|--------|-----------|--------------|
| Text | `.txt` | Built-in |
| HTML/Markdown | `.html`, `.md` | `beautifulsoup4` |
| Word Documents | `.docx` | `python-docx` |
| PDF | `.pdf` | `marker` (OCR) |
| CSV | `.csv` | `pandas` |
| Excel | `.xlsx` | `openpyxl` |
| Parquet | `.parquet` | `pyarrow` |
| Feather | `.feather` | `pyarrow` |

## Documentation

- [Schema Reference](SCHEMA_REFERENCE.md) - Detailed schema configuration guide
- [Configuration Examples](example.config.yaml) - Complete configuration templates
- [Schema Examples](example.schema_spec.yaml) - Schema specification templates

## Acknowledgments

- Built on [Instructor](https://python.useinstructor.com/) for structured outputs
- Uses [Marker](https://pypi.org/project/marker-pdf/) for PDF processing
- Developed at the Center for Applied AI at Chicago Booth

---

**DELM v0.2.0** - Making data extraction with LLMs accessible, reliable, and cost-effective. 