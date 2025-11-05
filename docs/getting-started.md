# Getting Started

Install DELM, create your first configuration files, and run your first extraction pipeline.

## Installation

Install from PyPI:

```bash
pip install delm
```

Or install from source:

```bash
git clone https://github.com/Center-for-Applied-AI/delm.git
cd delm
pip install -e .
```

If you use the optional developer tooling (tests, linters, notebooks), install the `dev` extra:

```bash
pip install -e .[dev]
```

## Configure Environment Variables

DELM requires API keys for the LLM providers you use. You are responsible for loading these environment variables in whatever way works best for your workflow.

### Required Environment Variables by Provider

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_API_KEY`
- **Groq**: `GROQ_API_KEY`
- **Together AI**: `TOGETHER_API_KEY`
- **Fireworks AI**: `FIREWORKS_API_KEY`

### Option 1: Export in Your Shell

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
```

### Option 2: Use python-dotenv (Optional)

If you prefer using `.env` files, install and use `python-dotenv`:

```bash
pip install python-dotenv
```

Then in your script:

```python
from dotenv import load_dotenv
load_dotenv()  # Load from .env file in current directory
```

**Note**: You only need to set the API key for the provider you're using. DELM accesses environment variables directly via the LLM client libraries (OpenAI, Anthropic, etc.).

## Create Your Pipeline Configuration

Create a file called `config.yaml` in your project directory:

```yaml
llm_extraction:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.0
  batch_size: 10

schema:
  spec_path: "schema_spec.yaml"
```

This minimal configuration:
- Uses OpenAI's GPT-4o-mini model
- Sets temperature to 0.0 for deterministic results
- Processes 10 records per batch
- Points to your schema specification file

## Create Your Schema Specification

Create a file called `schema_spec.yaml` in your project directory:

```yaml
schema_type: "nested"
container_name: "commodities"
variables:
  - name: "commodity_type"
    description: "Type of commodity mentioned"
    data_type: "string"
    required: true
  - name: "price_value"
    description: "Price value mentioned"
    data_type: "number"
    required: false
```

This schema:
- Extracts a list of commodity objects from each text chunk
- Each object has a required commodity type and optional price value
- Uses a nested schema structure for multiple items per chunk

## Run Your First Extraction

Now you can run your first extraction:

```python
from pathlib import Path
from delm import DELM

# 1. Create pipeline from config
pipeline = DELM.from_yaml(
    config_path="config.yaml",
    experiment_name="my_first_extraction",
    experiment_directory=Path("experiments"),
)

# 2. Prepare your data
pipeline.prep_data("data/input.txt")

# 3. Run extraction
pipeline.process_via_llm()

# 4. Get results
results = pipeline.get_extraction_results()
cost_summary = pipeline.get_cost_summary()
```

### Project Layout

A typical project structure keeps inputs, configuration, and outputs separated:

```
project/
├── data/
│   └── input.txt
├── config.yaml
├── schema_spec.yaml
└── experiments/
    └── my_first_extraction/
        ├── delm_data/
        ├── delm_logs/
        └── cost_summary.json
```

- **Pipeline configuration** (`config.yaml`) controls providers, preprocessing, and batching
- **Schema specification** (`schema_spec.yaml`) declares the fields you want to extract
- **Experiments directory** stores run artifacts, logs, and summaries

## Understanding Your Results

After running extraction, you'll get:

### Extraction Results
```python
results = pipeline.get_extraction_results()
print(results.head())
```

The results DataFrame contains:
- `delm_raw_data`: Original text chunks
- `delm_extracted_data_json`: Extracted JSON for each chunk
- `delm_chunk_id`: Unique identifier for each chunk

### Cost Summary
```python
cost_summary = pipeline.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost']:.4f}")
print(f"Input tokens: {cost_summary['total_input_tokens']:,}")
print(f"Output tokens: {cost_summary['total_output_tokens']:,}")
```

### Example Output

For a text chunk about "Oil prices rose to $75 per barrel", your schema would extract:

```json
{
  "commodities": [
    {
      "commodity_type": "oil",
      "price_value": 75
    }
  ]
}
```

## Next Steps

Now that you've run your first extraction, explore these advanced workflows:

### Cost Estimation
Before running large extractions, estimate costs to stay within budget:
- [Cost Estimation Tutorial](tutorials/cost-estimation.md) - Learn to estimate costs before running full extractions

### Performance Evaluation  
Evaluate extraction quality against human-labeled data:
- [Performance Evaluation Tutorial](tutorials/performance-evaluation.md) - Learn to measure precision, recall, and F1 scores

### Advanced Configuration
Customize your pipeline for production use:
- [Pipeline Configuration](configuration/pipeline-config.md) - Complete reference for all configuration options
- [Schema Design](configuration/schema-design.md) - Advanced schema patterns and validation features

### Built-in Features
Explore DELM's production-ready features:
- [Caching](features/caching.md) - Reduce costs with semantic caching
- [Text Processing](features/text-processing.md) - Advanced splitting and scoring strategies
- [Batch Processing](features/batch-processing.md) - Optimize performance with batching and checkpointing
- [Cost Tracking](features/cost-tracking.md) - Monitor spending and set budget limits
- [Post-Processing](features/post-processing.md) - Transform results into tabular format
- [File Formats](features/file-formats.md) - Supported input formats and requirements