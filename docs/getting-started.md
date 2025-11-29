# Getting Started

Install DELM and run your first extraction pipeline in minutes.

## Installation

Install from PyPI:

```bash
pip install delm
```

Or with optional dependencies (pdf, excel, alternative caching, etc)
```
pip install delm[extras]
```

## Environment Variables

DELM requires API keys for the LLM providers you use. You must set these environment variables before using DELM.

For a complete list of supported providers and their required environment variable names, see the [Instructor documentation](https://python.useinstructor.com/hub/).

**Quick Example**: For OpenAI, you would set:

```bash
export OPENAI_API_KEY="sk-..."
```

<<<<<<< HEAD
**Optional**: If you prefer using `.env` files with `python-dotenv`:
=======
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
>>>>>>> origin/main

```python
from dotenv import load_dotenv
load_dotenv()
```

## Define Your Schema

Import the necessary classes and define what you want to extract:

```python
from delm import DELM, Schema, ExtractionVariable

# Define extraction schema
schema = Schema.nested(
    container_name="commodities",
    ExtractionVariable(
        name="commodity_type",
        description="Type of commodity mentioned",
        data_type="string",
        required=True,
    ),
    ExtractionVariable(
        name="price_value",
        description="Price value mentioned",
        data_type="number",
        required=False,
    ),
)
```

## Run Extraction

Create a DELM pipeline and extract structured data from your text:

```python
import pandas as pd

# Initialize pipeline
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
)

# Prepare input data
data = pd.DataFrame({
    "text": [
        "Oil prices rose to $75 per barrel while gold fell to $1,850 per ounce.",
    ]
})

# Run extraction
results = delm.extract(data)
print(results)
```

## Understanding Results

The `results` DataFrame will contain your original data plus extracted information. For the example above, DELM would extract:

**Input text**: "Oil prices rose to $75 per barrel while gold fell to $1,850 per ounce."

**Extracted data**:

```json
{
  "commodities": [
    {
      "commodity_type": "oil",
      "price_value": 75.0
    },
    {
      "commodity_type": "gold",
      "price_value": 1850.0
    }
  ]
}
```

The results DataFrame includes all your original columns plus extraction results:

| text | delm_record_id | delm_chunk_id | delm_extracted_data_json |
|------|--------------|--------------|-------------------------|
| Oil prices rose to $75 per barrel... | 0 | 0 | {"commodities": [{"commodity_type": "oil", "price_value": 75.0}, ...]} |
| ... | ... | ... | ... |