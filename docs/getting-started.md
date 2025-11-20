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

**Optional**: If you prefer using `.env` files with `python-dotenv`:

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

Key columns:
  - **All original metadata columns**: Your input data is preserved
  - **`delm_record_id`**: Unique identifier for each record prior to any splitting
  - **`delm_chunk_id`**: Unique identifier for each processed text chunk
  - **`delm_extracted_data_json`**: Raw JSON extraction results