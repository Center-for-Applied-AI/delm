<a href="#"><img src="delm_logo.png" align="left" width="160" style="margin-right: 15px;" alt="DELM logo"/></a>

# Data Extraction with Language Models

<br clear="left"/>

DELM is a Python toolkit for extracting structured data from unstructured text using language models.

📖 **[Full Documentation](https://center-for-applied-ai.github.io/delm/)**

## Features

- **Multiple input formats**: TXT, HTML, MD, DOCX, PDF, CSV, Excel, Parquet, Feather
- **Flexible schemas**: Simple key-value → nested objects → multiple schemas
- **Multiple LLM providers**: OpenAI, Anthropic, Google, Groq, Together AI, Fireworks AI
- **Cost management**: Automatic cost tracking, caching, and budget limits
- **Built for scale**: Batch processing with parallel execution and checkpointing

## Installation

```bash
pip install delm
```

## Quick Start

Define your extraction schema and extract structured data in just a few lines:

```python
from delm import DELM, Schema, ExtractionVariable

# Define what to extract
schema = Schema.simple(
    variables_list=[
        ExtractionVariable(
            name="company",
            description="Company name mentioned",
            data_type="string",
            required=True,
        ),
        ExtractionVariable(
            name="price",
            description="Price value if mentioned",
            data_type="number",
            required=False,
        ),
    ]
)

# Initialize and extract
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
)

# Extract from any supported file format
results = delm.extract("data/earnings_calls.txt")
print(results)

# Check costs
print(delm.get_cost_summary())
```

## Schema Types

DELM supports three schema types for different extraction needs:

### Simple Schema
Extract key-value pairs from text:

```python
schema = Schema.simple(
    variables_list=[
        ExtractionVariable(name="author", data_type="string"),
        ExtractionVariable(name="date", data_type="date"),
    ]
)
```

### Nested Schema
Extract lists of structured objects:

```python
schema = Schema.nested(
    container_name="products",
    variables_list=[
        ExtractionVariable(name="name", data_type="string"),
        ExtractionVariable(name="price", data_type="number"),
        ExtractionVariable(name="features", data_type="[string]"),
    ]
)
```

### Multiple Schemas
Extract multiple different schemas simultaneously:

```python
schema = Schema.multiple({
    "companies": Schema.nested(
        container_name="companies",
        variables_list=[...],
    ),
    "products": Schema.nested(
        container_name="products",
        variables_list=[...],
    ),
})
```

## Supported Data Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"Apple Inc."` |
| `number` | Floating-point | `150.5` |
| `integer` | Whole numbers | `2024` |
| `boolean` | True/False | `true` |
| `date` | Date strings | `"2025-09-15"` |
| `[string]` | List of strings | `["oil", "gas"]` |
| `[number]` | List of numbers | `[100, 200]` |

## Advanced Features

### Custom Prompts

```python
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    prompt_template="""You are a financial data extraction expert.

Extract the following information:
{variables}

Text to analyze:
{text}""",
)
```

### Process CSV/Structured Data

```python
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    target_column="transcript_text",  # Column containing text to process
)

results = delm.extract("earnings_data.csv")
```

### Cost Tracking & Limits

```python
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    track_cost=True,
    max_budget=10.0,  # Stop if cost exceeds $10
)

results = delm.extract("data.txt")
summary = delm.get_cost_summary()
print(f"Total cost: ${summary['total_cost']:.2f}")
```

### Batch Processing

```python
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    batch_size=50,      # Process 50 records per batch
    max_workers=5,      # Use 5 parallel workers
)

results = delm.extract("large_dataset.csv")
```

## Configuration Options

For a complete list of configuration options, see the [documentation](https://center-for-applied-ai.github.io/delm/).

**Common parameters:**
- `provider`: LLM provider (`"openai"`, `"anthropic"`, `"google"`, etc.)
- `model`: Model name (`"gpt-4o-mini"`, `"claude-3-sonnet-20240229"`, etc.)
- `temperature`: Generation temperature (default: `0.0`)
- `batch_size`: Records per batch (default: `10`)
- `max_workers`: Concurrent workers (default: `1`)
- `track_cost`: Enable cost tracking (default: `True`)
- `max_budget`: Maximum cost limit in dollars (default: `None`)
- `target_column`: Column name for CSV/tabular data (default: `None`)

## Documentation

📖 **[Full Documentation](https://center-for-applied-ai.github.io/delm/)**

Learn more about:
- [Getting Started Guide](https://center-for-applied-ai.github.io/delm/getting-started/)
- [Schema Design](https://center-for-applied-ai.github.io/delm/user-guide/schemas/)
- [Text Processing & Filtering](https://center-for-applied-ai.github.io/delm/user-guide/text-preprocessing/)
- [Cost Management](https://center-for-applied-ai.github.io/delm/user-guide/cost-management/)
- [API Reference](https://center-for-applied-ai.github.io/delm/reference/)

## File Format Support

| Format | Extensions | Additional Dependencies |
|--------|-----------|------------------------|
| Text | `.txt` | None |
| HTML/Markdown | `.html`, `.htm`, `.md` | `beautifulsoup4` |
| Word | `.docx` | `python-docx` |
| PDF | `.pdf` | `marker-pdf` |
| CSV | `.csv` | `pandas` |
| Excel | `.xlsx` | `openpyxl` |
| Parquet | `.parquet` | `pyarrow` |
| Feather | `.feather` | `pyarrow` |

## Contributing

We welcome contributions! Please see our [documentation](https://center-for-applied-ai.github.io/delm/) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Built on [Instructor](https://python.useinstructor.com/) for structured outputs
- Uses [Marker](https://pypi.org/project/marker-pdf/) for PDF processing
- Developed at the [Center for Applied AI](https://www.chicagobooth.edu/research/center-for-applied-artificial-intelligence) at Chicago Booth
