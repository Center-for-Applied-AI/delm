# DELM (Data Labeling with Language Models)

A comprehensive toolkit for extracting structured data from unstructured text and images using language models. DELM provides a configurable, scalable pipeline for data extraction with built-in evaluation capabilities and cost tracking.

## 🚀 Features

### Core Pipeline
- **Multi-format Support**: TXT, HTML, MD, DOCX, PDF, CSV, JSON, images
- **Pluggable Strategies**: Customizable text splitting and relevance scoring
- **Unified Schema System**: Progressive complexity from simple to nested to multiple schemas
- **Structured Extraction**: Instructor + Pydantic schemas with fallback mechanisms
- **Batch Processing**: Parallel execution for efficient large-scale processing

### Advanced Extraction
- **Progressive Complexity**: Start simple, scale to complex nested structures
- **Schema Registry**: Extensible system for custom schema types
- **Context-Aware Processing**: Metadata integration and contextual prompts
- **Validation & Error Handling**: Robust error recovery and validation

### Evaluation & Monitoring
- **Comprehensive Metrics**: Accuracy, precision, recall, cost analysis
- **Real-time Monitoring**: Progress tracking and performance insights
- **Configurable Thresholds**: Customizable relevance and confidence filters

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/delm.git
cd delm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp example.env .env
# Edit .env with your API keys
```

## 🔧 Quick Start

### Basic Usage

```python
from DELM import DELM, ParagraphSplit, KeywordScorer

# Initialize DELM with schema specification
delm = DELM(
    schema_spec_path="example.schema_spec.yaml",
    dotenv_path=".env",
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_retries=3,
    batch_size=10,
    max_workers=4,
    split_strategy=ParagraphSplit(),
    relevance_scorer=KeywordScorer(["price", "forecast", "estimate"]),
    regex_fallback_pattern=r'\d+'  # Extract numbers as fallback
)

# Load and process data
df = delm.prep_data("data/input/report.txt")
processed_df = delm.process_via_llm(df, verbose=True)

# Get results
results_df = delm.parse_to_dataframe(processed_df)
metrics = delm.evaluate_output_metrics(processed_df)
```

### Commodity Price Extraction Example

```python
# Use the commodity extraction schema
delm = DELM(
    schema_spec_path="commodity_extraction_schema.yaml",
    dotenv_path=".env",
    model_name="gpt-4o-mini",
    temperature=0.0
)

# Process earnings call transcripts
df = delm.prep_data(report_df, target_column="text")
relevant_chunks = df[df["score"] > 0]
results = delm.process_via_llm(relevant_chunks, use_batching=True)
commodity_data = delm.parse_to_dataframe(results)
```

## 📋 Configuration

DELM separates configuration into two parts:
1. **Schema Specification**: YAML file defining the extraction schema
2. **Runtime Parameters**: Constructor arguments for model and processing settings

### Schema Specification

The schema specification file (e.g., `example.schema_spec.yaml`) defines what data to extract:

```yaml
# Schema type: simple, nested, or multiple
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

prompt_template: |
  Extract commodity information from the text:
  {variables}
  
  Text: {text}
```

### Runtime Configuration

Model and processing parameters are passed to the constructor:

```python
delm = DELM(
    schema_spec_path="example.schema_spec.yaml",
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_retries=3,
    batch_size=10,
    max_workers=4,
    regex_fallback_pattern=r'\d+'  # Optional: custom regex for fallback extraction
)
```

#### Regex Fallback Configuration

You can provide a custom regex pattern for fallback extraction when LLM processing fails:

```python
# Extract numbers when LLM fails
delm = DELM(regex_fallback_pattern=r'\d+')

# Extract email addresses
delm = DELM(regex_fallback_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Extract currency amounts
delm = DELM(regex_fallback_pattern=r'\$\d+(?:,\d{3})*(?:\.\d{2})?')

# No regex fallback (default)
delm = DELM()  # regex_fallback_pattern=None
```

### Schema Types

DELM supports three levels of schema complexity:

- **Simple Schema (Level 1)**: Basic key-value extraction
- **Nested Schema (Level 2)**: Structured objects with multiple fields  
- **Multiple Schemas (Level 3)**: Multiple independent structured objects

For detailed examples and configuration options, see [SCHEMA_REFERENCE.md](SCHEMA_REFERENCE.md).

## 🏗️ Architecture

### Pipeline Components

1. **Data Loading**: Multi-format file loaders with OCR support
2. **Text Processing**: Configurable chunking and relevance scoring
3. **LLM Extraction**: Structured extraction with Instructor
4. **Response Parsing**: Validation and DataFrame conversion
5. **Evaluation**: Metrics calculation and cost tracking

### Strategy Classes

- **SplitStrategy**: Text chunking strategies (Paragraph, FixedWindow, Regex)
- **RelevanceScorer**: Content relevance scoring (Keyword, Fuzzy)
- **SchemaRegistry**: Unified schema system with progressive complexity
- **BaseSchema**: Abstract interface for all schema types
- **SimpleSchema**: Basic key-value extraction (Level 1)
- **NestedSchema**: Complex nested structures (Level 2)
- **MultipleSchema**: Multiple independent schemas (Level 3)

## 📊 Supported File Formats

| Format | Extension | Requirements |
|--------|-----------|--------------|
| Text | `.txt` | Built-in |
| HTML/Markdown | `.html`, `.md` | `beautifulsoup4` |
| Word Documents | `.docx` | `python-docx` |
| PDF | `.pdf` | `marker` |
| CSV | `.csv` | `pandas` |
| Excel | `.xlsx` | `openpyxl` |
| Parquet | `.parquet` | `pyarrow` |
| Images | `.png`, `.jpg` | OCR support |

## 🔍 Use Cases

### Financial Data Extraction
- Earnings call transcript analysis
- Commodity price forecasting
- Financial report parsing
- Market sentiment analysis

### Research Data Collection
- Academic paper analysis
- Survey response processing
- Interview transcript coding
- Literature review automation

### Business Intelligence
- Customer feedback analysis
- Product review extraction
- Competitor analysis
- Market research automation

## 📈 Performance & Cost

### Cost Optimization
- **Batch Processing**: Reduce API calls with parallel execution
- **Relevance Filtering**: Only process relevant content
- **Smart Chunking**: Optimize chunk sizes for accuracy vs cost
- **Model Selection**: Choose cost-effective models for your use case

### Performance Monitoring
```python
# Get cost summary
cost_summary = delm.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost_usd']}")
print(f"Average cost per request: ${cost_summary['avg_cost_per_request']}")

# Get performance metrics
metrics = delm.evaluate_output_metrics(processed_df)
print(f"Success rate: {metrics['valid_json']:.2%}")
```

## 🧪 Testing

Run the test suite:

```bash
# Test basic functionality
python tests/test_open_ai_key_and_instructor.py

# Test earnings report extraction
python tests/earning_report_delm_testing.py

# Test commodity extraction
python tests/commodity_extraction_example.py
```

## 🔧 Customization

### Adding New Split Strategies

```python
class CustomSplitStrategy(SplitStrategy):
    def split(self, text: str) -> List[str]:
        # Your custom splitting logic
        return chunks
```

### Adding New Relevance Scorers

```python
class CustomScorer(RelevanceScorer):
    def score(self, paragraph: str) -> float:
        # Your custom scoring logic
        return score
```

### Custom Extraction Schemas

```python
# Define your schema in YAML
extraction:
  variables:
    - name: "custom_field"
      description: "Your custom field"
      data_type: "string"
      allowed_values: ["option1", "option2"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of [Instructor](https://python.useinstructor.com/) for structured outputs
- Uses [Marker](https://pypi.org/project/marker-pdf/) for PDF processing
- Inspired by research practices at the Center for Applied AI

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review example configurations

---

**DELM v0.2** - Making data extraction with LLMs accessible, reliable, and cost-effective. 