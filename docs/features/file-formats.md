# File Formats

Learn about the file formats supported by DELM and their specific requirements.

## Supported Formats

DELM supports a wide range of input formats for text extraction:

| Format | Extension | Requirements | Best For |
|--------|-----------|--------------|----------|
| Text | `.txt` | Built-in | Plain text documents |
| HTML | `.html`, `.htm` | `beautifulsoup4` | Web pages, HTML documents |
| Markdown | `.md` | `beautifulsoup4` | Documentation, README files |
| Word Documents | `.docx` | `python-docx` | Microsoft Word documents |
| PDF | `.pdf` | `marker-pdf` | Scanned documents, reports |
| CSV | `.csv` | `pandas` | Structured data with text columns |
| Excel | `.xlsx`, `.xls` | `openpyxl` | Spreadsheets with text data |
| Parquet | `.parquet` | `pyarrow` | High-performance columnar data |
| Feather | `.feather` | `pyarrow` | Fast serialization format |

## Installation Requirements

### Core Dependencies

```bash
# Install DELM with all format support
pip install delm[all]

# Or install specific format dependencies
pip install beautifulsoup4 python-docx marker-pdf openpyxl pyarrow
```

### Individual Format Dependencies

```bash
# HTML/Markdown support
pip install beautifulsoup4

# Word document support  
pip install python-docx

# PDF support (with OCR)
pip install marker-pdf

# Excel support
pip install openpyxl

# Parquet/Feather support
pip install pyarrow
```

## Format-Specific Usage

### Text Files

**Supported extensions**: `.txt`

**Usage**:
```python
# Direct file path
pipeline.prep_data("document.txt")

# Multiple text files
pipeline.prep_data("text_files/")
```

**Best for**: Plain text documents, logs, simple text files

### HTML Files

**Supported extensions**: `.html`, `.htm`

**Requirements**: `beautifulsoup4`

**Usage**:
```python
# Single HTML file
pipeline.prep_data("webpage.html")

# Directory of HTML files
pipeline.prep_data("html_documents/")
```

**Features**:
- Extracts text content from HTML tags
- Removes HTML markup and formatting
- Preserves text structure and paragraphs

**Best for**: Web pages, HTML reports, scraped content

### Markdown Files

**Supported extensions**: `.md`

**Requirements**: `beautifulsoup4`

**Usage**:
```python
# Markdown documentation
pipeline.prep_data("README.md")

# Multiple markdown files
pipeline.prep_data("docs/")
```

**Features**:
- Converts markdown formatting to plain text
- Preserves document structure
- Handles headers, lists, and formatting

**Best for**: Documentation, README files, technical writing

### Word Documents

**Supported extensions**: `.docx`

**Requirements**: `python-docx`

**Usage**:
```python
# Word document
pipeline.prep_data("report.docx")

# Multiple Word files
pipeline.prep_data("word_documents/")
```

**Features**:
- Extracts text from Word documents
- Preserves paragraph structure
- Handles tables and formatting

**Best for**: Business documents, reports, formatted text

### PDF Files

**Supported extensions**: `.pdf`

**Requirements**: `marker-pdf`

**Usage**:
```python
# PDF document
pipeline.prep_data("document.pdf")

# Directory of PDFs
pipeline.prep_data("pdf_documents/")
```

**Features**:
- OCR support for scanned documents
- Text extraction from native PDFs
- Handles complex layouts and formatting

**Best for**: Scanned documents, reports, academic papers

**Note**: PDF processing may be slower due to OCR requirements

### CSV Files

**Supported extensions**: `.csv`

**Requirements**: `pandas`

**Usage**:
```python
# CSV with text column
pipeline.prep_data("data.csv")

# Specify text column
pipeline.prep_data("data.csv", target_column="text_content")
```

**Configuration**:
```yaml
data_preprocessing:
  target_column: "text_content"  # Column containing text to extract
```

**Best for**: Structured data with text columns, survey responses, tabular data

### Excel Files

**Supported extensions**: `.xlsx`, `.xls`

**Requirements**: `openpyxl`

**Usage**:
```python
# Excel file
pipeline.prep_data("spreadsheet.xlsx")

# Multiple Excel files
pipeline.prep_data("excel_files/")
```

**Configuration**:
```yaml
data_preprocessing:
  target_column: "description"  # Column with text content
```

**Best for**: Spreadsheets with text data, structured reports

### Parquet Files

**Supported extensions**: `.parquet`

**Requirements**: `pyarrow`

**Usage**:
```python
# Parquet file
pipeline.prep_data("data.parquet")

# Specify text column
pipeline.prep_data("data.parquet", target_column="content")
```

**Features**:
- High-performance columnar format
- Efficient compression
- Fast reading and writing

**Best for**: Large datasets, high-performance processing

### Feather Files

**Supported extensions**: `.feather`

**Requirements**: `pyarrow`

**Usage**:
```python
# Feather file
pipeline.prep_data("data.feather")

# Multiple feather files
pipeline.prep_data("feather_files/")
```

**Features**:
- Fast serialization
- Cross-language compatibility
- Efficient storage

**Best for**: Fast data exchange, temporary storage

## Configuration Examples

### CSV with Custom Column

```yaml
data_preprocessing:
  target_column: "text_content"  # Use specific column for text
```

```python
# Load CSV and specify text column
pipeline.prep_data("survey_responses.csv")
```

### Excel with Multiple Sheets

```python
# Load specific sheet from Excel
import pandas as pd

# Load Excel file
df = pd.read_excel("report.xlsx", sheet_name="responses")
pipeline.prep_data(df)
```

### Directory Processing

```python
# Process all files in directory
pipeline.prep_data("documents/")  # Mix of formats supported

# Process specific file types
pipeline.prep_data("pdf_documents/")  # Only PDFs
```

## Performance Considerations

### Format Performance Ranking

1. **Fastest**: Text, CSV, Parquet, Feather
2. **Medium**: HTML, Markdown, Excel
3. **Slowest**: PDF (due to OCR), Word documents

### Optimization Tips

#### For Large Datasets
```python
# Use Parquet for large datasets
pipeline.prep_data("large_dataset.parquet")

# Process in chunks
for chunk in pd.read_csv("large_file.csv", chunksize=1000):
    pipeline.prep_data(chunk)
```

#### For PDF Processing
```python
# Pre-process PDFs to text for better performance
import PyPDF2

def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Convert PDFs to text first
text_content = pdf_to_text("document.pdf")
pipeline.prep_data(pd.DataFrame({"text": [text_content]}))
```

## Error Handling

### Common Issues

#### Missing Dependencies
```python
# Check for required dependencies
try:
    import beautifulsoup4
    print("HTML support available")
except ImportError:
    print("Install beautifulsoup4 for HTML support: pip install beautifulsoup4")
```

#### Unsupported Formats
```python
# Check file extension
import os

file_path = "document.unknown"
extension = os.path.splitext(file_path)[1]

supported_extensions = ['.txt', '.html', '.md', '.docx', '.pdf', '.csv', '.xlsx', '.parquet', '.feather']

if extension not in supported_extensions:
    print(f"Unsupported format: {extension}")
    print(f"Supported formats: {supported_extensions}")
```

#### Corrupted Files
```python
# Handle corrupted files gracefully
def safe_load_file(file_path):
    try:
        return pipeline.prep_data(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
```

## Best Practices

### 1. Choose Appropriate Format

```python
# For structured data with text
pipeline.prep_data("survey_data.csv")  # Use CSV

# For documents
pipeline.prep_data("report.pdf")  # Use PDF

# For web content
pipeline.prep_data("webpage.html")  # Use HTML
```

### 2. Preprocess When Needed

```python
# Convert complex formats to simpler ones
def preprocess_documents(input_dir, output_dir):
    for file_path in Path(input_dir).glob("*.pdf"):
        # Convert PDF to text
        text = extract_text_from_pdf(file_path)
        
        # Save as text file
        output_path = Path(output_dir) / f"{file_path.stem}.txt"
        output_path.write_text(text)
```

### 3. Handle Mixed Formats

```python
# Process directory with mixed formats
def process_mixed_directory(dir_path):
    supported_files = []
    
    for file_path in Path(dir_path).rglob("*"):
        if file_path.suffix.lower() in ['.txt', '.html', '.md', '.docx', '.pdf', '.csv', '.xlsx']:
            supported_files.append(file_path)
    
    # Process each file
    for file_path in supported_files:
        try:
            pipeline.prep_data(file_path)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
```

### 4. Optimize for Performance

```python
# Use fastest format for your data
# Convert to Parquet for large datasets
df = pd.read_csv("large_data.csv")
df.to_parquet("large_data.parquet")

# Use Parquet for processing
pipeline.prep_data("large_data.parquet")
```

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.

## Next Steps

- [Text Processing](text-processing.md) - Optimize text splitting and scoring
- [Batch Processing](batch-processing.md) - Handle large datasets efficiently
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
