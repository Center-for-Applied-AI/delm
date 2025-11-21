# Loading Data

DELM supports a wide range of input formats for text extraction. You can load single files, entire directories, or raw pandas DataFrames.

## Supported File Types

DELM supports the following file formats out of the box:

| Format | Extension | Notes |
|--------|-----------|-------|
| **CSV** | `.csv` | Requires target column |
| **Parquet** | `.parquet` | Requires target column |
| **Feather** | `.feather` | Requires target column |
| **Text** | `.txt` | |
| **Markdown** | `.md` |  |
| **Word** | `.docx` | |
| **HTML** | `.html`, `.htm` | Automatically strips tags to extract text |


### Optional Formats

The following formats require extra dependencies included in the `extras` package:

| Format | Extension | Installation | Notes |
|--------|-----------|--------------|-------|
| **PDF** | `.pdf` | `pip install delm[extras]` |  |
| **Excel** | `.xlsx`, `.xls` | `pip install delm[extras]` | Requires target column |

## Input Methods

You can load data into DELM using the `delm.prep_data()` method in three ways.

### 1. Single File

Pass the path to a single file. DELM will detect the format and load it.

```python
delm = DELM(
    ...
)

# Load a single document
delm.extract("documents/report_2024.pdf")
```

**Specifying Text Columns**:
If your CSV/Excel/Parquet file has text in a specific column (e.g., "comments"), specify it:

```python
delm = DELM(
    # ...
    target_column="comments"
)
delm.prep_data("data/survey_responses.csv")
```

### 2. Directory of Files

Pass a directory path to load all supported files within it. DELM find and loads any valid files.

```python
# Load all supported files in a directory
delm.prep_data("data/financial_reports/")
```

This is useful for processing a mixed collection of PDFs, Word docs, and text files in one go.

### 3. Pandas DataFrame

If you already have data in memory, you can pass a pandas DataFrame directly.

```python
import pandas as pd

df = pd.DataFrame({
    "text": [
        "Company A reported $10M revenue.",
        "Company B reported $5M revenue."
    ],
    "meta_id": [101, 102]
})

delm.prep_data(df)
```
