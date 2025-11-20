# System Constants

DELM system constants for column names and defaults.

## System Columns

Column names automatically added by DELM.

```python
from delm import (
    SYSTEM_FILE_NAME_COLUMN,
    SYSTEM_RAW_DATA_COLUMN,
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_SCORE_COLUMN,
    SYSTEM_BATCH_ID_COLUMN,
    SYSTEM_ERRORS_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN
)
```

| Constant | Value | Description |
|----------|-------|-------------|
| `SYSTEM_FILE_NAME_COLUMN` | `"delm_file_name"` | Source filename (directories only) |
| `SYSTEM_RAW_DATA_COLUMN` | `"delm_raw_data"` | Original raw text (before splitting) |
| `SYSTEM_RECORD_ID_COLUMN` | `"delm_record_id"` | Unique record identifier |
| `SYSTEM_CHUNK_COLUMN` | `"delm_text_chunk"` | Text chunk (after splitting) |
| `SYSTEM_CHUNK_ID_COLUMN` | `"delm_chunk_id"` | Unique chunk identifier |
| `SYSTEM_SCORE_COLUMN` | `"delm_score"` | Relevance score (if scorer used) |
| `SYSTEM_BATCH_ID_COLUMN` | `"delm_batch_id"` | Batch number |
| `SYSTEM_ERRORS_COLUMN` | `"delm_errors"` | Extraction errors (if any) |
| `SYSTEM_EXTRACTED_DATA_JSON_COLUMN` | `"delm_extracted_data_json"` | Extracted JSON data |

## Experiment Directory

Constants for disk storage structure.

```python
from delm import (
    DATA_DIR_NAME,
    PROCESSING_CACHE_DIR_NAME,
    BATCH_FILE_PREFIX,
    STATE_FILE_NAME,
    CONSOLIDATED_RESULT_FILE_NAME,
    PREPROCESSED_DATA_FILE_NAME
)
```

| Constant | Value | Description |
|----------|-------|-------------|
| `DATA_DIR_NAME` | `"delm_data"` | Preprocessed data directory |
| `PROCESSING_CACHE_DIR_NAME` | `"delm_llm_processing"` | LLM processing cache directory |
| `BATCH_FILE_PREFIX` | `"batch_"` | Batch file prefix |
| `STATE_FILE_NAME` | `"state.json"` | Checkpoint state file |
| `CONSOLIDATED_RESULT_FILE_NAME` | `"extraction_result.feather"` | Final results file |
| `PREPROCESSED_DATA_FILE_NAME` | `"preprocessed.feather"` | Preprocessed data file |

## Other Constants

```python
from delm import SYSTEM_RANDOM_SEED, IGNORE_FILES
```

| Constant | Value | Description |
|----------|-------|-------------|
| `SYSTEM_RANDOM_SEED` | `42` | Random seed for sampling |
| `IGNORE_FILES` | `[".DS_Store", ...]` | Files to ignore when loading directories |

## Usage Example

```python
from delm import DELM, SYSTEM_EXTRACTED_DATA_JSON_COLUMN, SYSTEM_SCORE_COLUMN

delm = DELM(schema=schema, relevance_scorer=scorer)
results = delm.extract("data.csv")

# Access system columns
results[SYSTEM_EXTRACTED_DATA_JSON_COLUMN]  # Extracted JSON
results[SYSTEM_SCORE_COLUMN]  # Relevance scores
```

