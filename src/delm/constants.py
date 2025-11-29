"""
DELM Constants
==============
Default values and system constants for the DELM (Data Extraction with Language Models) framework.

This file contains all configuration defaults and system constants organized by category.
"""

from pathlib import Path

# =============================================================================
# SYSTEM CONSTANTS (Internal Use Only)
# =============================================================================
# These constants define internal column names and system behavior.
# They should NOT be used in user data or configuration.

# System Column Names
SYSTEM_FILE_NAME_COLUMN = "delm_file_name"  # Column for source file names
SYSTEM_RAW_DATA_COLUMN = "delm_raw_data"  # Column for original text data
SYSTEM_RECORD_ID_COLUMN = "delm_record_id"  # Column for internal unique record IDs
SYSTEM_CHUNK_COLUMN = "delm_text_chunk"  # Column for text chunks
SYSTEM_CHUNK_ID_COLUMN = "delm_chunk_id"  # Column for internal chunk IDs
SYSTEM_SCORE_COLUMN = "delm_score"  # Column for relevance scores
SYSTEM_BATCH_ID_COLUMN = "delm_batch_id"  # Column for batch IDs
SYSTEM_ERRORS_COLUMN = "delm_errors"  # Column for error messages

# Data Storage Columns
SYSTEM_EXTRACTED_DATA_JSON_COLUMN = (
    "delm_extracted_data_json"  # Column for extracted JSON data
)

# System Behavior Constants
SYSTEM_RANDOM_SEED = 42  # Random seed for reproducibility

# =============================================================================
# FILE AND DIRECTORY CONSTANTS
# =============================================================================

# Directory Names
DATA_DIR_NAME = "delm_data"  # Name of data directory
PROCESSING_CACHE_DIR_NAME = (
    "delm_llm_processing"  # Name of processing cache subdirectory
)

# File Naming Patterns
BATCH_FILE_PREFIX = "batch_"  # Prefix for batch files
BATCH_FILE_SUFFIX = ".feather"  # Suffix for batch files
BATCH_FILE_DIGITS = 6  # Number of digits in batch file names

# State and Result Files
STATE_FILE_NAME = "state.json"  # Name of state file
CONSOLIDATED_RESULT_FILE_NAME = (
    "extraction_result.feather"  # File name for consolidated results
)

# Preprocessed Data Files
PREPROCESSED_DATA_FILE_NAME = (
    "preprocessed.feather"  # File name for preprocessed data files
)

# Metadata Files
META_DATA_FILE_NAME = "meta_data.feather"  # File name for metadata files

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Logging Settings
SYSTEM_LOG_FILE_PREFIX = "delm_"  # Default prefix for log files
SYSTEM_LOG_FILE_SUFFIX = ".log"  # Default suffix for log files

# =============================================================================
# UTILITY CONSTANTS
# =============================================================================

# Files to Ignore
IGNORE_FILES = [
    ".DS_Store",  # macOS system files
]

LLM_NULL_WORDS_LOWERCASE = [
    "none",
    "null",
    "unknown",
    "n/a",
    "",
]
