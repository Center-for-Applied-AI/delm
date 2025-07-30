"""
DELM Constants
==============
Default values and system constants for the DELM (Data Extraction with Language Models) framework.

This file contains all configuration defaults and system constants organized by category.
"""

from pathlib import Path

# =============================================================================
# LLM/API CONFIGURATION DEFAULTS
# =============================================================================

# Provider and Model Settings
DEFAULT_PROVIDER = "openai"           # LLM provider (openai, anthropic, google, etc.)
DEFAULT_MODEL_NAME = "gpt-4o-mini"    # LLM model name
DEFAULT_TEMPERATURE = 0.0             # Temperature for LLM responses (0.0 = deterministic)

# API Request Settings
DEFAULT_MAX_RETRIES = 3               # Maximum retry attempts for failed API calls
DEFAULT_BASE_DELAY = 1.0              # Base delay between retries (seconds)

# Processing Settings
DEFAULT_BATCH_SIZE = 10               # Number of records to process in each batch
DEFAULT_MAX_WORKERS = 1               # Number of concurrent worker processes

# Cost and Budget Settings
DEFAULT_TRACK_COST = True             # Whether to track API call costs
DEFAULT_MAX_BUDGET = None             # Maximum budget limit (None = no limit)

# Environment Settings
DEFAULT_DOTENV_PATH = None            # Path to .env file

# =============================================================================
# DATA PROCESSING DEFAULTS
# =============================================================================

## Splitting Defaults
# FixedWindowSplit
DEFAULT_FIXED_WINDOW_SIZE = 5         # Number of sentences per chunk
DEFAULT_FIXED_WINDOW_STRIDE = 5       # Number of sentences to overlap
# RegexSplit
DEFAULT_REGEX_PATTERN = "\n\n"        # Regex pattern to split on

# Column and Data Settings
DEFAULT_DROP_TARGET_COLUMN = False    # Whether to drop the target column after processing
DEFAULT_PANDAS_SCORE_FILTER = None    # Pandas query string for filtering by score (None = no filter)

# Extraction Settings
DEFAULT_EXPLODE_JSON_RESULTS = False  # Whether to convert extracted JSON to DataFrame

# =============================================================================
# SCHEMA CONFIGURATION DEFAULTS
# =============================================================================

# Schema File Settings
DEFAULT_SCHEMA_PATH = None            # Default path to schema specification file

# Prompt Settings
DEFAULT_PROMPT_TEMPLATE = """Extract the following information from the text:

{variables}

Text to analyze:
{text}

Please extract the requested information accurately and return it in the specified format. If a field is not mentioned in the text, use null/None rather than guessing."""

DEFAULT_SYSTEM_PROMPT = "You are a precise data‑extraction assistant."

# =============================================================================
# EXPERIMENT MANAGEMENT DEFAULTS
# =============================================================================

DEFAULT_EXPERIMENT_DIR = Path("delm_experiments")  # Default directory for experiment outputs
DEFAULT_OVERWRITE_EXPERIMENT = False               # Whether to overwrite existing experiments
DEFAULT_AUTO_CHECKPOINT_AND_RESUME = True         # Whether to automatically checkpoint and resume

# =============================================================================
# SEMANTIC CACHE DEFAULTS
# =============================================================================

# Cache Backend Settings
DEFAULT_SEMANTIC_CACHE_BACKEND = "sqlite"          # Cache backend: "sqlite" | "lmdb" | "filesystem"
DEFAULT_SEMANTIC_CACHE_PATH = ".delm_cache"        # Cache directory path
DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB = 512          # Maximum cache size before pruning
DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS = "normal"      # SQLite sync mode: "normal" | "full"

# =============================================================================
# SYSTEM CONSTANTS (Internal Use Only)
# =============================================================================
# These constants define internal column names and system behavior.
# They should NOT be used in user data or configuration.

# System Column Names
SYSTEM_FILE_NAME_COLUMN = "delm_file_name"                    # Column for source file names
SYSTEM_RAW_DATA_COLUMN = "delm_raw_data"                      # Column for original text data
SYSTEM_RECORD_ID_COLUMN = "delm_record_id"                    # Column for internal unique record IDs
SYSTEM_CHUNK_COLUMN = "delm_text_chunk"                       # Column for text chunks
SYSTEM_CHUNK_ID_COLUMN = "delm_chunk_id"                      # Column for internal chunk IDs
SYSTEM_SCORE_COLUMN = "delm_score"                            # Column for relevance scores
SYSTEM_BATCH_ID_COLUMN = "delm_batch_id"                      # Column for batch IDs
SYSTEM_ERRORS_COLUMN = "delm_errors"                          # Column for error messages

# Data Storage Columns
SYSTEM_EXTRACTED_DATA_JSON_COLUMN = "delm_extracted_data_json"  # Column for extracted JSON data

# System Behavior Constants
SYSTEM_RANDOM_SEED = 42                                        # Random seed for reproducibility

# =============================================================================
# FILE AND DIRECTORY CONSTANTS
# =============================================================================

# Directory Names
DATA_DIR_NAME = "delm_data"                                    # Name of data directory
CACHE_DIR_NAME = ".delm_cache"                                 # Name of cache directory
PROCESSING_CACHE_DIR_NAME = "llm_processing"                   # Name of processing cache subdirectory

# File Naming Patterns
BATCH_FILE_PREFIX = "batch_"                                   # Prefix for batch files
BATCH_FILE_SUFFIX = ".feather"                                 # Suffix for batch files
BATCH_FILE_DIGITS = 6                                          # Number of digits in batch file names

# State and Result Files
STATE_FILE_NAME = "state.json"                                 # Name of state file
CONSOLIDATED_RESULT_PREFIX = "extraction_result_"              # Prefix for consolidated results
CONSOLIDATED_RESULT_SUFFIX = ".feather"                        # Suffix for consolidated results

# Preprocessed Data Files
PREPROCESSED_DATA_PREFIX = "preprocessed_"                     # Prefix for preprocessed data files
PREPROCESSED_DATA_SUFFIX = ".feather"                          # Suffix for preprocessed data files

# Metadata Files
META_DATA_PREFIX = "meta_data_"                                # Prefix for metadata files
META_DATA_SUFFIX = ".feather"                                  # Suffix for metadata files

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Logging Settings
DEFAULT_LOG_DIR = "delm_logs"                                  # Default directory for log files
SYSTEM_LOG_FILE_PREFIX = "delm_"                              # Default prefix for log files
SYSTEM_LOG_FILE_SUFFIX = ".log"                              # Default suffix for log files
DEFAULT_CONSOLE_LOG_LEVEL = "INFO"                            # Default console log level
DEFAULT_FILE_LOG_LEVEL = "DEBUG"                              # Default file log level

# =============================================================================
# UTILITY CONSTANTS
# =============================================================================

# Files to Ignore
IGNORE_FILES = [
    ".DS_Store",                                               # macOS system files
]

LLM_NULL_WORDS_LOWERCASE = [
    "none",
    "null",
    "unknown",
    "n/a",
    "",
] 