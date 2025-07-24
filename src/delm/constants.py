"""
DELM Constants
==============
Default values for configuration, grouped by category.
"""
from pathlib import Path

# API/Model Defaults
DEFAULT_PROVIDER = "openai"           # Default LLM provider (openai, anthropic, google, etc.)
DEFAULT_MODEL_NAME = "gpt-4o-mini"    # Default LLM model name
DEFAULT_TEMPERATURE = 0.0             # Default temperature for LLM
DEFAULT_MAX_RETRIES = 3              # Default max retries for API calls
DEFAULT_BATCH_SIZE = 10              # Default batch size for processing
DEFAULT_MAX_WORKERS = 1              # Default number of concurrent workers
DEFAULT_BASE_DELAY = 1.0             # Default base delay for retry handler
DEFAULT_DOTENV_PATH = None           # Default dotenv path (None = no .env file)
DEFAULT_REGEX_FALLBACK_PATTERN = None # Default regex fallback pattern (None = no fallback)
DEFAULT_TRACK_COST = True            # Default bool to track cost of API calls

# Data Processing Defaults
DEFAULT_TARGET_COLUMN = "text"       # Default target column in data
DEFAULT_DROP_TARGET_COLUMN = True    # Whether to drop target column after processing
DEFAULT_PANDAS_SCORE_FILTER = None  # Default pandas score filter (None = no filter)

# Schema Defaults
DEFAULT_SCHEMA_CONTAINER = "data"    # Default container name for schema
DEFAULT_PROMPT_TEMPLATE = """Extract the following information from the text:

{variables}

Text to analyze:
{text}

Please extract the requested information accurately and return it in the specified format. If a field is not mentioned in the text, use null/None rather than guessing."""  # Default prompt template

# Experiment Defaults
DEFAULT_EXPERIMENT_DIR = Path("delm_experiments") # Default experiment directory
DEFAULT_OVERWRITE_EXPERIMENT = False # Whether to overwrite existing experiments by default

# Extraction Defaults
DEFAULT_EXTRACT_TO_DATAFRAME = False  # Whether to extract JSON to DataFrame by default

# System Prompt Default
DEFAULT_SYSTEM_PROMPT = "You are a precise data‑extraction assistant."

# System Constants (Internal - Not User Configurable)
# TODO: Throw error if these are used in the data.
SYSTEM_RECORD_ID_COLUMN = "delm_record_id"
SYSTEM_CHUNK_COLUMN = "delm_text_chunk"   # Internal column name for text chunks
SYSTEM_SCORE_COLUMN = "delm_score"        # Internal column name for relevance scores
SYSTEM_CHUNK_ID_COLUMN = "delm_chunk_id"  # Internal column name for chunk IDs
# Internal column name for extracted data (dict in memory, JSON string on disk)
SYSTEM_EXTRACTED_DATA_JSON_COLUMN = "delm_extracted_data_json" # For saving to disk
SYSTEM_BATCH_ID_COLUMN = "delm_batch_id"  # Internal column name for batch IDs
SYSTEM_ERRORS_COLUMN = "delm_errors"
SYSTEM_REGEX_EXTRACTED_KEY = "delm_regex_extracted_data"

SYSTEM_RANDOM_SEED = 42

DATA_DIR_NAME = "delm_data"

# Checkpointing and Cache Constants
CACHE_DIR_NAME = ".delm_cache"
PROCESSING_CACHE_DIR_NAME = "llm_processing"
BATCH_FILE_PREFIX = "batch_"
BATCH_FILE_SUFFIX = ".feather"
BATCH_FILE_DIGITS = 6
STATE_FILE_NAME = "state.json"
CONSOLIDATED_RESULT_PREFIX = "extraction_result_"
CONSOLIDATED_RESULT_SUFFIX = ".feather"
PREPROCESSED_DATA_PREFIX = "preprocessed_"
PREPROCESSED_DATA_SUFFIX = ".feather"
META_DATA_PREFIX = "meta_data_"
META_DATA_SUFFIX = ".feather"

# Config key for auto checkpointing
DEFAULT_AUTO_CHECKPOINT_AND_RESUME = True 