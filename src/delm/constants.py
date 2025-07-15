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
DEFAULT_CHUNK_COLUMN = "text_chunk"  # Default column name for text chunks
DEFAULT_SCORE_COLUMN = "score"       # Default column name for relevance scores

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
DEFAULT_VERBOSE = False             # Whether to enable verbose logging by default

# Extraction Defaults
DEFAULT_EXTRACT_TO_DATAFRAME = False  # Whether to extract JSON to DataFrame by default

# System Constants (Internal - Not User Configurable)
SYSTEM_CHUNK_COLUMN = "text_chunk"   # Internal column name for text chunks
SYSTEM_SCORE_COLUMN = "score"        # Internal column name for relevance scores
SYSTEM_CHUNK_ID_COLUMN = "chunk_id"  # Internal column name for chunk IDs
SYSTEM_EXTRACTED_DATA_COLUMN = "extracted_data"  # Internal column name for extracted JSON output
PREPROCESSED_DIR_NAME = "preprocessed"  # Directory name for preprocessed data 