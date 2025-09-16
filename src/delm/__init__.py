"""
DELM - Data Extraction Language Model
A pipeline for extracting structured data from text using language models.
"""

import logging

# Library-local logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())   # avoids spurious warnings

from delm.delm import DELM
from delm.logging import configure as configure_logging
from delm.config import DELMConfig, LLMExtractionConfig, DataPreprocessingConfig, SchemaConfig, SplittingConfig, ScoringConfig
from delm.exceptions import (
    DELMError, 
    ExperimentManagementError, 
    InstructorError
)
from .constants import (
    # LLM/API Configuration
    DEFAULT_PROVIDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_BASE_DELAY,
    DEFAULT_TRACK_COST,
    DEFAULT_MAX_BUDGET,
    DEFAULT_DOTENV_PATH,
    
    # Data Processing
    DEFAULT_DROP_TARGET_COLUMN,
    DEFAULT_PANDAS_SCORE_FILTER,
    
    # Schema Configuration
    DEFAULT_SCHEMA_PATH,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    
    # Experiment Management
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_OVERWRITE_EXPERIMENT,
    DEFAULT_AUTO_CHECKPOINT_AND_RESUME,
    
    # Semantic Cache
    DEFAULT_SEMANTIC_CACHE_BACKEND,
    DEFAULT_SEMANTIC_CACHE_PATH,
    DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB,
    DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS,
    
    # System Constants
    SYSTEM_FILE_NAME_COLUMN,
    SYSTEM_RAW_DATA_COLUMN,
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_SCORE_COLUMN,
    SYSTEM_BATCH_ID_COLUMN,
    SYSTEM_ERRORS_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
    SYSTEM_RANDOM_SEED,
    
    # File and Directory Constants
    DATA_DIR_NAME,
    CACHE_DIR_NAME,
    PROCESSING_CACHE_DIR_NAME,
    BATCH_FILE_PREFIX,
    BATCH_FILE_SUFFIX,
    BATCH_FILE_DIGITS,
    STATE_FILE_NAME,
    CONSOLIDATED_RESULT_PREFIX,
    CONSOLIDATED_RESULT_SUFFIX,
    PREPROCESSED_DATA_PREFIX,
    PREPROCESSED_DATA_SUFFIX,
    META_DATA_PREFIX,
    META_DATA_SUFFIX,
    
    # Utility Constants
    IGNORE_FILES,
)

__version__ = "0.1.3"
__author__ = "Eric Fithian - Chicago Booth CAAI Lab"

__all__ = [
    # Main Classes
    "DELM",
    "DELMConfig",
    "LLMExtractionConfig",
    "DataPreprocessingConfig",
    "SchemaConfig",
    "SplittingConfig",
    "ScoringConfig",
    
    # Exceptions
    "DELMError",
    "ExperimentManagementError",
    "InstructorError",
    
    # LLM/API Configuration
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_TRACK_COST",
    "DEFAULT_MAX_BUDGET",
    "DEFAULT_DOTENV_PATH",
    
    # Data Processing
    "DEFAULT_DROP_TARGET_COLUMN",
    "DEFAULT_PANDAS_SCORE_FILTER",
    
    # Schema Configuration
    "DEFAULT_SCHEMA_PATH",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_SYSTEM_PROMPT",
    
    # Experiment Management
    "DEFAULT_EXPERIMENT_DIR",
    "DEFAULT_OVERWRITE_EXPERIMENT",
    "DEFAULT_AUTO_CHECKPOINT_AND_RESUME",
    
    # Semantic Cache
    "DEFAULT_SEMANTIC_CACHE_BACKEND",
    "DEFAULT_SEMANTIC_CACHE_PATH",
    "DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB",
    "DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS",
    
    # System Constants
    "SYSTEM_FILE_NAME_COLUMN",
    "SYSTEM_RAW_DATA_COLUMN",
    "SYSTEM_RECORD_ID_COLUMN",
    "SYSTEM_CHUNK_COLUMN",
    "SYSTEM_CHUNK_ID_COLUMN",
    "SYSTEM_SCORE_COLUMN",
    "SYSTEM_BATCH_ID_COLUMN",
    "SYSTEM_ERRORS_COLUMN",
    "SYSTEM_EXTRACTED_DATA_JSON_COLUMN",
    "SYSTEM_RANDOM_SEED",
    
    # File and Directory Constants
    "DATA_DIR_NAME",
    "CACHE_DIR_NAME",
    "PROCESSING_CACHE_DIR_NAME",
    "BATCH_FILE_PREFIX",
    "BATCH_FILE_SUFFIX",
    "BATCH_FILE_DIGITS",
    "STATE_FILE_NAME",
    "CONSOLIDATED_RESULT_PREFIX",
    "CONSOLIDATED_RESULT_SUFFIX",
    "PREPROCESSED_DATA_PREFIX",
    "PREPROCESSED_DATA_SUFFIX",
    "META_DATA_PREFIX",
    "META_DATA_SUFFIX",
    
    # Utility Constants
    "IGNORE_FILES",
    
    # Logging
    "configure_logging",
] 