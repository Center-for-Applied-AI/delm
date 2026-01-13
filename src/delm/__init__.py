"""
DELM - Data Extraction Language Model
A pipeline for extracting structured data from text using language models.
"""

import logging

# Library-local logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())  # avoids spurious warnings

from delm.delm import DELM
from delm.config import (
    DELMConfig,
    LLMExtractionConfig,
    DataPreprocessingConfig,
    SemanticCacheConfig,
)
from delm.exceptions import DELMError, ExperimentManagementError, InstructorError
from .constants import (
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
    PROCESSING_CACHE_DIR_NAME,
    BATCH_FILE_PREFIX,
    BATCH_FILE_SUFFIX,
    BATCH_FILE_DIGITS,
    STATE_FILE_NAME,
    CONSOLIDATED_RESULT_FILE_NAME,
    PREPROCESSED_DATA_FILE_NAME,
    META_DATA_FILE_NAME,
    # Utility Constants
    IGNORE_FILES,
)
from delm.schemas import Schema
from delm.models import ExtractionVariable

__version__ = "1.0.1"
__author__ = "Eric Fithian - Chicago Booth CAAI Lab"

__all__ = [
    # Main Classes
    "DELM",
    "DELMConfig",
    "Schema",
    "ExtractionVariable",
    "LLMExtractionConfig",
    "DataPreprocessingConfig",
    "SemanticCacheConfig",
    # Exceptions
    "DELMError",
    "ExperimentManagementError",
    "InstructorError",
    # Schema Configuration
    # Experiment Management
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
    "PROCESSING_CACHE_DIR_NAME",
    "BATCH_FILE_PREFIX",
    "BATCH_FILE_SUFFIX",
    "BATCH_FILE_DIGITS",
    "STATE_FILE_NAME",
    "CONSOLIDATED_RESULT_FILE_NAME",
    "PREPROCESSED_DATA_FILE_NAME",
    "META_DATA_FILE_NAME",
    # Utility Constants
    "IGNORE_FILES",
]
