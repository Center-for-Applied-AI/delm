"""
DELM - Data Extraction Language Model
A pipeline for extracting structured data from text using language models.
"""

from .DELM import DELM
from .splitting_strategies import SplitStrategy, ParagraphSplit, FixedWindowSplit, RegexSplit
from .scoring_strategies import RelevanceScorer, KeywordScorer, FuzzyScorer
from .schemas import SchemaRegistry
from .models import ExtractionVariable
from .extraction_engine import ExtractionEngine
from .retry_handler import RetryHandler
from .config import DELMConfig, ModelConfig, DataConfig, SchemaConfig, ExperimentConfig, SplittingConfig, ScoringConfig, ConfigValidationError
from .constants import (
    DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS, DEFAULT_BASE_DELAY, DEFAULT_DOTENV_PATH, DEFAULT_REGEX_FALLBACK_PATTERN,
    DEFAULT_TARGET_COLUMN, DEFAULT_DROP_TARGET_COLUMN, DEFAULT_CHUNK_COLUMN, DEFAULT_SCORE_COLUMN,
    DEFAULT_SCHEMA_CONTAINER, DEFAULT_PROMPT_TEMPLATE, DEFAULT_EXPERIMENT_DIR,
    DEFAULT_SAVE_INTERMEDIATES, DEFAULT_OVERWRITE_EXPERIMENT, DEFAULT_VERBOSE, DEFAULT_KEYWORDS,
    # System constants
    SYSTEM_CHUNK_COLUMN, SYSTEM_SCORE_COLUMN, SYSTEM_LLM_JSON_COLUMN, SYSTEM_CHUNK_ID_COLUMN
)

__version__ = "0.2.0"
__author__ = "Eric Fithian - Chicago Booth CAAI Lab"

__all__ = [
    "DELM",
    "SplitStrategy",
    "ParagraphSplit", 
    "FixedWindowSplit",
    "RegexSplit",
    "RelevanceScorer",
    "KeywordScorer",
    "FuzzyScorer",
    "SchemaRegistry",
    "ExtractionVariable",
    "ExtractionEngine",
    "RetryHandler",
    "DELMConfig",
    "ModelConfig",
    "DataConfig",
    "SchemaConfig",
    "ExperimentConfig",
    "SplittingConfig",
    "ScoringConfig",
    "ConfigValidationError",
    # Constants
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEMPERATURE", 
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_DOTENV_PATH",
    "DEFAULT_REGEX_FALLBACK_PATTERN",
    "DEFAULT_TARGET_COLUMN",
    "DEFAULT_DROP_TARGET_COLUMN",
    "DEFAULT_CHUNK_COLUMN",
    "DEFAULT_SCORE_COLUMN",
    "DEFAULT_SCHEMA_CONTAINER",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_EXPERIMENT_DIR",
    "DEFAULT_SAVE_INTERMEDIATES",
    "DEFAULT_OVERWRITE_EXPERIMENT",
    "DEFAULT_VERBOSE",
    "DEFAULT_KEYWORDS",
    # System constants
    "SYSTEM_CHUNK_COLUMN",
    "SYSTEM_SCORE_COLUMN", 
    "SYSTEM_LLM_JSON_COLUMN",
    "SYSTEM_CHUNK_ID_COLUMN"
] 