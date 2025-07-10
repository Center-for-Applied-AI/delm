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
    "RetryHandler"
] 