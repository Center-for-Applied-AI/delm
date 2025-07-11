"""
DELM Scoring Strategies
======================
Pluggable strategies for scoring text relevance.
"""

from abc import ABC, abstractmethod
from typing import Sequence


class RelevanceScorer(ABC):
    """Abstract base class for relevance scoring strategies."""
    
    @abstractmethod
    def score(self, text_chunk: str) -> float:
        raise NotImplementedError


class KeywordScorer(RelevanceScorer):
    """Score text based on keyword presence."""
    
    def __init__(self, keywords: Sequence[str]):
        self.keywords = [kw.lower() for kw in keywords]

    def score(self, text_chunk: str) -> float:
        lowered = text_chunk.lower()
        return float(any(kw in lowered for kw in self.keywords))


class FuzzyScorer(RelevanceScorer):
    """Score text using fuzzy matching with rapidfuzz."""
    
    def __init__(self, keywords: Sequence[str]):
        self.keywords = [kw.lower() for kw in keywords]
        try:
            from rapidfuzz import fuzz  # type: ignore
        except ImportError:
            fuzz = None  # type: ignore
        self.fuzz = fuzz

    def score(self, text_chunk: str) -> float:  # 0‑1 range
        if self.fuzz is None:
            return KeywordScorer(self.keywords).score(text_chunk)
        lowered = text_chunk.lower()
        return max(self.fuzz.partial_ratio(lowered, kw) / 100 for kw in self.keywords) 