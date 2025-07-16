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

    @classmethod
    def from_dict(cls, data: dict) -> "RelevanceScorer":
        if "type" not in data:
            raise ValueError("Scorer config must include a 'type' field.")
        scorer_type = data["type"]
        if scorer_type not in SCORER_REGISTRY:
            raise ValueError(f"Unknown scorer type: {scorer_type}")
        return SCORER_REGISTRY[scorer_type].from_dict(data)

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class KeywordScorer(RelevanceScorer):
    """Score text based on keyword presence."""

    def __init__(self, keywords: Sequence[str]):
        if not keywords or not isinstance(keywords, (list, tuple)):
            raise ValueError("KeywordScorer requires a non-empty 'keywords' list.")
        self.keywords = [kw.lower() for kw in keywords]

    def score(self, text_chunk: str) -> float:
        lowered = text_chunk.lower()
        return float(any(kw in lowered for kw in self.keywords))

    @classmethod
    def from_dict(cls, data: dict) -> "KeywordScorer":
        if "keywords" not in data:
            raise ValueError("KeywordScorer config requires a 'keywords' field.")
        return cls(data["keywords"])

    def to_dict(self) -> dict:
        return {"type": "KeywordScorer", "keywords": self.keywords}


class FuzzyScorer(RelevanceScorer):
    """Score text using fuzzy matching with rapidfuzz."""

    def __init__(self, keywords: Sequence[str]):
        if not keywords or not isinstance(keywords, (list, tuple)):
            raise ValueError("FuzzyScorer requires a non-empty 'keywords' list.")
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

    @classmethod
    def from_dict(cls, data: dict) -> "FuzzyScorer":
        if "keywords" not in data:
            raise ValueError("FuzzyScorer config requires a 'keywords' field.")
        return cls(data["keywords"])

    def to_dict(self) -> dict:
        return {"type": "FuzzyScorer", "keywords": self.keywords}


# Factory registry for scorer types
SCORER_REGISTRY = {
    "KeywordScorer": KeywordScorer,
    "FuzzyScorer": FuzzyScorer,
}