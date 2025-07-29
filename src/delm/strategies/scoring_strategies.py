"""
DELM Scoring Strategies
======================
Pluggable strategies for scoring text relevance.
"""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

# Module-level logger
log = logging.getLogger(__name__)


class RelevanceScorer(ABC):
    """Abstract base class for relevance scoring strategies."""

    @abstractmethod
    def score(self, text_chunk: str) -> float:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "RelevanceScorer":
        log.debug(f"Creating RelevanceScorer from dict: {data}")
        if "type" not in data:
            log.error("Scorer config missing 'type' field")
            raise ValueError("Scorer config must include a 'type' field.")
        scorer_type = data["type"]
        log.debug(f"Scorer type: {scorer_type}")
        if scorer_type not in SCORER_REGISTRY:
            log.error(f"Unknown scorer type: {scorer_type}, available: {list(SCORER_REGISTRY.keys())}")
            raise ValueError(f"Unknown scorer type: {scorer_type}")
        scorer = SCORER_REGISTRY[scorer_type].from_dict(data)
        log.debug(f"Created scorer: {type(scorer).__name__}")
        return scorer

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class KeywordScorer(RelevanceScorer):
    """Score text based on keyword presence."""

    def __init__(self, keywords: Sequence[str]):
        log.debug(f"Initializing KeywordScorer with {len(keywords)} keywords")
        if not keywords or not isinstance(keywords, (list, tuple)):
            log.error("KeywordScorer requires a non-empty 'keywords' list")
            raise ValueError("KeywordScorer requires a non-empty 'keywords' list.")
        self.keywords = [kw.lower() for kw in keywords]
        log.debug(f"KeywordScorer initialized with keywords: {self.keywords}")

    def score(self, text_chunk: str) -> float:
        log.debug(f"Scoring text chunk with KeywordScorer (length: {len(text_chunk)})")
        lowered = text_chunk.lower()
        score = float(any(kw in lowered for kw in self.keywords))
        log.debug(f"KeywordScorer score: {score}")
        return score

    @classmethod
    def from_dict(cls, data: dict) -> "KeywordScorer":
        log.debug(f"Creating KeywordScorer from dict: {data}")
        if "keywords" not in data:
            log.error("KeywordScorer config missing 'keywords' field")
            raise ValueError("KeywordScorer config requires a 'keywords' field.")
        scorer = cls(data["keywords"])
        log.debug(f"KeywordScorer created from dict with {len(scorer.keywords)} keywords")
        return scorer

    def to_dict(self) -> dict:
        return {"type": "KeywordScorer", "keywords": self.keywords}


class FuzzyScorer(RelevanceScorer):
    """Score text using fuzzy matching with rapidfuzz."""

    def __init__(self, keywords: Sequence[str]):
        log.debug(f"Initializing FuzzyScorer with {len(keywords)} keywords")
        if not keywords or not isinstance(keywords, (list, tuple)):
            log.error("FuzzyScorer requires a non-empty 'keywords' list")
            raise ValueError("FuzzyScorer requires a non-empty 'keywords' list.")
        self.keywords = [kw.lower() for kw in keywords]
        log.debug(f"FuzzyScorer keywords: {self.keywords}")
        
        try:
            from rapidfuzz import fuzz  # type: ignore
            log.debug("rapidfuzz imported successfully")
        except ImportError:
            log.warning("rapidfuzz not available, falling back to KeywordScorer")
            fuzz = None  # type: ignore
        self.fuzz = fuzz

    def score(self, text_chunk: str) -> float:  # 0‑1 range
        log.debug(f"Scoring text chunk with FuzzyScorer (length: {len(text_chunk)})")
        if self.fuzz is None:
            log.debug("Using KeywordScorer fallback (rapidfuzz not available)")
            return KeywordScorer(self.keywords).score(text_chunk)
        lowered = text_chunk.lower()
        score = max(self.fuzz.partial_ratio(lowered, kw) / 100 for kw in self.keywords)
        log.debug(f"FuzzyScorer score: {score}")
        return score

    @classmethod
    def from_dict(cls, data: dict) -> "FuzzyScorer":
        log.debug(f"Creating FuzzyScorer from dict: {data}")
        if "keywords" not in data:
            log.error("FuzzyScorer config missing 'keywords' field")
            raise ValueError("FuzzyScorer config requires a 'keywords' field.")
        scorer = cls(data["keywords"])
        log.debug(f"FuzzyScorer created from dict with {len(scorer.keywords)} keywords")
        return scorer

    def to_dict(self) -> dict:
        return {"type": "FuzzyScorer", "keywords": self.keywords}


# Factory registry for scorer types
SCORER_REGISTRY = {
    "KeywordScorer": KeywordScorer,
    "FuzzyScorer": FuzzyScorer,
}