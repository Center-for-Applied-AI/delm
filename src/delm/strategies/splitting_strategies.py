"""
DELM Splitting Strategies
========================
Pluggable strategies for splitting text into chunks.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Type


class SplitStrategy(ABC):
    """Return list[str] given raw document text – override .split."""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "SplitStrategy":
        if "type" not in data:
            raise ValueError("Splitter config must include a 'type' field.")
        splitter_type = data["type"]
        if splitter_type not in SPLITTER_REGISTRY:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
        return SPLITTER_REGISTRY[splitter_type].from_dict(data)

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class ParagraphSplit(SplitStrategy):
    """Split text into paragraph text chunks by newlines."""
    
    REGEX = re.compile(r"\r?\n\s*\r?\n")

    def split(self, text: str) -> List[str]:
        return [p.strip() for p in self.REGEX.split(text) if p.strip()]

    @classmethod
    def from_dict(cls, data: dict) -> "ParagraphSplit":
        return cls()

    def to_dict(self) -> dict:
        return {"type": "ParagraphSplit"}


class FixedWindowSplit(SplitStrategy):
    """Split text into fixed-size windows of sentences."""
    
    def __init__(self, window: int = 5, stride: int | None = None):
        self.window, self.stride = window, stride or window

    def split(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, i = [], 0
        while i < len(sentences):
            chunk = " ".join(sentences[i : i + self.window])
            chunks.append(chunk.strip())
            i += self.stride
        return [c for c in chunks if c]

    @classmethod
    def from_dict(cls, data: dict) -> "FixedWindowSplit":
        window = data.get("window", 5)
        stride = data.get("stride", None)
        return cls(window=window, stride=stride)

    def to_dict(self) -> dict:
        return {"type": "FixedWindowSplit", "window": self.window, "stride": self.stride}


class RegexSplit(SplitStrategy):
    """Split text using a custom regex pattern."""
    
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)
        self.pattern_str = pattern

    def split(self, text: str) -> List[str]:
        return [p.strip() for p in self.pattern.split(text) if p.strip()]

    @classmethod
    def from_dict(cls, data: dict) -> "RegexSplit":
        if "pattern" not in data:
            raise ValueError("RegexSplit config requires a 'pattern' field.")
        return cls(data["pattern"])

    def to_dict(self) -> dict:
        return {"type": "RegexSplit", "pattern": self.pattern_str}


# Factory registry for splitter types
SPLITTER_REGISTRY: Dict[str, Type[SplitStrategy]] = {
    "ParagraphSplit": ParagraphSplit,
    "FixedWindowSplit": FixedWindowSplit,
    "RegexSplit": RegexSplit,
}