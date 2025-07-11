"""
DELM Splitting Strategies
========================
Pluggable strategies for splitting text into chunks.
"""

import re
from abc import ABC, abstractmethod
from typing import List


class SplitStrategy(ABC):
    """Return list[str] given raw document text – override .split."""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


class ParagraphSplit(SplitStrategy):
    """Split text into paragraph text chunks by newlines."""
    
    REGEX = re.compile(r"\r?\n\s*\r?\n")

    def split(self, text: str) -> List[str]:
        return [p.strip() for p in self.REGEX.split(text) if p.strip()]


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


class RegexSplit(SplitStrategy):
    """Split text using a custom regex pattern."""
    
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def split(self, text: str) -> List[str]:
        return [p.strip() for p in self.pattern.split(text) if p.strip()] 