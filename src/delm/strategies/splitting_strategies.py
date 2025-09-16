"""
DELM Splitting Strategies
========================
Pluggable strategies for splitting text into chunks.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Type

# Module-level logger
log = logging.getLogger(__name__)


class SplitStrategy(ABC):
    """Return list[str] given raw document text – override .split."""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "SplitStrategy":
        """Create a SplitStrategy from a dictionary.

        Args:
            data: A dictionary containing the splitter configuration.

        Returns:
            A SplitStrategy instance.

        Raises:
            ValueError: If the splitter config is missing the 'type' field or the type is unknown or the splitter config is invalid.
        """
        log.debug(f"Creating SplitStrategy from dict: {data}")
        if "type" not in data:
            log.error(
                "Splitter config missing 'type' field, available types: %s",
                list(SPLITTER_REGISTRY.keys()),
            )
            raise ValueError(
                "Splitter config must include a 'type' field, available types: %s"
                % list(SPLITTER_REGISTRY.keys())
            )
        splitter_type = data["type"]
        log.debug(f"Splitter type: {splitter_type}")
        if splitter_type not in SPLITTER_REGISTRY:
            log.error(
                f"Unknown splitter type: {splitter_type}, available: {list(SPLITTER_REGISTRY.keys())}"
            )
            raise ValueError(f"Unknown splitter type: {splitter_type}")
        splitter = SPLITTER_REGISTRY[splitter_type].from_dict(data)
        log.debug(f"Created splitter: {type(splitter).__name__}")
        return splitter

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a dictionary representation of the splitter."""
        raise NotImplementedError


class ParagraphSplit(SplitStrategy):
    """Split text into paragraph text chunks by newlines."""

    REGEX = re.compile(r"\r?\n\s*\r?\n")

    def split(self, text: str) -> List[str]:
        log.debug(f"Splitting text with ParagraphSplit (length: {len(text)})")
        chunks = [p.strip() for p in self.REGEX.split(text) if p.strip()]
        log.debug(f"ParagraphSplit created {len(chunks)} chunks")
        return chunks

    @classmethod
    def from_dict(cls, data: dict) -> "ParagraphSplit":
        log.debug(f"Creating ParagraphSplit from dict: {data}")
        splitter = cls()
        log.debug("ParagraphSplit created")
        return splitter

    def to_dict(self) -> dict:
        return {"type": "ParagraphSplit"}


class FixedWindowSplit(SplitStrategy):
    """Split text into fixed-size windows of sentences."""

    def __init__(self, window: int = 5, stride: Optional[int] = None):
        log.debug(
            f"Initializing FixedWindowSplit with window={window}, stride={stride or window}"
        )
        self.window, self.stride = window, stride or window

    def split(self, text: str) -> List[str]:
        log.debug(
            f"Splitting text with FixedWindowSplit (length: {len(text)}, window={self.window}, stride={self.stride})"
        )
        sentences = re.split(r"(?<=[.!?])\s+", text)
        log.debug(f"FixedWindowSplit found {len(sentences)} sentences")
        chunks, i = [], 0
        while i < len(sentences):
            chunk = " ".join(sentences[i : i + self.window])
            chunks.append(chunk.strip())
            i += self.stride
        result = [c for c in chunks if c]
        log.debug(f"FixedWindowSplit created {len(result)} chunks")
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "FixedWindowSplit":
        log.debug(f"Creating FixedWindowSplit from dict: {data}")
        window = data.get("window", 5)
        stride = data.get("stride", None)
        splitter = cls(window=window, stride=stride)
        log.debug(
            f"FixedWindowSplit created with window={window}, stride={stride or window}"
        )
        return splitter

    def to_dict(self) -> dict:
        return {
            "type": "FixedWindowSplit",
            "window": self.window,
            "stride": self.stride,
        }


class RegexSplit(SplitStrategy):
    """Split text using a custom regex pattern."""

    def __init__(self, pattern: str):
        log.debug(f"Initializing RegexSplit with pattern: {pattern}")
        self.pattern = re.compile(pattern)
        self.pattern_str = pattern

    def split(self, text: str) -> List[str]:
        log.debug(
            f"Splitting text with RegexSplit (length: {len(text)}, pattern: {self.pattern_str})"
        )
        chunks = [p.strip() for p in self.pattern.split(text) if p.strip()]
        log.debug(f"RegexSplit created {len(chunks)} chunks")
        return chunks

    @classmethod
    def from_dict(cls, data: dict) -> "RegexSplit":
        log.debug(f"Creating RegexSplit from dict: {data}")
        if "pattern" not in data:
            log.error("RegexSplit config missing 'pattern' field")
            raise ValueError("RegexSplit config requires a 'pattern' field.")
        splitter = cls(data["pattern"])
        log.debug(f"RegexSplit created with pattern: {data['pattern']}")
        return splitter

    def to_dict(self) -> dict:
        return {"type": "RegexSplit", "pattern": self.pattern_str}


# Factory registry for splitter types
SPLITTER_REGISTRY: Dict[str, Type[SplitStrategy]] = {
    "ParagraphSplit": ParagraphSplit,
    "FixedWindowSplit": FixedWindowSplit,
    "RegexSplit": RegexSplit,
}
