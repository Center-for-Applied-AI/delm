# Splitting Strategies

Text chunking strategies for preprocessing.

## Base Class

```python
from delm.strategies import SplitStrategy

class CustomSplitter(SplitStrategy):
    def split(self, text: str) -> List[str]:
        # Return list of text chunks
        return chunks
    
    def to_dict(self) -> dict:
        return {"type": "CustomSplitter", ...}
    
    @classmethod
    def from_dict(cls, data: dict) -> "CustomSplitter":
        return cls(...)
```

**Required for disk storage:** Implement `to_dict()` and `from_dict()`, then register:

```python
from delm.strategies import SPLITTER_REGISTRY
SPLITTER_REGISTRY["CustomSplitter"] = CustomSplitter
```

## Built-in Strategies

### ParagraphSplit

Split by double newlines (paragraphs).

```python
from delm import DELM

delm = DELM(
    schema=schema,
    splitting_strategy={"type": "paragraph"}
)
```

**Output:** One chunk per paragraph.

---

### FixedWindowSplit

Split into sliding windows of N sentences.

```python
delm = DELM(
    schema=schema,
    splitting_strategy={
        "type": "fixed_window",
        "window": 5,    # 5 sentences per chunk
        "stride": 3     # Move 3 sentences (overlap = window - stride)
    }
)
```

**Parameters:**
- `window` (int): Number of sentences per chunk
- `stride` (int, optional): Step size (default = window, no overlap)

**Example:**
```
Text: S1. S2. S3. S4. S5. S6. S7.
window=3, stride=2:
  Chunk 1: S1. S2. S3.
  Chunk 2: S3. S4. S5.
  Chunk 3: S5. S6. S7.
```

---

### RegexSplit

Split by custom regex pattern.

```python
delm = DELM(
    schema=schema,
    splitting_strategy={
        "type": "regex",
        "pattern": r"\n\s*---\s*\n"  # Split by "---" separator
    }
)
```

**Parameters:**
- `pattern` (str): Regex pattern to split on

## Class-based Definition

```python
from delm.strategies import ParagraphSplit, FixedWindowSplit

splitter = ParagraphSplit()
# Or
splitter = FixedWindowSplit(window=5, stride=3)

delm = DELM(
    schema=schema,
    splitting_strategy=splitter
)
```

## Registry

Access all available splitters:

```python
from delm.strategies import SPLITTER_REGISTRY

print(SPLITTER_REGISTRY.keys())
# dict_keys(['paragraph', 'fixed_window', 'regex', ...])
```

