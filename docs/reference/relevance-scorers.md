# Relevance Scorers

Score text chunks by relevance for filtering.

## Base Class

```python
from delm.strategies import RelevanceScorer

class CustomScorer(RelevanceScorer):
    def score(self, text_chunk: str) -> float:
        # Return 0.0-1.0 score
        return score
    
    def to_dict(self) -> dict:
        return {"type": "CustomScorer", ...}
    
    @classmethod
    def from_dict(cls, data: dict) -> "CustomScorer":
        return cls(...)
```

**Required for disk storage:** Implement `to_dict()` and `from_dict()`, then register:

```python
from delm.strategies import SCORER_REGISTRY
SCORER_REGISTRY["CustomScorer"] = CustomScorer
```

## Built-in Scorers

### KeywordScorer

Binary score (0.0 or 1.0) based on keyword presence.

```python
from delm import DELM

delm = DELM(
    schema=schema,
    relevance_scorer={
        "type": "keyword",
        "keywords": ["price", "revenue", "earnings"]
    },
    score_filter="delm_score > 0"  # Keep only matching chunks
)
```

**Parameters:**
- `keywords` (List[str]): Keywords to search for (case-insensitive)

**Score:**
- `1.0` if any keyword found
- `0.0` otherwise

---

### FuzzyScorer

Fuzzy matching score (0.0-1.0) using `rapidfuzz`.

```python
delm = DELM(
    schema=schema,
    relevance_scorer={
        "type": "fuzzy",
        "target_phrases": ["quarterly earnings report", "financial statement"]
    },
    score_filter="delm_score > 0.7"  # Keep high-similarity chunks
)
```

**Parameters:**
- `target_phrases` (List[str]): Phrases to fuzzy match against

**Score:** Max fuzzy match score (0.0-1.0) across all target phrases

**Requirements:** `pip install rapidfuzz`

## Filtering

Use `score_filter` with pandas query syntax:

```python
delm = DELM(
    schema=schema,
    relevance_scorer={"type": "keyword", "keywords": ["revenue"]},
    score_filter="delm_score > 0.5"  # Only process chunks with score > 0.5
)
```

**Valid filters:**
- `"delm_score > 0.5"`
- `"delm_score >= 0.8"`
- `"delm_score == 1.0"`

**Note:** `score_filter` requires a `relevance_scorer`.

## Class-based Definition

```python
from delm.strategies import KeywordScorer, FuzzyScorer

scorer = KeywordScorer(keywords=["price", "cost"])
# Or
scorer = FuzzyScorer(target_phrases=["quarterly earnings"])

delm = DELM(
    schema=schema,
    relevance_scorer=scorer,
    score_filter="delm_score > 0"
)
```

## Registry

Access all available scorers:

```python
from delm.strategies import SCORER_REGISTRY

print(SCORER_REGISTRY.keys())
# dict_keys(['keyword', 'fuzzy', ...])
```

