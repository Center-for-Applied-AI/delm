# Two-Stage Processing

Separate preprocessing from extraction to optimize costs and iterate faster.

## When to Use

- **Expensive preprocessing**: PDFs, large documents, complex splitting/scoring
- **Multiple extraction configs**: Test different prompts, models, or schemas on the same preprocessed data
- **Iterative development**: Tune extraction parameters without re-running preprocessing

## Example 1: Basic Two-Stage Split

Split a single extraction into two steps for better control.

```python
from delm import DELM, Schema, ExtractionVariable

# Slow preprocessing (PDFs, complex scoring)
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    splitting_strategy={"type": "paragraph"},
    relevance_scorer={"type": "fuzzy", "target_phrases": ["quarterly earnings"]}
)

# Stage 1: Preprocess (slow - only run once)
delm.prep_data("data/pdfs/")

# Stage 2: Extract (can re-run with different prompts/configs)
results = delm.process_via_llm()
```

## Example 2: Shared Preprocessing Across Configs

Preprocess once, extract many times with different configurations.

```python
# Step 1: Preprocess with first config (saves to disk)
config1 = DELM(
    schema=Schema.simple([ExtractionVariable(name="price", data_type="number")]),
    provider="openai",
    model="gpt-4o-mini",
    splitting_strategy={"type": "paragraph"},
    relevance_scorer={"type": "keyword", "keywords": ["price", "cost"]},
    use_disk_storage=True,
    experiment_path="experiments/run1"
)
config1.prep_data("data/documents.csv")  # Preprocessing happens here
results1 = config1.process_via_llm()

# Step 2: Use preprocessed data with different config
config2 = DELM(
    schema=Schema.simple([ExtractionVariable(name="revenue", data_type="number")]),
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    use_disk_storage=True,
    experiment_path="experiments/run2"
)
# Point to the preprocessed data from run1 (skips preprocessing entirely)
results2 = config2.process_via_llm("experiments/run1/delm_data/preprocessed.feather")
```

## Best Practices

- Always use `use_disk_storage=True` when sharing preprocessed data
- Preprocessing is only expensive if you have PDFs, splitting, or scoring
- For simple CSV data with no preprocessing, just use `delm.extract()`
