# Text Processing

Learn how to configure text splitting, relevance scoring, and filtering to optimize your extraction pipeline.

## Text Splitting Strategies

DELM supports multiple strategies for splitting large documents into manageable chunks for LLM processing.

### Paragraph Split (Default)

Splits text at paragraph boundaries:

```yaml
data_preprocessing:
  splitting:
    type: "ParagraphSplit"
```

**Best for**: Most document types, maintains natural text boundaries

### Fixed Window Split

Splits text into fixed-size windows with optional overlap:

```yaml
data_preprocessing:
  splitting:
    type: "FixedWindowSplit"
    window: 5      # Number of sentences per chunk
    stride: 2       # Number of sentences to overlap
```

**Best for**: 
- Consistent chunk sizes
- Capturing context across boundaries
- Processing structured documents

### Regex Split

Splits text using custom regular expressions:

```yaml
data_preprocessing:
  splitting:
    type: "RegexSplit"
    pattern: "\n\n"  # Split on double newlines
```

**Best for**:
- Custom document formats
- Specific structural patterns
- Domain-specific splitting needs

### No Splitting

Process entire documents as single chunks:

```yaml
data_preprocessing:
  splitting:
    type: null
```

**Best for**:
- Short documents
- When document-level context is critical
- Simple extraction tasks

## Relevance Scoring

Filter chunks based on relevance to your extraction task using scoring strategies.

### Keyword Scorer

Scores chunks based on keyword presence:

```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance", "revenue"]
```

**How it works**:
- Counts keyword occurrences in each chunk
- Scores range from 0.0 (no keywords) to 1.0 (all keywords present)
- Higher scores indicate more relevant content

### Fuzzy Scorer

Scores chunks using fuzzy string matching:

```yaml
data_preprocessing:
  scoring:
    type: "FuzzyScorer"
    keywords: ["price", "forecast", "guidance", "revenue"]
```

**Requirements**: Install `rapidfuzz`:
```bash
pip install rapidfuzz
```

**How it works**:
- Uses fuzzy string matching to find similar terms
- Handles typos, variations, and partial matches
- More flexible than exact keyword matching

### No Scoring

Process all chunks without filtering:

```yaml
data_preprocessing:
  scoring:
    type: null
```

## Chunk Filtering

Filter chunks based on relevance scores to focus processing on the most relevant content.

### Score-Based Filtering

```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance"]
  pandas_score_filter: "delm_score >= 0.7"  # Only process chunks with score >= 0.7
```

### Filtering Strategies

#### Conservative Filtering
```yaml
pandas_score_filter: "delm_score >= 0.8"  # High threshold, fewer chunks
```

#### Moderate Filtering  
```yaml
pandas_score_filter: "delm_score >= 0.5"  # Medium threshold, balanced
```

#### Liberal Filtering
```yaml
pandas_score_filter: "delm_score >= 0.2"  # Low threshold, more chunks
```

#### No Filtering
```yaml
# Omit pandas_score_filter to process all chunks
```

## Advanced Configuration

### Custom Splitting Parameters

#### Fixed Window with Overlap
```yaml
data_preprocessing:
  splitting:
    type: "FixedWindowSplit"
    window: 10      # Larger chunks
    stride: 3        # More overlap for context
```

#### Regex with Custom Pattern
```yaml
data_preprocessing:
  splitting:
    type: "RegexSplit"
    pattern: "\\n\\n\\n"  # Split on triple newlines
```

### Custom Scoring Parameters

#### Extended Keyword List
```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: 
      - "price"
      - "cost"
      - "revenue"
      - "forecast"
      - "guidance"
      - "outlook"
      - "projection"
      - "estimate"
```

#### Fuzzy Scoring with Threshold
```yaml
data_preprocessing:
  scoring:
    type: "FuzzyScorer"
    keywords: ["price", "forecast", "guidance"]
  pandas_score_filter: "delm_score >= 0.6"  # Adjust threshold for fuzzy matching
```

## Performance Optimization

### Chunk Size Optimization

**Small chunks** (2-3 sentences):
- ✅ Faster processing per chunk
- ✅ More precise extraction
- ❌ May miss context across boundaries
- ❌ More API calls

**Large chunks** (5-10 sentences):
- ✅ Better context preservation
- ✅ Fewer API calls
- ❌ Slower processing per chunk
- ❌ May include irrelevant content

### Scoring Optimization

**High thresholds** (0.8+):
- ✅ Focus on most relevant content
- ✅ Lower processing costs
- ❌ May miss important information
- ❌ Requires careful keyword selection

**Low thresholds** (0.3-0.5):
- ✅ Captures more information
- ✅ Better recall
- ❌ Higher processing costs
- ❌ May include irrelevant content


## Advanced Usage

### Comparing Splitting Strategies

Test different text splitting approaches and compare their performance:

```python
# Test different splitting strategies
configs = [
    {"type": "ParagraphSplit"},
    {"type": "FixedWindowSplit", "window": 3, "stride": 1},
    {"type": "FixedWindowSplit", "window": 5, "stride": 2},
]

results = {}
for i, config in enumerate(configs):
    # Create pipeline with different splitting
    pipeline = DELM.from_yaml(
        config_path=f"config_split_{i}.yaml",
        experiment_name=f"split_test_{i}",
        experiment_directory=Path("experiments")
    )
    
    # Run extraction and performance evaluation
    pipeline.prep_data("test_data.csv")
    pipeline.process_via_llm()
    
    # Evaluate performance
    metrics, _ = estimate_performance(
        config=f"config_split_{i}.yaml",
        data_source="test_data.csv",
        expected_extraction_output_df=human_labeled_df,
        true_json_column="expected_json",
        matching_id_column="id"
    )
    
    results[f"config_{i}"] = metrics

# Compare results
for config_name, metrics in results.items():
    avg_f1 = sum(m.get('f1', 0) for m in metrics.values()) / len(metrics)
    print(f"{config_name}: Average F1 = {avg_f1:.3f}")
```

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.

## Next Steps

- [Batch Processing](batch-processing.md) - Optimize performance with batching and checkpointing
- [Cost Tracking](cost-tracking.md) - Monitor costs and budget limits
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
