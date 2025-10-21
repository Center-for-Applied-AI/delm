# Performance Evaluation Tutorial

Learn how to measure extraction quality using precision, recall, and F1 metrics against human-labeled data.

## When to Use Performance Evaluation

Performance evaluation helps you:
- **Quality assurance**: Measure how well your extraction performs
- **Schema optimization**: Identify which fields are extracted accurately
- **Model comparison**: Compare different models or configurations
- **Production readiness**: Ensure your pipeline meets quality thresholds

## Preparing Human-Labeled Data

You need a dataset with expected extraction results to evaluate against. Your labeled data should include:

### Required Columns
- **ID column**: Unique identifier for matching records
- **Text column**: The source text that was processed
- **Expected JSON column**: The correct extraction results in JSON format

### Example Labeled Data

```csv
id,text,expected_json
1,"Oil prices rose to $75 per barrel","{\"commodities\":[{\"commodity_type\":\"oil\",\"price_value\":75}]}"
2,"Gold reached $1950 per ounce","{\"commodities\":[{\"commodity_type\":\"gold\",\"price_value\":1950}]}"
3,"No commodity prices mentioned","{\"commodities\":[]}"
```

### JSON Format Requirements

Your expected JSON should match your schema structure:

```json
{
  "commodities": [
    {
      "commodity_type": "oil",
      "price_value": 75
    }
  ]
}
```

## Running Performance Evaluation

Use `estimate_performance` to evaluate your extraction pipeline:

```python
from delm.utils.performance_estimation import estimate_performance
import pandas as pd

# Load your human-labeled data
human_labeled_df = pd.read_csv("labeled_data.csv")

# Run performance evaluation
metrics, comparison_df = estimate_performance(
    config="config.yaml",
    data_source="test_data.csv",
    expected_extraction_output_df=human_labeled_df,
    true_json_column="expected_json",
    matching_id_column="id",
    record_sample_size=50  # Optional: limit sample size
)

# Display results
for field, metrics_dict in metrics.items():
    precision = metrics_dict.get("precision", 0)
    recall = metrics_dict.get("recall", 0)
    f1 = metrics_dict.get("f1", 0)
    print(f"{field:<30} Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
```

### Parameters Explained

- `config`: Your pipeline configuration file
- `data_source`: The source data that was processed
- `expected_extraction_output_df`: DataFrame with human-labeled results
- `true_json_column`: Column name containing expected JSON results
- `matching_id_column`: Column name for matching records between datasets
- `record_sample_size`: Number of records to evaluate (optional, -1 for all)

## Interpreting Results

### Understanding Metrics

**Precision**: Of the items extracted, how many were correct?
- High precision = Few false positives
- Formula: `True Positives / (True Positives + False Positives)`

**Recall**: Of the correct items, how many were extracted?
- High recall = Few false negatives  
- Formula: `True Positives / (True Positives + False Negatives)`

**F1 Score**: Harmonic mean of precision and recall
- Balanced measure of overall performance
- Formula: `2 * (Precision * Recall) / (Precision + Recall)`

### Example Output

```
commodity_type              Precision: 0.950  Recall: 0.900  F1: 0.924
price_value                 Precision: 0.875  Recall: 0.933  F1: 0.903
```

### Interpreting Scores

| Score Range | Quality Level | Interpretation |
|-------------|---------------|----------------|
| 0.9 - 1.0   | Excellent     | Production ready |
| 0.8 - 0.9   | Good          | Minor improvements needed |
| 0.7 - 0.8   | Fair          | Significant improvements needed |
| 0.6 - 0.7   | Poor          | Major schema or model changes needed |
| < 0.6       | Very Poor     | Complete redesign recommended |

## Detailed Analysis

### Field-Level Performance

The `comparison_df` contains detailed results for each record:

```python
# Examine specific cases
print(comparison_df[['id', 'expected_dict', 'extracted_dict']].head())

# Find cases where extraction failed
failed_cases = comparison_df[
    comparison_df['expected_dict'] != comparison_df['extracted_dict']
]
print(f"Failed extractions: {len(failed_cases)}")
```

### Common Issues and Solutions

#### Low Precision (Many False Positives)

**Problem**: Extracting items that shouldn't be extracted

**Solutions**:
1. **Improve schema validation**:
```yaml
variables:
  - name: "commodity_type"
    validate_in_text: true  # Only extract if explicitly mentioned
    allowed_values: ["oil", "gas", "gold", "silver"]
```

2. **Add filtering criteria**:
```yaml
data_preprocessing:
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "cost", "rate"]
  pandas_score_filter: "delm_score >= 0.8"
```

3. **Refine field descriptions**:
```yaml
- name: "commodity_type"
  description: "Type of commodity explicitly mentioned with price information"
```

#### Low Recall (Many False Negatives)

**Problem**: Missing items that should be extracted

**Solutions**:
1. **Improve prompt clarity**:
```yaml
schema:
  prompt_template: |
    Extract ALL commodity information mentioned in the text.
    Look for any price, cost, or rate information.
    
    {variables}
    
    Text: {text}
```

2. **Adjust text splitting**:
```yaml
data_preprocessing:
  splitting:
    type: "FixedWindowSplit"
    window: 5  # Larger chunks capture more context
    stride: 2  # Overlap to avoid missing information
```

3. **Use more specific descriptions**:
```yaml
- name: "price_value"
  description: "Any numeric price, cost, or rate value mentioned"
```

#### Low F1 Score (Both Precision and Recall Issues)

**Problem**: Both false positives and false negatives

**Solutions**:
1. **Review your schema design**:
   - Ensure field descriptions are clear and specific
   - Use appropriate data types
   - Set reasonable validation rules

2. **Test different models**:
```yaml
llm_extraction:
  provider: "anthropic"
  name: "claude-3-sonnet"  # Try different models
```

3. **Optimize preprocessing**:
```yaml
data_preprocessing:
  splitting:
    type: "ParagraphSplit"  # Try different splitting strategies
  scoring:
    type: "FuzzyScorer"    # Try fuzzy matching
    keywords: ["price", "cost", "rate", "value"]
```

## Best Practices

### 1. Create High-Quality Labeled Data

- **Consistent labeling**: Use the same criteria across all records
- **Complete coverage**: Include both positive and negative examples
- **Edge cases**: Include challenging or ambiguous cases
- **Multiple annotators**: Have different people label the same data to check consistency

### 2. Use Representative Samples

- **Size**: Use at least 100-500 records for reliable metrics
- **Diversity**: Include various text types and complexity levels
- **Distribution**: Match the distribution of your production data

### 3. Iterative Improvement

1. **Baseline**: Run initial evaluation
2. **Identify issues**: Look at failed cases and low-scoring fields
3. **Make changes**: Adjust schema, prompts, or preprocessing
4. **Re-evaluate**: Test changes on the same labeled data
5. **Repeat**: Continue until you reach acceptable performance

### 4. Set Performance Thresholds

Define minimum acceptable scores for production:

```python
# Example quality gates
MIN_F1_SCORE = 0.8
MIN_PRECISION = 0.75
MIN_RECALL = 0.75

for field, metrics in metrics.items():
    f1 = metrics.get("f1", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    
    if f1 < MIN_F1_SCORE:
        print(f"WARNING: {field} F1 score {f1:.3f} below threshold {MIN_F1_SCORE}")
```

## Troubleshooting

If you encounter issues, review the logs in your experiment directory for detailed error information.

## Next Steps

- [Cost Estimation Tutorial](cost-estimation.md) - Learn to estimate costs before running extractions
- [Schema Design](../configuration/schema-design.md) - Advanced schema patterns and validation
- [Pipeline Configuration](../configuration/pipeline-config.md) - Complete configuration reference
- [Text Processing](../features/text-processing.md) - Advanced preprocessing strategies
