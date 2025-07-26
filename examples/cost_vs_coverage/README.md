# Cost vs Coverage Keyword Filtering

Optimize extraction costs using keyword filtering to reduce LLM API calls while maintaining coverage.

## Usage

```bash
pip install delm scikit-learn matplotlib seaborn tqdm
python cost_vs_coverage.py
```

## Approach

1. **Baseline**: Process all text chunks without filtering
2. **Keyword Discovery**: Use TF-IDF to find relevant keywords
3. **Pareto Analysis**: Test keyword list sizes for optimal balance
4. **Results**: Compare recall and cost across configurations

## Output

- `cost_vs_coverage_results.csv`: Metrics for each configuration
- `cost_vs_coverage_results.png`: Pareto curve visualization

## Expected Results

- **Cost savings**: 60-80% reduction in API costs
- **Coverage**: 80-90% extraction coverage maintained
- **Optimal keywords**: 10-30 keywords typically optimal
