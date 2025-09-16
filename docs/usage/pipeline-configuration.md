# Pipeline Configuration

Every DELM run is driven by a pipeline configuration file (YAML) that describes how to prepare documents, which model to call, and how to manage costs. The configuration keeps extraction logic declarative so you can version and review changes alongside your code.

## Minimal Configuration

```yaml
llm_extraction:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.0
  batch_size: 10
  track_cost: true
  max_budget: 50.0

data_preprocessing:
  target_column: "text"
  splitting:
    type: "ParagraphSplit"
  scoring:
    type: "KeywordScorer"
    keywords: ["price", "forecast", "guidance"]

schema:
  spec_path: "schema_spec.yaml"
```

## Key Sections

### `llm_extraction`

Controls which provider and model to use, as well as sampling and batching behavior.

- `provider`: Any supported provider slug (`openai`, `anthropic`, `google`, `groq`, `together`, `fireworks`).
- `name`: Model identifier as expected by the provider.
- `temperature`: Sampling temperature; keep at `0.0` for deterministic extraction.
- `batch_size`: Number of chunks processed concurrently. Tune based on provider rate limits.
- `track_cost`: Enable token and currency tracking.
- `max_budget`: Optional soft ceiling; processing stops when the budget is exceeded.

### `data_preprocessing`

Defines how input documents are transformed before sending prompts.

- `target_column`: DataFrame column used as the primary text input.
- `splitting`: Strategy for splitting large documents (`ParagraphSplit`, `SentenceSplit`, custom classes).
- `scoring`: Optional filter to select relevant chunks (e.g., `KeywordScorer`).
- Additional preprocessing components such as cleaning or enrichment can be added as nested blocks.

### `schema`

Points to the schema specification file and optional prompt customization.

- `spec_path`: Path to a `schema_spec.yaml` file that describes the extraction schema.
- `system_prompt`: Override the default system prompt for all runs.
- `prompt_template`: Format string used to render each prompt with `{variables}`, `{text}`, and optional `{context}` placeholders.

## Experiment Management

When you instantiate `DELM.from_yaml`, you pass an `experiment_name` and `experiment_directory`. DELM creates a subdirectory that stores:

- Input checkpoints and chunk metadata.
- Cached responses and retry logs.
- Cost summaries (`cost_summary.json`).
- Final extraction outputs.

Keeping experiments isolated makes it easy to resume failed runs or compare different configurations.

## Cost Tracking

If `track_cost` is enabled, the pipeline records token usage and provider fees. After a run, call:

```python
summary = pipeline.get_cost_summary()
print(summary["total_cost"])
```

The summary includes per-provider totals, cached token counts, and budget status. Use this data to optimize prompts before running at production scale.

## Config Best Practices

- Commit configuration files to version control and review changes like any other code.
- Start with smaller batch sizes until you confirm provider rate limits.
- Specify `max_budget` when iterating on new schemas to avoid unexpected spending.
- Use descriptive experiment names (e.g., include dataset and model) to keep folders organized.
