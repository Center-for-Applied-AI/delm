# DELM

DELM (Data Extraction with Language Models) is a Python toolkit for turning unstructured documents into structured datasets with the help of large language models. The library provides a configurable pipeline that handles document ingestion, schema-driven prompting, cost tracking, and evaluation so that you can focus on defining the data you need.

## Why DELM?

- **Schema-first extraction** – declare the structure you want, from simple key-value pairs to deeply nested objects, and let DELM handle prompting and validation.
- **Flexible ingestion** – process TXT, HTML, Markdown, DOCX, PDF, CSV, Excel, Parquet, and Feather sources with built-in preprocessing.
- **Provider agnostic** – switch between OpenAI, Anthropic, Google, Groq, Together AI, and Fireworks AI without changing your pipeline.
- **Production ready** – built-in caching, batching, checkpointing, and resume support keep long-running jobs manageable.
- **Built for observability** – monitor token usage and budget, review extraction logs, and evaluate accuracy with the bundled metrics utilities.

## Key Capabilities

### Configurable Pipeline

Configure preprocessing, chunking, filtering, and extraction logic from a single YAML file. Override settings per-dataset without touching code.

### Progressive Schema System

Start with simple fields and evolve toward complex nested schemas or multiple schemas per prompt. Validation rules and enum restrictions keep results clean.

### Cost Awareness

Track token usage and costs per provider, set budgets, and export summaries for downstream reporting.

### Extensible Architecture

Augment the default pipeline with custom scorers, schema components, or post-processing hooks. The toolkit is designed to integrate into larger data workflows.

Continue with the guides below to install DELM, configure your pipeline, and design schemas that match your project.
