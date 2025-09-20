# DELM

DELM (Data Extraction with Language Models) is a Python toolkit for extracting structured data from unstructured text using language models. It provides a configurable pipeline with cost tracking, caching, and evaluation capabilities.

## Why DELM?

- **Schema-first extraction** – declare the structure you want, from simple key-value pairs to deeply nested objects, and let DELM handle prompting and validation.
- **Flexible ingestion** – process TXT, HTML, Markdown, DOCX, PDF, CSV, Excel, Parquet, and Feather sources with built-in preprocessing.
- **Provider agnostic** – switch between OpenAI, Anthropic, Google, Groq, Together AI, and Fireworks AI without changing your pipeline.
- **Production ready** – built-in caching, batching, checkpointing, and resume support keep long-running jobs manageable.
- **Built for observability** – monitor token usage and budget, review extraction logs, and evaluate accuracy with the bundled metrics utilities.

## Key Capabilities

### Configurable processing

Text splitting, relevance scoring, filtering, and extraction logic in one YAML

### Progressive Schema System

Start with simple fields and grow to nested schemas or multiple schemas per prompt. Validation rules and enums keep results clean.

### Cost management

Cost tracking, caching, budget limits

### Extensible Architecture

Add custom scorers, schema components, or post-processing hooks. DELM integrates into larger data workflows.

Use the guides below to install DELM, configure a pipeline, and design schemas for your project. For a full quick start and configuration examples, see the README.
