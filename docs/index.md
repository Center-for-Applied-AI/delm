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

## Quick Start

Get up and running with DELM in minutes:

1. **[Getting Started](getting-started.md)** - Install DELM, create your first config and schema files, and run your first extraction
2. **[Cost Estimation Tutorial](tutorials/cost-estimation.md)** - Learn to estimate costs before running large extractions  
3. **[Performance Evaluation Tutorial](tutorials/performance-evaluation.md)** - Learn to measure extraction quality with precision, recall, and F1 metrics

## Configuration

Customize your extraction pipeline:

- **[Pipeline Configuration](configuration/pipeline-config.md)** - Complete reference for all configuration options
- **[Schema Design](configuration/schema-design.md)** - Advanced schema patterns, validation features, and examples

## Features

Explore DELM's production-ready capabilities:

- **[Caching](features/caching.md)** - Reduce costs with semantic caching
- **[Text Processing](features/text-processing.md)** - Advanced splitting and scoring strategies  
- **[Batch Processing](features/batch-processing.md)** - Optimize performance with batching and checkpointing
- **[Cost Tracking](features/cost-tracking.md)** - Monitor costs and budget limits
- **[Post-Processing](features/post-processing.md)** - Transform results into tabular format
- **[File Formats](features/file-formats.md)** - Supported input formats and requirements

## API Reference

Complete API documentation for developers:

- **[API Overview](reference/index.md)** - Browse all available APIs
- **[Pipeline API](reference/pipeline.md)** - High-level orchestration class
- **[Configuration Objects](reference/config.md)** - Typed configuration classes
- **[Core Managers](reference/managers.md)** - Internal pipeline components
- **[Utilities](reference/utilities.md)** - Supporting helper functions
