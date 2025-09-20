# Core Managers

Modules that power DELM beyond the high-level pipeline: data preprocessing,
experiment storage, schema coordination, and batched extraction.

## Data Processor

::: delm.core.data_processor.DataProcessor
    options:
      show_source: false

## Experiment Managers

::: delm.core.experiment_manager.BaseExperimentManager
    options:
      show_source: false

::: delm.core.experiment_manager.DiskExperimentManager
    options:
      show_source: false

::: delm.core.experiment_manager.InMemoryExperimentManager
    options:
      show_source: false

## Extraction Manager

::: delm.core.extraction_manager.ExtractionManager
    options:
      show_source: false

## Schema Manager

::: delm.schemas.SchemaManager
    options:
      show_source: false
