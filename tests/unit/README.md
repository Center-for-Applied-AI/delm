# DELM Unit Tests

This directory contains comprehensive unit tests for the DELM library modules.

## Test Structure

The unit tests are organized into separate directories for each module:

### `data_loaders/`
Tests for the data loading functionality in `delm.strategies.data_loaders`
- Tests all loader classes (TextLoader, HtmlLoader, DocxLoader, etc.)
- Tests the DataLoaderFactory
- Tests file and directory loading
- Tests error handling and edge cases
- Includes test data files in `test_data/`

### `schemas/`
Tests for the schema system in `delm.schemas.schemas`
- Tests SimpleSchema, NestedSchema, and MultipleSchema classes
- Tests the SchemaRegistry
- Tests utility functions
- Tests validation and parsing logic
- Includes test schema configurations in `test_data/`

### `semantic_cache/`
Tests for the semantic caching system in `delm.utils.semantic_cache`
- Tests all cache backends (FilesystemJSONCache, SQLiteWALCache, LMDBCache)
- Tests the SemanticCacheFactory
- Tests utility functions
- Tests compression and metadata handling
- Tests thread safety

### `retry_handler/`
Tests for the retry handling system in `delm.utils.retry_handler`
- Tests RetryHandler class
- Tests exponential backoff logic
- Tests exception handling
- Tests logging behavior
- Tests various function types

### `concurrent_processing/`
Tests for the concurrent processing system in `delm.utils.concurrent_processing`
- Tests ConcurrentProcessor class
- Tests thread pool execution
- Tests order preservation
- Tests error handling
- Tests thread safety

## Running the Tests

### Run all unit tests:
```bash
pytest tests/unit/
```

### Run tests for a specific module:
```bash
pytest tests/unit/data_loaders/
pytest tests/unit/schemas/
pytest tests/unit/semantic_cache/
pytest tests/unit/retry_handler/
pytest tests/unit/concurrent_processing/
```

### Run with verbose output:
```bash
pytest tests/unit/ -v
```

### Run with coverage:
```bash
pytest tests/unit/ --cov=delm --cov-report=html
```

### Run specific test classes:
```bash
pytest tests/unit/data_loaders/test_data_loaders.py::TestTextLoader
pytest tests/unit/schemas/test_schemas.py::TestSimpleSchema
```

### Run specific test methods:
```bash
pytest tests/unit/data_loaders/test_data_loaders.py::TestTextLoader::test_load_text_file
```

## Test Data

Some tests include sample data files in `test_data/` subdirectories:
- CSV files for testing CSV loader
- Text files for testing text loader
- HTML files for testing HTML loader
- YAML schema configurations for testing schemas

## Test Coverage

The unit tests provide comprehensive coverage of:
- ✅ All public methods and classes
- ✅ Error handling and edge cases
- ✅ Different input types and configurations
- ✅ Thread safety where applicable
- ✅ Logging behavior
- ✅ Integration between components

## Dependencies

The tests use:
- `pytest` for test framework
- `unittest.mock` for mocking
- `pandas` for DataFrame testing
- `pydantic` for schema testing
- Standard library modules (threading, time, etc.)

## Notes

- Tests are designed to be independent and can run in any order
- Mock objects are used extensively to avoid external dependencies
- Temporary files and directories are used for file-based tests
- Thread safety is tested where relevant
- All tests include appropriate assertions and error checking 