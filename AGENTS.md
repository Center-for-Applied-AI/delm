# AGENTS.md

## Cursor Cloud specific instructions

DELM is a pure Python library (no web server, no Docker, no databases). See `README.md` for usage overview and `pyproject.toml` for dependency groups.

### Virtual environment

A `.venv` exists at the project root. Always activate it before running anything:

```
source .venv/bin/activate
```

### Key commands

| Task | Command |
|---|---|
| Install (dev) | `pip install -e ".[dev]"` |
| Unit tests | `pytest tests/unit/ -v --cov=delm` |
| Lint (format) | `black --check src/ tests/` |
| Lint (style) | `flake8 src/ --max-line-length=120` |
| Build docs | `mkdocs build` |
| Serve docs | `mkdocs serve` |

### Testing notes

- Unit tests (`tests/unit/`) run without any API key and use mocking. This is the CI test suite.
- Integration tests (`tests/calls_test/`, `tests/pdf_climate_test/`, etc.) require a real LLM API key and are marked `slow` — they are skipped by default via `addopts = "-m 'not slow'"` in `pyproject.toml`.
- `mkdocstrings[python]` is needed for docs but is not listed in `pyproject.toml [dev]` dependencies; the update script installs it alongside `mkdocs<2` to avoid the incompatible MkDocs 2.0.

### Gotchas

- `python3.12-venv` system package is required to create the `.venv` (already installed in the snapshot).
- MkDocs 2.0 is incompatible with mkdocs-material; pin `mkdocs<2` when installing.
- The `mypy` config in `pyproject.toml` has `python_version = "1.1.0"` (appears to be a copy-paste from the project version) which will cause mypy to fail. This is a pre-existing issue in the repo.
