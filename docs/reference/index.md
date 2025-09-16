# API Reference

This section documents the DELM codebase directly from its Python docstrings. Each page
is generated with [mkdocstrings](https://mkdocstrings.github.io) so the documentation stays
in sync with the source. Use these pages to explore constructor arguments, return types,
and behaviour of the pipeline components.

## Reference Guide

- [Pipeline API](pipeline.md) – High-level orchestration class that coordinates the end-to-end extraction workflow.
- [Configuration Objects](config.md) – Typed configuration classes that validate pipeline settings.
- [Core Managers](managers.md) – Batching, experiment tracking, and schema helpers that power the pipeline internals.
- [Utility Modules](utilities.md) – Supporting helpers such as the cost tracker, concurrency primitives, and type utilities.

Because these pages are rendered from docstrings, improvements to inline documentation will
automatically appear the next time the docs build.
