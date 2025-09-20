# Getting Started

Install DELM, connect a language-model provider, and run your first extraction pipeline.

## Installation

Install from PyPI:

```bash
pip install delm
```

Or install from source:

```bash
git clone https://github.com/Center-for-Applied-AI/delm.git
cd delm
pip install -e .
```

If you use the optional developer tooling (tests, linters, notebooks), install the `dev` extra:

```bash
pip install -e .[dev]
```

## Configure Environment Variables

Create an `.env` file (or export in your shell) with credentials for the LLM providers you use. A minimal configuration:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
TOGETHER_API_KEY=...
```

Replace the values with your credentials. DELM only loads providers that have available keys.

## Quick Start

Use the high-level `DELM` class to load a pipeline configuration and run a job:

```python
from pathlib import Path
from delm import DELM

pipeline = DELM.from_yaml(
    config_path="example.config.yaml",
    experiment_name="my_experiment",
    experiment_directory=Path("experiments"),
)

pipeline.prep_data("data/input.txt")
pipeline.process_via_llm()

results = pipeline.get_extraction_results()
cost_summary = pipeline.get_cost_summary()
```

### Project Layout

A typical project structure keeps inputs, configuration, and outputs separated:

```
project/
├── data/
│   └── input.txt
├── config/
│   └── pipeline.yaml
├── schema/
│   └── schema_spec.yaml
└── experiments/
    └── my_experiment/
```

- **Pipeline configuration** controls providers, preprocessing, and batching.
- **Schema specification** declares the fields you want to extract.
- **Experiments directory** stores run artifacts, logs, and summaries.

## Next Steps

1. Review [Pipeline Configuration](usage/pipeline-configuration.md) to understand every option in the YAML config.
2. Explore the [Schema Reference](schema/reference.md) for guidance on designing robust schemas.
3. Check the `examples/` directory for sample configs covering common extraction scenarios.
