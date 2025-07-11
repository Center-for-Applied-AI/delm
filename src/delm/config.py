from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
import os

from .scoring_strategies import RelevanceScorer, KeywordScorer, FuzzyScorer
from .splitting_strategies import SplitStrategy, ParagraphSplit, FixedWindowSplit, RegexSplit
from .constants import (
    DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS, DEFAULT_BASE_DELAY, DEFAULT_DOTENV_PATH, DEFAULT_REGEX_FALLBACK_PATTERN,
    DEFAULT_TARGET_COLUMN, DEFAULT_DROP_TARGET_COLUMN, DEFAULT_SCHEMA_CONTAINER,
    DEFAULT_PROMPT_TEMPLATE, DEFAULT_EXPERIMENT_DIR, DEFAULT_SAVE_INTERMEDIATES,
    DEFAULT_OVERWRITE_EXPERIMENT, DEFAULT_VERBOSE, DEFAULT_KEYWORDS
)

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

def scorer_from_config(cfg):
    if isinstance(cfg, RelevanceScorer):
        return cfg
    if isinstance(cfg, dict):
        scorer_type = cfg.get("type", "KeywordScorer")
        if scorer_type == "KeywordScorer":
            return KeywordScorer(cfg.get("keywords", []))
        elif scorer_type == "FuzzyScorer":
            return FuzzyScorer(cfg.get("keywords", []))
        else:
            raise ValueError(f"Unknown scorer type: {scorer_type}")
    if isinstance(cfg, str):
        if cfg == "KeywordScorer":
            return KeywordScorer([])
        elif cfg == "FuzzyScorer":
            return FuzzyScorer([])
        else:
            raise ValueError(f"Unknown scorer type: {cfg}")
    return KeywordScorer([])

def splitter_from_config(cfg):
    if isinstance(cfg, SplitStrategy):
        return cfg
    if isinstance(cfg, dict):
        split_type = cfg.get("type", "ParagraphSplit")
        if split_type == "ParagraphSplit":
            return ParagraphSplit()
        elif split_type == "FixedWindowSplit":
            return FixedWindowSplit(cfg.get("window", 5), cfg.get("stride", 5))
        elif split_type == "RegexSplit":
            return RegexSplit(cfg.get("pattern", "\n\n"))
        else:
            raise ValueError(f"Unknown split strategy: {split_type}")
    if isinstance(cfg, str):
        if cfg == "ParagraphSplit":
            return ParagraphSplit()
        elif cfg == "FixedWindowSplit":
            return FixedWindowSplit()
        elif cfg == "RegexSplit":
            return RegexSplit("\n\n")
        else:
            raise ValueError(f"Unknown split strategy: {cfg}")
    return ParagraphSplit()

@dataclass
class ModelConfig:
    """Configuration for the LLM model and API usage."""
    name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    batch_size: int = DEFAULT_BATCH_SIZE
    max_workers: int = DEFAULT_MAX_WORKERS
    base_delay: float = DEFAULT_BASE_DELAY
    dotenv_path: Optional[Union[str, Path]] = DEFAULT_DOTENV_PATH
    regex_fallback_pattern: Optional[str] = DEFAULT_REGEX_FALLBACK_PATTERN

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ConfigValidationError("Model name must be a non-empty string.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigValidationError("Temperature must be between 0.0 and 2.0.")
        if self.max_retries < 0:
            raise ConfigValidationError("max_retries must be non-negative.")
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive.")
        if self.max_workers <= 0:
            raise ConfigValidationError("max_workers must be positive.")
        if self.base_delay < 0:
            raise ConfigValidationError("base_delay must be non-negative.")
        if self.dotenv_path is not None and not Path(self.dotenv_path).exists():
            raise ConfigValidationError(f"dotenv_path does not exist: {self.dotenv_path}")
        if self.regex_fallback_pattern is not None and not isinstance(self.regex_fallback_pattern, str):
            raise ConfigValidationError("regex_fallback_pattern must be a string or None.")

@dataclass
class SplittingConfig:
    """Configuration for text splitting strategy."""
    strategy: SplitStrategy = field(default_factory=ParagraphSplit)

    def validate(self):
        if not isinstance(self.strategy, SplitStrategy):
            raise ConfigValidationError("strategy must be a SplitStrategy instance.")

    @classmethod
    def from_config(cls, cfg):
        return cls(strategy=splitter_from_config(cfg.get("strategy", {})))

@dataclass
class ScoringConfig:
    """Configuration for relevance scoring strategy."""
    scorer: RelevanceScorer = field(default_factory=lambda: KeywordScorer(DEFAULT_KEYWORDS))

    def validate(self):
        if not isinstance(self.scorer, RelevanceScorer):
            raise ConfigValidationError("scorer must be a RelevanceScorer instance.")

    @classmethod
    def from_config(cls, cfg):
        return cls(scorer=scorer_from_config(cfg.get("scorer", {})))

@dataclass
class DataConfig:
    """Configuration for data processing."""
    target_column: str = DEFAULT_TARGET_COLUMN
    drop_target_column: bool = DEFAULT_DROP_TARGET_COLUMN
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def validate(self):
        if not isinstance(self.target_column, str) or not self.target_column:
            raise ConfigValidationError("target_column must be a non-empty string.")
        if not isinstance(self.drop_target_column, bool):
            raise ConfigValidationError("drop_target_column must be a boolean.")
        self.splitting.validate()
        self.scoring.validate()

@dataclass
class SchemaConfig:
    """Configuration for extraction schema."""
    spec_path: Path = Path("")
    container_name: str = DEFAULT_SCHEMA_CONTAINER
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE

    def validate(self):
        if not isinstance(self.spec_path, Path) or not str(self.spec_path):
            raise ConfigValidationError("spec_path must be a valid Path.")
        if not self.spec_path.exists():
            raise ConfigValidationError(f"Schema spec file does not exist: {self.spec_path}")
        if not isinstance(self.container_name, str) or not self.container_name:
            raise ConfigValidationError("container_name must be a non-empty string.")
        if self.prompt_template is not None and not isinstance(self.prompt_template, str):
            raise ConfigValidationError("prompt_template must be a string or None.")

@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    name: str = ""
    directory: Path = DEFAULT_EXPERIMENT_DIR
    save_intermediates: bool = DEFAULT_SAVE_INTERMEDIATES
    overwrite_experiment: bool = DEFAULT_OVERWRITE_EXPERIMENT
    verbose: bool = DEFAULT_VERBOSE

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ConfigValidationError("Experiment name must be a non-empty string.")
        if not isinstance(self.directory, Path):
            raise ConfigValidationError("directory must be a Path object.")
        if not isinstance(self.save_intermediates, bool):
            raise ConfigValidationError("save_intermediates must be a boolean.")
        if not isinstance(self.overwrite_experiment, bool):
            raise ConfigValidationError("overwrite_experiment must be a boolean.")
        if not isinstance(self.verbose, bool):
            raise ConfigValidationError("verbose must be a boolean.")

@dataclass
class DELMConfig:
    """Top-level configuration for DELM pipeline."""
    model: ModelConfig
    data: DataConfig
    schema: SchemaConfig
    experiment: ExperimentConfig

    def validate(self):
        self.model.validate()
        self.data.validate()
        self.schema.validate()
        self.experiment.validate()



    @classmethod
    def from_yaml(cls, path: Path) -> "DELMConfig":
        if not path.exists():
            raise ConfigValidationError(f"YAML config file does not exist: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DELMConfig":
        try:
            # Handle model config
            model_data = data.get("model", {})
            model = ModelConfig(**model_data)
            
            # Handle splitting config
            splitting_data = data.get("data", {}).get("splitting", {})
            splitting = SplittingConfig.from_config(splitting_data)
            
            # Handle scoring config
            scoring_data = data.get("data", {}).get("scoring", {})
            scoring = ScoringConfig.from_config(scoring_data)
            
            # Handle data config
            data_config_data = data.get("data", {})
            
            data_cfg = DataConfig(
                target_column=data_config_data.get("target_column", DEFAULT_TARGET_COLUMN),
                drop_target_column=data_config_data.get("drop_target_column", DEFAULT_DROP_TARGET_COLUMN),
                splitting=splitting,
                scoring=scoring,
            )
            
            # Handle schema config
            schema_data = data.get("schema", {})
            spec_path = schema_data.get("spec_path", "")
            if isinstance(spec_path, str):
                spec_path = Path(spec_path)
            
            schema = SchemaConfig(
                spec_path=spec_path,
                container_name=schema_data.get("container_name", DEFAULT_SCHEMA_CONTAINER),
                prompt_template=schema_data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
            )
            
            # Handle experiment config
            experiment_data = data.get("experiment", {})
            directory = experiment_data.get("directory", DEFAULT_EXPERIMENT_DIR)
            if isinstance(directory, str):
                directory = Path(directory)
            
            experiment = ExperimentConfig(
                name=experiment_data.get("name", ""),
                directory=directory,
                save_intermediates=experiment_data.get("save_intermediates", DEFAULT_SAVE_INTERMEDIATES),
                overwrite_experiment=experiment_data.get("overwrite_experiment", DEFAULT_OVERWRITE_EXPERIMENT),
                verbose=experiment_data.get("verbose", DEFAULT_VERBOSE)
            )
            
            config = cls(model=model, data=data_cfg, schema=schema, experiment=experiment)
            
            return config
        except Exception as e:
            raise ConfigValidationError(f"Failed to load DELMConfig from dict: {e}")

    @classmethod
    def from_env(cls) -> "DELMConfig":
        # Example: minimal env loader, expects all fields as env vars (not recommended for complex configs)
        try:
            model = ModelConfig(
                name=os.getenv("DELM_MODEL_NAME", DEFAULT_MODEL_NAME),
                temperature=float(os.getenv("DELM_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
                max_retries=int(os.getenv("DELM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
                batch_size=int(os.getenv("DELM_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
                max_workers=int(os.getenv("DELM_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
                base_delay=float(os.getenv("DELM_BASE_DELAY", str(DEFAULT_BASE_DELAY))),
            )
            # DataConfig and SchemaConfig from env is not robust, so recommend using from_yaml or from_dict
            raise NotImplementedError("from_env is not fully implemented. Use from_yaml or from_dict instead.")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load DELMConfig from environment: {e}") 