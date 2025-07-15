from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import os

from .strategies import RelevanceScorer, KeywordScorer, FuzzyScorer
from .strategies import SplitStrategy, ParagraphSplit, FixedWindowSplit, RegexSplit
from .constants import (
    DEFAULT_MODEL_NAME, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS, DEFAULT_BASE_DELAY, DEFAULT_DOTENV_PATH, DEFAULT_REGEX_FALLBACK_PATTERN,
    DEFAULT_TARGET_COLUMN, DEFAULT_DROP_TARGET_COLUMN, DEFAULT_SCHEMA_CONTAINER,
    DEFAULT_PROMPT_TEMPLATE, DEFAULT_EXPERIMENT_DIR,
    DEFAULT_OVERWRITE_EXPERIMENT, DEFAULT_VERBOSE, DEFAULT_EXTRACT_TO_DATAFRAME, DEFAULT_TRACK_COST, DEFAULT_PANDAS_SCORE_FILTER
)
from .exceptions import ConfigurationError

def _scorer_from_config(cfg):
    if isinstance(cfg, RelevanceScorer):
        return cfg
    if isinstance(cfg, dict):
        scorer_type = cfg.get("type", "None") 
        if scorer_type == "KeywordScorer":
            keywords = cfg.get("keywords", [])
            if not keywords:
                raise ConfigurationError(
                    "KeywordScorer requires a non-empty keywords list",
                    {"scorer_type": scorer_type, "suggestion": "Provide keywords list or use 'None' for no scoring"}
                )
            return KeywordScorer(keywords)
        elif scorer_type == "FuzzyScorer":
            keywords = cfg.get("keywords", [])
            if not keywords:
                raise ConfigurationError(
                    "FuzzyScorer requires a non-empty keywords list",
                    {"scorer_type": scorer_type, "suggestion": "Provide keywords list or use 'None' for no scoring"}
                )
            return FuzzyScorer(keywords)
        elif scorer_type == "None" or scorer_type is None:
            return None
        else:
            raise ConfigurationError(
                f"Unknown scorer type: {scorer_type}",
                {"scorer_type": scorer_type, "suggestion": "Use 'KeywordScorer', 'FuzzyScorer', or 'None'"}
            )
    # Default to no scoring for any other input
    return None

def _splitter_from_config(cfg):
    if isinstance(cfg, SplitStrategy):
        return cfg
    if isinstance(cfg, dict):
        split_type = cfg.get("type", "None")  # Default to None instead of ParagraphSplit
        if split_type == "ParagraphSplit":
            return ParagraphSplit()
        elif split_type == "FixedWindowSplit":
            return FixedWindowSplit(cfg.get("window", 5), cfg.get("stride", 5))
        elif split_type == "RegexSplit":
            return RegexSplit(cfg.get("pattern", "\n\n"))
        elif split_type == "None" or split_type is None:
            return None
        else:
            raise ConfigurationError(
                f"Unknown split strategy: {split_type}",
                {"split_type": split_type, "suggestion": "Use 'ParagraphSplit', 'FixedWindowSplit', 'RegexSplit', or 'None'"}
            )
    # Default to no splitting for any other input
    return None

@dataclass
class ModelConfig:
    """Configuration for the LLM model and API usage."""
    provider: str = DEFAULT_PROVIDER  # e.g., 'openai', 'anthropic', 'google', etc.
    name: str = DEFAULT_MODEL_NAME    # e.g., 'gpt-4o-mini', 'claude-3-sonnet', etc.
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    batch_size: int = DEFAULT_BATCH_SIZE
    max_workers: int = DEFAULT_MAX_WORKERS
    base_delay: float = DEFAULT_BASE_DELAY
    dotenv_path: Optional[Union[str, Path]] = DEFAULT_DOTENV_PATH
    regex_fallback_pattern: Optional[str] = DEFAULT_REGEX_FALLBACK_PATTERN
    extract_to_dataframe: bool = DEFAULT_EXTRACT_TO_DATAFRAME
    track_cost: bool = DEFAULT_TRACK_COST

    def get_provider_string(self) -> str:
        """Get the combined provider string for Instructor (e.g., 'openai/gpt-4o-mini')."""
        return f"{self.provider}/{self.name}"

    def validate(self):
        if not isinstance(self.provider, str) or not self.provider:
            raise ConfigurationError(
                "Provider must be a non-empty string.",
                {"provider": self.provider, "suggestion": "Use e.g. 'openai', 'anthropic', 'google', etc."}
            )
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError(
                "Model name must be a non-empty string.",
                {"name": self.name, "suggestion": "Use e.g. 'gpt-4o-mini', 'claude-3-sonnet', etc."}
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError(
                "Temperature must be between 0.0 and 2.0.",
                {"temperature": self.temperature, "suggestion": "Use a value between 0.0 and 2.0"}
            )
        if self.max_retries < 0:
            raise ConfigurationError(
                "max_retries must be non-negative.",
                {"max_retries": self.max_retries, "suggestion": "Use a non-negative integer"}
            )
        if self.batch_size <= 0:
            raise ConfigurationError(
                "batch_size must be positive.",
                {"batch_size": self.batch_size, "suggestion": "Use a positive integer"}
            )
        if self.max_workers <= 0:
            raise ConfigurationError(
                "max_workers must be positive.",
                {"max_workers": self.max_workers, "suggestion": "Use a positive integer"}
            )
        if self.base_delay < 0:
            raise ConfigurationError(
                "base_delay must be non-negative.",
                {"base_delay": self.base_delay, "suggestion": "Use a non-negative float"}
            )
        if self.dotenv_path is not None and not Path(self.dotenv_path).exists():
            raise ConfigurationError(
                f"dotenv_path does not exist: {self.dotenv_path}",
                {"dotenv_path": str(self.dotenv_path), "suggestion": "Check the file path or create the .env file"}
            )
        if self.regex_fallback_pattern is not None and not isinstance(self.regex_fallback_pattern, str):
            raise ConfigurationError(
                "regex_fallback_pattern must be a string or None.",
                {"regex_fallback_pattern": self.regex_fallback_pattern, "suggestion": "Provide a valid regex pattern or None"}
            )
        if not isinstance(self.extract_to_dataframe, bool):
            raise ConfigurationError(
                "extract_to_dataframe must be a boolean.",
                {"extract_to_dataframe": self.extract_to_dataframe, "suggestion": "Use True or False"}
            )
        if not isinstance(self.track_cost, bool):
            raise ConfigurationError(
                "track_cost must be a boolean.",
                {"track_cost": self.track_cost, "suggestion": "Use True or False"}
            )

@dataclass
class SplittingConfig:
    """Configuration for text splitting strategy."""
    strategy: Optional[SplitStrategy] = field(default=None)

    def validate(self):
        if self.strategy is not None and not isinstance(self.strategy, SplitStrategy):
            raise ConfigurationError(
                "strategy must be a SplitStrategy instance or None.",
                {"strategy_type": type(self.strategy).__name__, "suggestion": "Use a valid SplitStrategy subclass or None for no splitting"}
            )

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            cfg = {}
        elif not isinstance(cfg, dict):
            raise ConfigurationError(
                f"Splitting config must be a dictionary, got {type(cfg).__name__}",
                {"config_type": type(cfg).__name__, "suggestion": "Use dict format or omit splitting section entirely"}
            )
        return cls(strategy=_splitter_from_config(cfg))

@dataclass
class ScoringConfig:
    """Configuration for relevance scoring strategy."""
    scorer: Optional[RelevanceScorer] = field(default=None)

    def validate(self):
        if self.scorer is not None and not isinstance(self.scorer, RelevanceScorer):
            raise ConfigurationError(
                "scorer must be a RelevanceScorer instance or None.",
                {"scorer_type": type(self.scorer).__name__, "suggestion": "Use a valid RelevanceScorer subclass or None for no scoring"}
            )

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            cfg = {}
        elif not isinstance(cfg, dict):
            raise ConfigurationError(
                f"Scoring config must be a dictionary, got {type(cfg).__name__}",
                {"config_type": type(cfg).__name__, "suggestion": "Use dict format or omit scoring section entirely"}
            )
        return cls(scorer=_scorer_from_config(cfg))

@dataclass
class DataConfig:
    """Configuration for data processing."""
    target_column: str = DEFAULT_TARGET_COLUMN
    drop_target_column: bool = DEFAULT_DROP_TARGET_COLUMN
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pandas_score_filter: Optional[str] = DEFAULT_PANDAS_SCORE_FILTER

    def validate(self):
        if not isinstance(self.target_column, str) or not self.target_column:
            raise ConfigurationError(
                "target_column must be a non-empty string.",
                {"target_column": self.target_column, "suggestion": "Provide a valid column name"}
            )
        if not isinstance(self.drop_target_column, bool):
            raise ConfigurationError(
                "drop_target_column must be a boolean.",
                {"drop_target_column": self.drop_target_column, "suggestion": "Use True or False"}
            )
        if self.pandas_score_filter is not None:
            if self.pandas_score_filter is not isinstance(self.pandas_score_filter, str):
                raise ConfigurationError(
                    "pandas_score_filter must be a string or None.",
                    {"pandas_score_filter": self.pandas_score_filter, "suggestion": "Provide a valid pandas query string or None"}
                )
            # Create a dummy dataframe to check if the pandas_score_filter is valid
            import pandas as pd
            try:
                pd.DataFrame([1]).query(self.pandas_score_filter)
            except Exception as e:
                raise ConfigurationError(
                    f"pandas_score_filter is not a valid pandas query: {e}",
                    {"pandas_score_filter": self.pandas_score_filter, "suggestion": "Provide a valid pandas query string"}
                )
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
            raise ConfigurationError(
                "spec_path must be a valid Path.",
                {"spec_path": str(self.spec_path), "suggestion": "Provide a valid file path"}
            )
        if not self.spec_path.exists():
            raise ConfigurationError(
                f"Schema spec file does not exist: {self.spec_path}",
                {"spec_path": str(self.spec_path), "suggestion": "Check the file path or create the schema file"}
            )
        if not isinstance(self.container_name, str) or not self.container_name:
            raise ConfigurationError(
                "container_name must be a non-empty string.",
                {"container_name": self.container_name, "suggestion": "Provide a valid container name"}
            )
        if self.prompt_template is not None and not isinstance(self.prompt_template, str):
            raise ConfigurationError(
                "prompt_template must be a string or None.",
                {"prompt_template": self.prompt_template, "suggestion": "Provide a valid string or None"}
            )

@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    name: str = ""
    directory: Path = DEFAULT_EXPERIMENT_DIR
    overwrite_experiment: bool = DEFAULT_OVERWRITE_EXPERIMENT
    verbose: bool = DEFAULT_VERBOSE

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError(
                "Experiment name must be a non-empty string.",
                {"experiment_name": self.name, "suggestion": "Provide a valid experiment name"}
            )
        if not isinstance(self.directory, Path):
            raise ConfigurationError(
                "directory must be a Path object.",
                {"directory": str(self.directory), "suggestion": "Provide a valid Path object"}
            )
        if not isinstance(self.overwrite_experiment, bool):
            raise ConfigurationError(
                "overwrite_experiment must be a boolean.",
                {"overwrite_experiment": self.overwrite_experiment, "suggestion": "Use True or False"}
            )
        if not isinstance(self.verbose, bool):
            raise ConfigurationError(
                "verbose must be a boolean.",
                {"verbose": self.verbose, "suggestion": "Use True or False"}
            )

@dataclass
class DELMConfig:
    """Top-level configuration for DELM pipeline."""
    model: ModelConfig
    data: DataConfig
    schema: SchemaConfig
    experiment: ExperimentConfig

    def __post_init__(self):
        self.validate()

    def validate(self):
        self.model.validate()
        self.data.validate()
        self.schema.validate()
        self.experiment.validate()



    @classmethod
    def from_yaml(cls, path: Path) -> "DELMConfig":
        if not path.exists():
            raise ConfigurationError(
                f"YAML config file does not exist: {path}",
                {"file_path": str(path), "suggestion": "Check the file path or create the config file"}
            )
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML config file: {path}",
                {"file_path": str(path), "parse_error": str(e)}
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {path}", {"file_path": str(path)}) from e

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

                overwrite_experiment=experiment_data.get("overwrite_experiment", DEFAULT_OVERWRITE_EXPERIMENT),
                verbose=experiment_data.get("verbose", DEFAULT_VERBOSE)
            )
            
            config = cls(model=model, data=data_cfg, schema=schema, experiment=experiment)
            
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load DELMConfig from dict: {e}", {"error": str(e)}) 