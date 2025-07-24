from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import os

from delm.strategies import RelevanceScorer, KeywordScorer, FuzzyScorer
from delm.strategies import SplitStrategy, ParagraphSplit, FixedWindowSplit, RegexSplit
from delm.constants import (
    DEFAULT_MODEL_NAME, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS, DEFAULT_BASE_DELAY, DEFAULT_DOTENV_PATH, DEFAULT_REGEX_FALLBACK_PATTERN,
    DEFAULT_TARGET_COLUMN, DEFAULT_DROP_TARGET_COLUMN, DEFAULT_SCHEMA_CONTAINER,
    DEFAULT_PROMPT_TEMPLATE, DEFAULT_EXPERIMENT_DIR,
    DEFAULT_OVERWRITE_EXPERIMENT, DEFAULT_EXTRACT_TO_DATAFRAME, DEFAULT_TRACK_COST, DEFAULT_PANDAS_SCORE_FILTER,
    DEFAULT_AUTO_CHECKPOINT_AND_RESUME, DEFAULT_SYSTEM_PROMPT
)
from delm.exceptions import ConfigurationError

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
class LLMExtractionConfig:
    """Configuration for the LLM extraction process."""
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

    def to_dict(self) -> dict:
        return self.strategy.to_dict() if self.strategy else {"type": "None"}

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

    def to_dict(self) -> dict:
        return self.scorer.to_dict() if self.scorer else {"type": "None"}

@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing pipeline."""
    target_column: str = DEFAULT_TARGET_COLUMN
    drop_target_column: bool = DEFAULT_DROP_TARGET_COLUMN
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pandas_score_filter: Optional[str] = DEFAULT_PANDAS_SCORE_FILTER
    preprocessed_data_path: Optional[str] = None  # If set, use this preprocessed data file and skip other data config validation
    _explicitly_set_fields: set = field(default_factory=set, init=False)  # Track which fields were explicitly set

    def __post_init__(self):
        # Case when no scoring is specified, we don't want to drop the target column as the target column is the text_chunk? TODO: This needs to be looked into more to verify what exactly happens
        if self.scoring.scorer is None:
            self.drop_target_column = False

    def validate(self):
        if self.preprocessed_data_path:
            # If using external preprocessed data, skip other validation
            # Verify that the file is a feather file and has the correct columns
            if not self.preprocessed_data_path.endswith(".feather"):
                raise ConfigurationError(
                    "preprocessed_data_path must be a feather file.",
                    {"preprocessed_data_path": self.preprocessed_data_path, "suggestion": "Provide a valid feather file path"}
                )
            # Check for conflicting fields - only consider explicitly set fields
            conflicting = []
            if "target_column" in self._explicitly_set_fields:
                conflicting.append("target_column")
            if "drop_target_column" in self._explicitly_set_fields:
                conflicting.append("drop_target_column")
            if "pandas_score_filter" in self._explicitly_set_fields:
                conflicting.append("pandas_score_filter")
            if self.splitting.strategy is not None:
                conflicting.append("splitting")
            if self.scoring.scorer is not None:
                conflicting.append("scoring")
            if conflicting:
                raise ConfigurationError(
                    f"Cannot specify {', '.join(conflicting)} when preprocessed_data_path is set.",
                    {"preprocessed_data_path": self.preprocessed_data_path, "conflicting_fields": conflicting, "suggestion": "Remove other data fields when using preprocessed_data_path. Or do not specify preprocessed_data_path if you want to specify the other data fields."}
                )
            # Verify that the file has the correct columns
            import pandas as pd
            df = pd.read_feather(self.preprocessed_data_path)
            from .constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN
            if not all(col in df.columns for col in [SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN]):
                raise ConfigurationError(
                    "preprocessed_data_path must have the correct columns.",
                    {"preprocessed_data_path": self.preprocessed_data_path, "suggestion": "Provide a valid feather file path with the correct columns"}
                )
            return
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
            if not isinstance(self.pandas_score_filter, str):
                raise ConfigurationError(
                    "pandas_score_filter must be a string or None.",
                    {"pandas_score_filter": self.pandas_score_filter, "suggestion": "Provide a valid pandas query string or None"}
                )
            # Create a dummy dataframe with the system default score column to check if the pandas_score_filter is valid
            import pandas as pd
            from .constants import SYSTEM_SCORE_COLUMN
            try:
                pd.DataFrame({SYSTEM_SCORE_COLUMN: [1]}).query(self.pandas_score_filter)
            except Exception as e:
                raise ConfigurationError(
                    f"pandas_score_filter is not a valid pandas query: {e}",
                    {"pandas_score_filter": self.pandas_score_filter, "suggestion": "Provide a valid pandas query string"}
                )
        self.splitting.validate()
        self.scoring.validate()

    def to_dict(self) -> dict:
        d = {}
        if self.preprocessed_data_path:
            d["preprocessed_data_path"] = self.preprocessed_data_path
        else:
            d["target_column"] = self.target_column
            d["drop_target_column"] = self.drop_target_column
            d["pandas_score_filter"] = self.pandas_score_filter
            d["splitting"] = self.splitting.to_dict()
            d["scoring"] = self.scoring.to_dict()
        return d

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            cfg = {}
        elif not isinstance(cfg, dict):
            raise ConfigurationError(
                f"Data preprocessing config must be a dictionary, got {type(cfg).__name__}",
                {"config_type": type(cfg).__name__, "suggestion": "Use dict format or omit data_preprocessing section entirely"}
            )
        splitting = SplittingConfig.from_config(cfg.get("splitting", {}))
        scoring = ScoringConfig.from_config(cfg.get("scoring", {}))
        
        # Track which fields were explicitly set
        explicitly_set_fields = set()
        if "target_column" in cfg:
            explicitly_set_fields.add("target_column")
        if "drop_target_column" in cfg:
            explicitly_set_fields.add("drop_target_column")
        if "pandas_score_filter" in cfg:
            explicitly_set_fields.add("pandas_score_filter")
        if "preprocessed_data_path" in cfg:
            explicitly_set_fields.add("preprocessed_data_path")

        instance = cls(
            target_column=cfg.get("target_column", DEFAULT_TARGET_COLUMN),
            drop_target_column=cfg.get("drop_target_column", DEFAULT_DROP_TARGET_COLUMN),
            splitting=splitting,
            scoring=scoring,
            pandas_score_filter=cfg.get("pandas_score_filter", DEFAULT_PANDAS_SCORE_FILTER),
            preprocessed_data_path=cfg.get("preprocessed_data_path", None),
        )
        instance._explicitly_set_fields = explicitly_set_fields
        return instance

@dataclass
class SchemaConfig:
    """Configuration for extraction schema."""
    spec_path: Path = Path("")
    container_name: str = DEFAULT_SCHEMA_CONTAINER
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT

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
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise ConfigurationError(
                "system_prompt must be a string or None.",
                {"system_prompt": self.system_prompt, "suggestion": "Provide a valid string or None for the system prompt."}
            )

@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    # This dataclass is now empty, but kept for future experiment-defining parameters if needed.
    pass

    def validate(self):
        pass

@dataclass
class DELMConfig:
    """Top-level configuration for DELM pipeline."""
    llm_extraction: LLMExtractionConfig
    data_preprocessing: DataPreprocessingConfig
    schema: SchemaConfig
    experiment: ExperimentConfig

    def __post_init__(self):
        self.validate()

    def validate(self):
        self.llm_extraction.validate()
        self.data_preprocessing.validate()
        self.schema.validate()
        self.experiment.validate()

    def to_config_dict(self) -> dict:
        """Return a dictionary suitable for saving as the experiment config YAML (excluding runtime/operational fields)."""
        # Only include experiment-defining fields
        return {
            "llm_extraction": {
                "provider": self.llm_extraction.provider,
                "name": self.llm_extraction.name,
                "temperature": self.llm_extraction.temperature,
                "max_retries": self.llm_extraction.max_retries,
                "batch_size": self.llm_extraction.batch_size,
                "max_workers": self.llm_extraction.max_workers,
                "base_delay": self.llm_extraction.base_delay,
                "dotenv_path": str(self.llm_extraction.dotenv_path) if self.llm_extraction.dotenv_path else None,
                "regex_fallback_pattern": self.llm_extraction.regex_fallback_pattern,
                "extract_to_dataframe": self.llm_extraction.extract_to_dataframe,
                "track_cost": self.llm_extraction.track_cost,
            },
            "data_preprocessing": {
                "target_column": self.data_preprocessing.target_column,
                "drop_target_column": self.data_preprocessing.drop_target_column,
                "pandas_score_filter": self.data_preprocessing.pandas_score_filter,
                "splitting": self.data_preprocessing.splitting.to_dict(),
                "scoring": self.data_preprocessing.scoring.to_dict(),
                "preprocessed_data_path": self.data_preprocessing.preprocessed_data_path,
            },
            "schema": {
                "spec_path": str(self.schema.spec_path),
                "container_name": self.schema.container_name,
                "prompt_template": self.schema.prompt_template,
                "system_prompt": self.schema.system_prompt,
            }
        }

    def to_schema_dict(self) -> dict:
        """Load and return the schema spec as a dictionary (from the path in self.schema.spec_path)."""
        import yaml
        import json
        path = self.schema.spec_path
        if not path.exists():
            raise FileNotFoundError(f"Schema spec file does not exist: {path}")
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text()) or {}
        elif path.suffix.lower() == ".json":
            return json.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported schema file format: {path.suffix}")


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
            # Handle LLM extraction config
            llm_extraction_data = data.get("llm_extraction", {})
            llm_extraction = LLMExtractionConfig(**llm_extraction_data)
            # Handle data preprocessing config
            data_preprocessing_data = data.get("data_preprocessing", {})
            data_preprocessing_cfg = DataPreprocessingConfig.from_config(data_preprocessing_data)
            # Handle schema config
            schema_data = data.get("schema", {})
            spec_path = schema_data.get("spec_path", "")
            if isinstance(spec_path, str):
                spec_path = Path(spec_path)
            schema = SchemaConfig(
                spec_path=spec_path,
                container_name=schema_data.get("container_name", DEFAULT_SCHEMA_CONTAINER),
                prompt_template=schema_data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
                system_prompt=schema_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            )
            # Handle experiment config
            experiment = ExperimentConfig()
            config = cls(llm_extraction=llm_extraction, data_preprocessing=data_preprocessing_cfg, schema=schema, experiment=experiment)
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load DELMConfig from dict: {e}", {"error": str(e)}) 

    @staticmethod
    def from_any(config_like) -> "DELMConfig":
        if isinstance(config_like, DELMConfig):
            return config_like
        elif isinstance(config_like, str):
            return DELMConfig.from_yaml(Path(config_like))
        elif isinstance(config_like, dict):
            return DELMConfig.from_dict(config_like)
        else:
            raise ValueError("config must be a DELMConfig, dict, or path to YAML.") 