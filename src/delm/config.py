"""Configuration objects for DELM.

Defines typed, serializable, and validatable configuration classes used across
the DELM pipeline: LLM extraction, splitting/scoring, schema, semantic cache,
and the top‑level ``DELMConfig`` aggregator.

Docstrings follow Google style.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar
import yaml

T = TypeVar('T', bound='BaseConfig')

from delm.strategies import RelevanceScorer, KeywordScorer, FuzzyScorer
from delm.strategies import SplitStrategy, ParagraphSplit, FixedWindowSplit, RegexSplit
from delm.constants import (
    # LLM/API Configuration
    DEFAULT_PROVIDER, 
    DEFAULT_MODEL_NAME, 
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_RETRIES, 
    DEFAULT_BASE_DELAY, 
    DEFAULT_BATCH_SIZE, 
    DEFAULT_MAX_WORKERS,
    DEFAULT_TRACK_COST, 
    DEFAULT_MAX_BUDGET, 
    DEFAULT_DOTENV_PATH,
    
    # Data Processing
    # Splitting
    DEFAULT_FIXED_WINDOW_SIZE,
    DEFAULT_FIXED_WINDOW_STRIDE,
    DEFAULT_REGEX_PATTERN,

    DEFAULT_DROP_TARGET_COLUMN, 
    DEFAULT_PANDAS_SCORE_FILTER, 
    
    # Schema Configuration
    DEFAULT_SCHEMA_PATH, 
    DEFAULT_PROMPT_TEMPLATE, 
    DEFAULT_SYSTEM_PROMPT,
    
    # Semantic Cache
    DEFAULT_SEMANTIC_CACHE_BACKEND, 
    DEFAULT_SEMANTIC_CACHE_PATH,
    DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB, 
    DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS,
    
    # System Constants
    SYSTEM_RAW_DATA_COLUMN,
    DEFAULT_FIXED_WINDOW_SIZE,
    DEFAULT_FIXED_WINDOW_STRIDE,
    DEFAULT_REGEX_PATTERN,
)


class BaseConfig:
    """Base class for configuration objects.

    Subclasses should implement ``validate`` and ``to_dict`` to provide strict
    validation and stable serialization.
    """
    
    def validate(self):
        """Validate configuration.

        Subclasses should raise ``ValueError`` when fields are invalid.
        """
        pass
    
    def to_dict(self) -> dict:
        """Convert configuration to a serializable dictionary."""
        return {}
    
    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """Create configuration instance from a dictionary."""
        return cls(**data)


@dataclass
class LLMExtractionConfig(BaseConfig):
    """Configuration for the LLM extraction process."""
    provider: str = DEFAULT_PROVIDER
    name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    batch_size: int = DEFAULT_BATCH_SIZE
    max_workers: int = DEFAULT_MAX_WORKERS
    base_delay: float = DEFAULT_BASE_DELAY
    dotenv_path: Optional[Union[str, Path]] = DEFAULT_DOTENV_PATH
    track_cost: bool = DEFAULT_TRACK_COST
    max_budget: Optional[float] = DEFAULT_MAX_BUDGET
    model_input_cost_per_1M_tokens: Optional[float] = None
    model_output_cost_per_1M_tokens: Optional[float] = None

    def get_provider_string(self) -> str:
        """Return the combined provider string for Instructor.

        Returns:
            Provider string in the form ``"<provider>/<model>"``.
        """
        return f"{self.provider}/{self.name}"

    def validate(self):
        """Validate all LLM extraction fields.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError(
                f"Provider must be a non-empty string. provider: {self.provider}, Suggestion: Use e.g. 'openai', 'anthropic', 'google', etc."
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Model name must be a non-empty string. name: {self.name}, Suggestion: Use e.g. 'gpt-4o-mini', 'claude-3-sonnet', etc."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0. temperature: {self.temperature}, Suggestion: Use a value between 0.0 and 2.0"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative. max_retries: {self.max_retries}, Suggestion: Use a non-negative integer"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive. batch_size: {self.batch_size}, Suggestion: Use a positive integer"
            )
        if self.max_workers <= 0:
            raise ValueError(
                f"max_workers must be positive. max_workers: {self.max_workers}, Suggestion: Use a positive integer"
            )
        if self.base_delay < 0:
            raise ValueError(
                f"base_delay must be non-negative. base_delay: {self.base_delay}, Suggestion: Use a non-negative float"
            )
        if self.dotenv_path is not None and not Path(self.dotenv_path).exists():
            raise ValueError(
                f"dotenv_path does not exist: {self.dotenv_path}, Suggestion: Check the file path or create the .env file"
            )
        if not isinstance(self.track_cost, bool):
            raise ValueError(
                f"track_cost must be a boolean. track_cost: {self.track_cost}, Suggestion: Use True or False"
            )
        if self.max_budget is not None:
            if not self.track_cost:
                raise ValueError(
                    f"track_cost must be True if max_budget is specified. track_cost: {self.track_cost}"
                )
            if not isinstance(self.max_budget, (int, float)):
                raise ValueError(
                    f"max_budget must be a number. max_budget: {self.max_budget}, Suggestion: Use a number"
                )

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "name": self.name,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "base_delay": self.base_delay,
            "dotenv_path": str(self.dotenv_path) if self.dotenv_path else None,
            "track_cost": self.track_cost,
            "max_budget": self.max_budget,
            "model_input_cost_per_1M_tokens": self.model_input_cost_per_1M_tokens,
            "model_output_cost_per_1M_tokens": self.model_output_cost_per_1M_tokens,
        }


@dataclass
class SplittingConfig(BaseConfig):
    """Configuration for text splitting strategy."""
    strategy: Optional[SplitStrategy] = field(default=None)

    def validate(self):
        """Validate the configured split strategy.

        Raises:
            ValueError: If ``strategy`` is provided but not a ``SplitStrategy``.
        """
        if self.strategy is not None and not isinstance(self.strategy, SplitStrategy):
            raise ValueError(
                f"strategy must be a SplitStrategy instance or None. strategy_type: {type(self.strategy).__name__}, Suggestion: Use a valid SplitStrategy subclass or None for no splitting"
            )

    def to_dict(self) -> dict:
        """Serialize the strategy configuration to a dictionary.

        Returns:
            A dictionary with the strategy configuration or ``{"type": "None"}``.
        """
        return self.strategy.to_dict() if self.strategy else {"type": "None"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplittingConfig":
        """Construct a ``SplittingConfig`` from a mapping.

        Args:
            data: Mapping with a ``type`` key and optional parameters.

        Returns:
            A configured ``SplittingConfig`` instance.
        """
        strategy = cls._create_strategy(data)
        return cls(strategy=strategy)

    @staticmethod
    def _create_strategy(cfg: Dict[str, Any]) -> Optional[SplitStrategy]:
        """Create a split strategy from a mapping.

        Args:
            cfg: Mapping with a ``type`` key and optional parameters.

        Returns:
            A ``SplitStrategy`` instance or ``None``.

        Raises:
            ValueError: If the ``type`` is unknown or invalid.
        """
        if cfg == {} or cfg is None:
            return None
        
        split_type = cfg.get("type", None)
        if split_type == "ParagraphSplit":
            return ParagraphSplit()
        elif split_type == "FixedWindowSplit":
            return FixedWindowSplit(cfg.get("window", DEFAULT_FIXED_WINDOW_SIZE), cfg.get("stride", DEFAULT_FIXED_WINDOW_STRIDE))
        elif split_type == "RegexSplit":
            return RegexSplit(cfg.get("pattern", DEFAULT_REGEX_PATTERN))
        elif split_type in ("None", None):
            return None
        else:
            raise ValueError(
                f"Unknown split strategy: {split_type}",
                {"split_type": split_type, "suggestion": "Use 'ParagraphSplit', 'FixedWindowSplit', 'RegexSplit', or 'None'"}
            )


@dataclass
class ScoringConfig(BaseConfig):
    """Configuration for relevance scoring strategy."""
    scorer: Optional[RelevanceScorer] = field(default=None)

    def validate(self):
        """Validate the configured scorer.

        Raises:
            ValueError: If ``scorer`` is provided but not a ``RelevanceScorer``.
        """
        if self.scorer is not None and not isinstance(self.scorer, RelevanceScorer):
            raise ValueError(
                f"scorer must be a RelevanceScorer instance or None. scorer_type: {type(self.scorer).__name__}, Suggestion: Use a valid RelevanceScorer subclass or None for no scoring"
            )

    def to_dict(self) -> dict:
        """Serialize the scoring configuration to a dictionary."""
        return self.scorer.to_dict() if self.scorer else {"type": "None"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringConfig":
        """Construct a ``ScoringConfig`` from a mapping.

        Args:
            data: Mapping with a ``type`` key and optional parameters.

        Returns:
            A configured ``ScoringConfig`` instance.
        """
        scorer = cls._create_scorer(data)
        return cls(scorer=scorer)

    @staticmethod
    def _create_scorer(cfg: Dict[str, Any]) -> Optional[RelevanceScorer]:
        """Create a scorer from a mapping.

        Args:
            cfg: Mapping with a ``type`` key and optional parameters.

        Returns:
            A ``RelevanceScorer`` instance or ``None``.

        Raises:
            ValueError: If the ``type`` is unknown or invalid.
        """
        if cfg == {} or cfg is None:
            return None
        
        scorer_type = cfg.get("type", None)
        if scorer_type == "KeywordScorer":
            keywords = cfg.get("keywords", [])
            if not keywords:
                raise ValueError(
                    f"KeywordScorer requires a non-empty keywords list. scorer_type: {scorer_type}, Suggestion: Provide keywords list or use 'None' for no scoring"
                )
            return KeywordScorer(keywords)
        elif scorer_type == "FuzzyScorer":
            keywords = cfg.get("keywords", [])
            if not keywords:
                raise ValueError(
                    f"FuzzyScorer requires a non-empty keywords list. scorer_type: {scorer_type}, Suggestion: Provide keywords list or use 'None' for no scoring"
                )
            return FuzzyScorer(keywords)
        elif scorer_type in ("None", None):
            return None
        else:
            raise ValueError(
                f"Unknown scorer type: {scorer_type}. scorer_type: {scorer_type}, Suggestion: Use 'KeywordScorer', 'FuzzyScorer', or 'None'"
            )


@dataclass
class DataPreprocessingConfig(BaseConfig):
    """Configuration for the data preprocessing pipeline."""
    target_column: str = SYSTEM_RAW_DATA_COLUMN
    drop_target_column: bool = DEFAULT_DROP_TARGET_COLUMN
    splitting: SplittingConfig = field(default_factory=SplittingConfig) # use default factory because these types are mutable
    scoring: ScoringConfig = field(default_factory=ScoringConfig) # use default factory because these types are mutable
    pandas_score_filter: Optional[str] = DEFAULT_PANDAS_SCORE_FILTER
    preprocessed_data_path: Optional[str] = None
    _explicitly_set_fields: set = field(default_factory=set, init=False)

    def validate(self):
        """Validate the preprocessing configuration.

        Raises:
            ValueError: If any field is invalid or conflicts are found when
                ``preprocessed_data_path`` is provided.
        """
        if self.preprocessed_data_path:
            self._validate_preprocessed_data_path()
            self._validate_no_conflicts_with_preprocessed_data()
            return
        
        self._validate_basic_fields()
        self.splitting.validate()
        self.scoring.validate()

    def _validate_preprocessed_data_path(self):
        """Validate ``preprocessed_data_path`` when provided.

        Raises:
            ValueError: If the file is not a feather file or lacks required columns.
        """
        if self.preprocessed_data_path is None:
            return
            
        if not self.preprocessed_data_path.endswith(".feather"):
            raise ValueError(
                f"preprocessed_data_path must be a feather file. preprocessed_data_path: {self.preprocessed_data_path}, Suggestion: Provide a valid feather file path"
            )
        
        # Verify file has correct columns
        import pandas as pd
        from .constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN
        try:
            df = pd.read_feather(self.preprocessed_data_path)
            if not all(col in df.columns for col in [SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN]):
                raise ValueError(
                    f"preprocessed_data_path must have the correct columns. preprocessed_data_path: {self.preprocessed_data_path}, Suggestion: Provide a valid feather file path with the correct columns"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to read preprocessed data file. preprocessed_data_path: {self.preprocessed_data_path}"
            ) from e

    def _validate_no_conflicts_with_preprocessed_data(self):
        """Ensure no conflicting fields are set when using preprocessed data.

        Raises:
            ValueError: If mutually exclusive fields are provided.
        """
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
            raise ValueError(
                f"Cannot specify {', '.join(conflicting)} when preprocessed_data_path is set. preprocessed_data_path: {self.preprocessed_data_path}, Suggestion: Remove other data fields when using preprocessed_data_path."
            )

    def _validate_basic_fields(self):
        """Validate basic preprocessing fields.

        Raises:
            ValueError: If individual fields are malformed.
        """
        if not isinstance(self.target_column, str) or not self.target_column:
            raise ValueError(
                f"target_column must be a non-empty string. target_column: {self.target_column}, Suggestion: Provide a valid column name"
            )
        if not isinstance(self.drop_target_column, bool):
            raise ValueError(
                f"drop_target_column must be a boolean. drop_target_column: {self.drop_target_column}, Suggestion: Use True or False"
            )
        if self.pandas_score_filter is not None:
            if not isinstance(self.pandas_score_filter, str):
                raise ValueError(
                    f"pandas_score_filter must be a string or None. pandas_score_filter: {self.pandas_score_filter}, Suggestion: Provide a valid pandas query string or None"
                )
            # Validate pandas query syntax
            import pandas as pd
            from .constants import SYSTEM_SCORE_COLUMN
            try:
                pd.DataFrame({SYSTEM_SCORE_COLUMN: [1]}).query(self.pandas_score_filter)
            except Exception as e:
                raise ValueError(
                    f"pandas_score_filter is not a valid pandas query: {e}. pandas_score_filter: {self.pandas_score_filter}, Suggestion: Provide a valid pandas query string. Make sure to use the {SYSTEM_SCORE_COLUMN} column name."
                )

    def to_dict(self) -> dict:
        """Serialize preprocessing configuration.

        Returns:
            A dictionary representation suitable for YAML serialization.
        """
        if self.preprocessed_data_path:
            return {"preprocessed_data_path": self.preprocessed_data_path}
        
        return {
            "target_column": self.target_column,
            "drop_target_column": self.drop_target_column,
            "pandas_score_filter": self.pandas_score_filter,
            "splitting": self.splitting.to_dict(),
            "scoring": self.scoring.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPreprocessingConfig":
        """Construct a ``DataPreprocessingConfig`` from a mapping.

        Tracks which fields were explicitly set to detect conflicts when
        ``preprocessed_data_path`` is used.

        Args:
            data: Mapping of preprocessing options.

        Returns:
            A configured ``DataPreprocessingConfig`` instance.
        """
        # Track explicitly set fields
        explicitly_set_fields = set(data.keys())
        
        instance = cls(
            target_column=data.get("target_column", SYSTEM_RAW_DATA_COLUMN),
            drop_target_column=data.get("drop_target_column", DEFAULT_DROP_TARGET_COLUMN),
            splitting=SplittingConfig.from_dict(data.get("splitting", {})),
            scoring=ScoringConfig.from_dict(data.get("scoring", {})),
            pandas_score_filter=data.get("pandas_score_filter", DEFAULT_PANDAS_SCORE_FILTER),
            preprocessed_data_path=data.get("preprocessed_data_path", None),
        )
        instance._explicitly_set_fields = explicitly_set_fields
        return instance


@dataclass
class SchemaConfig(BaseConfig):
    """Configuration for extraction schema reference and settings.

    This config contains:
    - Path to the schema specification file (schema_spec.yaml)
    - Schema‑specific settings (prompts)

    The actual schema definition (including container_name) is stored in the
    separate schema_spec.yaml file.
    """
    spec_path: Optional[Union[str, Path]] = DEFAULT_SCHEMA_PATH
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def validate(self):
        """Validate schema configuration.

        Raises:
            ValueError: If the spec path does not exist or fields are malformed.
        """
        if not isinstance(self.spec_path, (Path, str)) or not self.spec_path:
            raise ValueError(
                f"spec_path must be a valid Path or string. spec_path: {str(self.spec_path)}, Suggestion: Provide a valid file path"
            )
        if isinstance(self.spec_path, str):
            spec_path = Path(self.spec_path)
        else:
            spec_path = self.spec_path
        if not spec_path.exists():
            raise ValueError(
                f"Schema spec file does not exist: {spec_path}, Suggestion: Check the file path or create the schema file"
            )
        if not isinstance(self.prompt_template, str):
            raise ValueError(
                f"prompt_template must be a string. prompt_template: {self.prompt_template}, Suggestion: Provide a valid string for the prompt template or omit to use the default prompt template."
            )
        if not isinstance(self.system_prompt, str):
            raise ValueError(
                f"system_prompt must be a string. system_prompt: {self.system_prompt}, Suggestion: Provide a valid string for the system prompt or omit to use the default system prompt."
            )

    def to_dict(self) -> dict:
        """Serialize schema configuration to a dictionary."""
        return {
            "spec_path": str(self.spec_path) if self.spec_path else None,
            "prompt_template": self.prompt_template,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaConfig":
        """Construct a ``SchemaConfig`` from a mapping."""
        if data is None:
            data = {}
        
        spec_path = data.get("spec_path", "")
        if isinstance(spec_path, str):
            spec_path = Path(spec_path)
        
        return cls(
            spec_path=spec_path,
            prompt_template=data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
            system_prompt=data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )


@dataclass
class SemanticCacheConfig(BaseConfig):
    """Persistent semantic‑cache settings."""
    backend: str = DEFAULT_SEMANTIC_CACHE_BACKEND
    path: Union[str, Path] = DEFAULT_SEMANTIC_CACHE_PATH
    max_size_mb: int = DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB
    synchronous: str = DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS

    def resolve_path(self) -> Path:
        """Resolve and return the cache path."""
        return Path(self.path).expanduser().resolve()

    def validate(self):
        """Validate semantic cache configuration.

        Raises:
            ValueError: If backend or parameters are invalid.
        """
        if self.backend not in {"sqlite", "lmdb", "filesystem"}:
            raise ValueError(
                f"cache.backend must be 'sqlite', 'lmdb', or 'filesystem'. backend: {self.backend}"
            )
        if not isinstance(self.max_size_mb, int) or self.max_size_mb <= 0:
            raise ValueError(
                f"cache.max_size_mb must be a positive integer. max_size_mb: {self.max_size_mb}"
            )
        if self.backend == "sqlite" and self.synchronous not in {"normal", "full"}:
            raise ValueError(
                f"cache.synchronous must be 'normal' or 'full' for SQLite. synchronous: {self.synchronous}"
            )

    def to_dict(self) -> dict:
        """Serialize semantic cache configuration."""
        return {
            "backend": self.backend,
            "path": str(self.path),
            "max_size_mb": self.max_size_mb,
            "synchronous": self.synchronous,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticCacheConfig":
        """Construct a ``SemanticCacheConfig`` from a mapping."""
        if data is None:
            data = {}
        
        return cls(
            backend=data.get("backend", DEFAULT_SEMANTIC_CACHE_BACKEND),
            path=data.get("path", DEFAULT_SEMANTIC_CACHE_PATH),
            max_size_mb=data.get("max_size_mb", DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB),
            synchronous=data.get("synchronous", DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS),
        )


@dataclass
class DELMConfig(BaseConfig):
    """Complete DELM configuration including pipeline and schema reference.

    Contains:
    - Pipeline configuration (LLM settings, data preprocessing, etc.)
    - Reference to a separate schema specification file

    The configuration can be loaded from:
    - A single pipeline config file (config.yaml) that references a schema file
    - Separate pipeline config and schema spec files
    """
    llm_extraction: LLMExtractionConfig
    data_preprocessing: DataPreprocessingConfig
    schema: SchemaConfig
    semantic_cache: SemanticCacheConfig

    def validate(self):
        """Validate all sub‑configurations."""
        self.llm_extraction.validate()
        self.data_preprocessing.validate()
        self.schema.validate()
        self.semantic_cache.validate()

    def to_serialized_config_dict(self) -> dict:
        """Return a dictionary suitable for saving as pipeline config YAML."""
        return {
            "llm_extraction": self.llm_extraction.to_dict(),
            "data_preprocessing": self.data_preprocessing.to_dict(),
            "schema": self.schema.to_dict(),
            "semantic_cache": self.semantic_cache.to_dict(),
        }

    def to_serialized_schema_spec_dict(self) -> dict:
        """Load and return the schema spec as a dictionary (schema_spec.yaml)."""
        import yaml
        import json
        
        path = self.schema.spec_path
        if path is None:
            raise ValueError("Schema spec path is None")
        
        if isinstance(path, str):
            path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Schema spec file does not exist: {path}")
        
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text()) or {}
        elif path.suffix.lower() == ".json":
            return json.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported schema file format: {path.suffix}")
        

    # Backward compatibility aliases
    def to_dict(self) -> dict:
        """Alias for ``to_serialized_config_dict`` for backward compatibility."""
        return self.to_serialized_config_dict()


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DELMConfig":
        """Create ``DELMConfig`` from a mapping."""
        if data is None:
            data = {}
        
        return cls(
            llm_extraction=LLMExtractionConfig.from_dict(data.get("llm_extraction", {})),
            data_preprocessing=DataPreprocessingConfig.from_dict(data.get("data_preprocessing", {})),
            schema=SchemaConfig.from_dict(data.get("schema", {})),
            semantic_cache=SemanticCacheConfig.from_dict(data.get("semantic_cache", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "DELMConfig":
        """Create ``DELMConfig`` from a pipeline config YAML file.

        Args:
            path: Path to the YAML configuration.

        Returns:
            A configured ``DELMConfig`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"YAML config file does not exist: {path}"
            )
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)

    @staticmethod
    def from_any(
        config_like: "DELMConfig | dict[str, Any] | str | Path",
    ) -> "DELMConfig":
        """Create ``DELMConfig`` from various input types.

        Args:
            config_like: Instance of ``DELMConfig``, dict, or path to YAML file.

        Returns:
            A configured ``DELMConfig`` instance.

        Raises:
            ValueError: If the input type is unsupported.
        """
        if isinstance(config_like, DELMConfig):
            return config_like
        elif isinstance(config_like, str):
            return DELMConfig.from_yaml(Path(config_like))
        elif isinstance(config_like, dict):
            return DELMConfig.from_dict(config_like)
        else:
            raise ValueError(f"config must be a DELMConfig, dict, or path to YAML. config_type: {type(config_like).__name__}") 