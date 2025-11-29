"""Configuration objects for DELM.

Defines typed, serializable, and validatable configuration classes used across
the DELM pipeline: LLM extraction, splitting/scoring, schema, semantic cache,
and the top‑level ``DELMConfig`` aggregator.

Docstrings follow Google style.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, List
import yaml

T = TypeVar("T", bound="BaseConfig")
T = TypeVar("T", bound="BaseConfig")

from delm.strategies import RelevanceScorer
from delm.strategies import SplitStrategy
from delm.schemas import Schema


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

    provider: str
    model: str
    base_url: Optional[str]
    mode: Optional[str]
    temperature: float
    prompt_template: str
    system_prompt: str
    max_retries: int
    batch_size: int
    max_workers: int
    base_delay: float
    tokens_per_minute: Optional[int]
    requests_per_minute: Optional[int]
    track_cost: bool
    max_budget: Optional[float]
    model_input_cost_per_1M_tokens: Optional[float]
    model_output_cost_per_1M_tokens: Optional[float]

    def get_provider_string(self) -> str:
        """Return the combined provider string for Instructor.

        Returns:
            Provider string in the form ``"<provider>/<model>"``.
        """
        return f"{self.provider}/{self.model}"

    def validate(self):
        """Validate all LLM extraction fields.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError(
                f"Provider must be a non-empty string. provider: {self.provider}, Suggestion: Use e.g. 'openai', 'anthropic', 'google', etc."
            )
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(
                f"Model name must be a non-empty string. model: {self.model}, Suggestion: Use e.g. 'gpt-4o-mini', 'claude-3-sonnet', etc."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0. temperature: {self.temperature}, Suggestion: Use a value between 0.0 and 2.0"
            )
        if not isinstance(self.prompt_template, str):
            raise ValueError(
                f"prompt_template must be a string. prompt_template: {self.prompt_template}, Suggestion: Provide a valid string for the prompt template or omit to use the default prompt template."
            )
        if not isinstance(self.system_prompt, str):
            raise ValueError(
                f"system_prompt must be a string. system_prompt: {self.system_prompt}, Suggestion: Provide a valid string for the system prompt or omit to use the default system prompt."
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
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            raise ValueError(
                f"tokens_per_minute must be positive. tokens_per_minute: {self.tokens_per_minute}, Suggestion: Use a positive integer"
            )
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise ValueError(
                f"requests_per_minute must be positive. requests_per_minute: {self.requests_per_minute}, Suggestion: Use a positive integer"
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
            "model": self.model,
            "base_url": self.base_url,
            "mode": self.mode,
            "temperature": self.temperature,
            "prompt_template": self.prompt_template,
            "system_prompt": self.system_prompt,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "base_delay": self.base_delay,
            "tokens_per_minute": self.tokens_per_minute,
            "requests_per_minute": self.requests_per_minute,
            "track_cost": self.track_cost,
            "max_budget": self.max_budget,
            "model_input_cost_per_1M_tokens": self.model_input_cost_per_1M_tokens,
            "model_output_cost_per_1M_tokens": self.model_output_cost_per_1M_tokens,
        }


# Note: SplittingConfig and ScoringConfig have been removed.
# The strategy classes (SplitStrategy, RelevanceScorer) now handle
# serialization/deserialization directly via their to_dict()/from_dict() methods.


@dataclass
class DataPreprocessingConfig(BaseConfig):
    """Configuration for the data preprocessing pipeline."""

    target_column: str
    drop_target_column: bool
    splitting_strategy: Optional[SplitStrategy] = None
    relevance_scorer: Optional[RelevanceScorer] = None
    score_filter: Optional[str] = None
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

        # Validate strategy objects if they exist
        if self.splitting_strategy is not None and not isinstance(
            self.splitting_strategy, SplitStrategy
        ):
            raise ValueError(
                f"splitting_strategy must be a SplitStrategy instance or None, got {type(self.splitting_strategy).__name__}"
            )
        if self.relevance_scorer is not None and not isinstance(
            self.relevance_scorer, RelevanceScorer
        ):
            raise ValueError(
                f"relevance_scorer must be a RelevanceScorer instance or None, got {type(self.relevance_scorer).__name__}"
            )

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
            if not all(
                col in df.columns
                for col in [SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN]
            ):
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
        if "score_filter" in self._explicitly_set_fields:
            conflicting.append("score_filter")
        if self.splitting_strategy is not None:
            conflicting.append("splitting_strategy")
        if self.relevance_scorer is not None:
            conflicting.append("relevance_scorer")

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
        if self.score_filter is not None:
            if not isinstance(self.score_filter, str):
                raise ValueError(
                    f"score_filter must be a string or None. score_filter: {self.score_filter}, Suggestion: Provide a valid pandas query string or None"
                )
            # Validate pandas query syntax
            import pandas as pd
            from .constants import SYSTEM_SCORE_COLUMN

            try:
                pd.DataFrame({SYSTEM_SCORE_COLUMN: [1]}).query(self.score_filter)
            except Exception as e:
                raise ValueError(
                    f"score_filter is not a valid pandas query: {e}. score_filter: {self.score_filter}, Suggestion: Provide a valid pandas query string. Make sure to use the {SYSTEM_SCORE_COLUMN} column name."
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
            "score_filter": self.score_filter,
            "splitting_strategy": (
                self.splitting_strategy.to_dict()
                if self.splitting_strategy
                else {"type": "None"}
            ),
            "relevance_scorer": (
                self.relevance_scorer.to_dict()
                if self.relevance_scorer
                else {"type": "None"}
            ),
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

        # Convert dict format back to strategy objects using their own from_dict methods
        splitting_dict = data["splitting_strategy"]
        if splitting_dict and splitting_dict.get("type") not in ("None", None):
            splitting_strategy = SplitStrategy.from_dict(splitting_dict)
        else:
            splitting_strategy = None

        scorer_dict = data["relevance_scorer"]
        if scorer_dict and scorer_dict.get("type") not in ("None", None):
            relevance_scorer = RelevanceScorer.from_dict(scorer_dict)
        else:
            relevance_scorer = None

        instance = cls(
            target_column=data["target_column"],
            drop_target_column=data["drop_target_column"],
            splitting_strategy=splitting_strategy,
            relevance_scorer=relevance_scorer,
            score_filter=data["score_filter"],
            preprocessed_data_path=data["preprocessed_data_path"],
        )
        instance._explicitly_set_fields = explicitly_set_fields
        return instance


@dataclass
class SemanticCacheConfig(BaseConfig):
    """Persistent semantic‑cache settings."""

    backend: Optional[str]
    path: Union[str, Path]
    max_size_mb: int
    synchronous: str

    def resolve_path(self) -> Path:
        """Resolve and return the cache path."""
        return Path(self.path).expanduser().resolve()

    def validate(self):
        """Validate semantic cache configuration.

        Raises:
            ValueError: If backend or parameters are invalid.
        """
        if self.backend not in {None, "none", "sqlite", "lmdb", "filesystem"}:
            raise ValueError(
                f"cache.backend must be None, 'none', 'sqlite', 'lmdb', or 'filesystem'. backend: {self.backend}"
            )
        # Skip remaining validation if caching is disabled
        if self.backend is None or self.backend == "none":
            return
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
            "cache_backend": self.backend,
            "cache_path": str(self.path),
            "cache_max_size_mb": self.max_size_mb,
            "cache_synchronous": self.synchronous,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticCacheConfig":
        """Construct a ``SemanticCacheConfig`` from a mapping."""
        if data is None:
            data = {}

        return cls(
            backend=data["cache_backend"],
            path=data["cache_path"],
            max_size_mb=data["cache_max_size_mb"],
            synchronous=data["cache_synchronous"],
        )


class DELMConfig:
    """Complete DELM configuration including pipeline and schema reference.

    Contains:
    - Pipeline configuration (LLM settings, data preprocessing, etc.)
    - Reference to a separate schema specification file

    The configuration can be loaded from:
    - A single pipeline config file (config.yaml) that references a schema file
    - Separate pipeline config and schema spec files
    """

    def __init__(
        self,
        *,
        schema: Union[str, Path, dict, Schema],
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: float = 0.0,
        batch_size: int = 10,
        max_workers: int = 1,
        max_retries: int = 3,
        base_delay: float = 1.0,
        tokens_per_minute: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        track_cost: bool = True,
        max_budget: Optional[float] = None,
        model_input_cost_per_1M_tokens: Optional[float] = None,
        model_output_cost_per_1M_tokens: Optional[float] = None,
        # Data Preprocessing (flat)
        target_column: str = "text",
        drop_target_column: bool = False,
        splitting_strategy: Optional[Union[dict, SplitStrategy]] = None,
        relevance_scorer: Optional[Union[dict, RelevanceScorer]] = None,
        score_filter: Optional[str] = None,  # pandas query syntax
        # Prompt Settings
        prompt_template: Optional[
            str
        ] = "Extract the following information from the text:\n\n{variables}\n\nText to analyze:\n{text}",
        system_prompt: Optional[str] = "You are a precise data-extraction assistant.",
        # Semantic Cache Settings
        cache_backend: Optional[str] = "sqlite",
        cache_path: Union[str, Path] = ".delm/cache",
        cache_max_size_mb: int = 512,
        cache_synchronous: str = "normal",
    ) -> None:
        """Initialize the DELM configuration."""

        # Load schema
        if isinstance(schema, (str, Path)):
            schema = Schema.from_yaml(schema)
        elif isinstance(schema, dict):
            schema = Schema.from_dict(schema)

        # Load SplittingStrategy
        if isinstance(splitting_strategy, dict):
            splitting_strategy = SplitStrategy.from_dict(splitting_strategy)
        elif isinstance(splitting_strategy, SplitStrategy):
            splitting_strategy = splitting_strategy

        # Load RelevanceScorer
        if isinstance(relevance_scorer, dict):
            relevance_scorer = RelevanceScorer.from_dict(relevance_scorer)
        elif isinstance(relevance_scorer, RelevanceScorer):
            relevance_scorer = relevance_scorer

        self.schema = schema

        self.llm_extraction_cfg = LLMExtractionConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            mode=mode,
            temperature=temperature,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            base_delay=base_delay,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
            track_cost=track_cost,
            max_budget=max_budget,
            model_input_cost_per_1M_tokens=model_input_cost_per_1M_tokens,
            model_output_cost_per_1M_tokens=model_output_cost_per_1M_tokens,
        )
        self.data_preprocessing_cfg = DataPreprocessingConfig(
            target_column=target_column,
            drop_target_column=drop_target_column,
            splitting_strategy=splitting_strategy,
            relevance_scorer=relevance_scorer,
            score_filter=score_filter,
        )
        self.semantic_cache_cfg = SemanticCacheConfig(
            backend=cache_backend,
            path=cache_path,
            max_size_mb=cache_max_size_mb,
            synchronous=cache_synchronous,
        )

    def validate(self):
        """Validate all sub‑configurations."""
        self.llm_extraction_cfg.validate()
        self.data_preprocessing_cfg.validate()
        self.semantic_cache_cfg.validate()

    def to_dict(self) -> dict:
        """Return a dictionary suitable for saving as pipeline config YAML."""
        data = {}

        data.update(self.llm_extraction_cfg.to_dict())
        data.update(self.data_preprocessing_cfg.to_dict())
        data.update(self.semantic_cache_cfg.to_dict())
        data["schema"] = self.schema.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DELMConfig":
        """Create ``DELMConfig`` from a mapping.

        Handles two formats:
        1. Nested format (from to_dict()): Has 'llm_extraction', 'data_preprocessing', 'semantic_cache' keys
        2. Flat format: All fields at top level
        """
        if data is None:
            data = {}

        # Check if this is nested format (from to_dict())
        return cls(
            schema=data["schema"],
            provider=data["provider"],
            model=data["model"],
            base_url=data["base_url"],
            mode=data["mode"],
            temperature=data["temperature"],
            prompt_template=data["prompt_template"],
            system_prompt=data["system_prompt"],
            batch_size=data["batch_size"],
            max_workers=data["max_workers"],
            max_retries=data["max_retries"],
            base_delay=data["base_delay"],
            tokens_per_minute=data["tokens_per_minute"],
            requests_per_minute=data["requests_per_minute"],
            track_cost=data["track_cost"],
            max_budget=data["max_budget"],
            model_input_cost_per_1M_tokens=data["model_input_cost_per_1M_tokens"],
            model_output_cost_per_1M_tokens=data["model_output_cost_per_1M_tokens"],
            target_column=data["target_column"],
            drop_target_column=data["drop_target_column"],
            splitting_strategy=data["splitting_strategy"],
            relevance_scorer=data["relevance_scorer"],
            score_filter=data["score_filter"],
            cache_backend=data["cache_backend"],
            cache_path=data["cache_path"],
            cache_max_size_mb=data["cache_max_size_mb"],
            cache_synchronous=data["cache_synchronous"],
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DELMConfig":
        """Create ``DELMConfig`` from a pipeline config YAML file.

        Args:
            path: Path to the YAML configuration.

        Returns:
            A configured ``DELMConfig`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML config file does not exist: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @staticmethod
    def from_any(
        config_like: "DELMConfig | dict[str, Any] | str | Path | DELM",
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
            raise ValueError(
                f"config must be a DELMConfig, dict, or path to YAML. config_type: {type(config_like).__name__}"
            )

            raise ValueError(
                f"config must be a DELMConfig, dict, or path to YAML. config_type: {type(config_like).__name__}"
            )
