"""
Unit tests for DELM configuration module.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd

from delm.config import (
    BaseConfig,
    LLMExtractionConfig,
    SplittingConfig,
    ScoringConfig,
    DataPreprocessingConfig,
    SchemaConfig,
    SemanticCacheConfig,
    DELMConfig,
)
from delm.strategies import (
    RelevanceScorer,
    KeywordScorer,
    FuzzyScorer,
    SplitStrategy,
    ParagraphSplit,
    FixedWindowSplit,
    RegexSplit,
)
from delm.constants import (
    DEFAULT_PROVIDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_BASE_DELAY,
    DEFAULT_TRACK_COST,
    DEFAULT_MAX_BUDGET,
    DEFAULT_DOTENV_PATH,
    DEFAULT_FIXED_WINDOW_SIZE,
    DEFAULT_FIXED_WINDOW_STRIDE,
    DEFAULT_REGEX_PATTERN,
    DEFAULT_DROP_TARGET_COLUMN,
    DEFAULT_PANDAS_SCORE_FILTER,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SEMANTIC_CACHE_BACKEND,
    DEFAULT_SEMANTIC_CACHE_PATH,
    DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB,
    DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS,
    SYSTEM_RAW_DATA_COLUMN,
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_SCORE_COLUMN,
)





class TestLLMExtractionConfig:
    """Test LLM extraction configuration."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        config = LLMExtractionConfig()
        assert config.provider == DEFAULT_PROVIDER
        assert config.name == DEFAULT_MODEL_NAME
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.max_workers == DEFAULT_MAX_WORKERS
        assert config.base_delay == DEFAULT_BASE_DELAY
        assert config.dotenv_path == DEFAULT_DOTENV_PATH
        assert config.track_cost == DEFAULT_TRACK_COST
        assert config.max_budget == DEFAULT_MAX_BUDGET

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        config = LLMExtractionConfig(
            provider="anthropic",
            name="claude-3-sonnet",
            temperature=0.5,
            max_retries=5,
            batch_size=10,
            max_workers=4,
            base_delay=1.0,
            track_cost=True,
            max_budget=100.0,
        )
        assert config.provider == "anthropic"
        assert config.name == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.batch_size == 10
        assert config.max_workers == 4
        assert config.base_delay == 1.0
        assert config.track_cost is True
        assert config.max_budget == 100.0

    def test_get_provider_string(self):
        """Test get_provider_string method."""
        config = LLMExtractionConfig(provider="openai", name="gpt-4")
        assert config.get_provider_string() == "openai/gpt-4"

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = LLMExtractionConfig()
        # Should not raise any exception
        config.validate()

    def test_validate_invalid_provider(self):
        """Test validation with invalid provider."""
        config = LLMExtractionConfig(provider="")
        with pytest.raises(ValueError, match="Provider must be a non-empty string"):
            config.validate()

        config = LLMExtractionConfig(provider=123)
        with pytest.raises(ValueError, match="Provider must be a non-empty string"):
            config.validate()

    def test_validate_invalid_name(self):
        """Test validation with invalid model name."""
        config = LLMExtractionConfig(name="")
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            config.validate()

        config = LLMExtractionConfig(name=123)
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            config.validate()

    def test_validate_invalid_temperature(self):
        """Test validation with invalid temperature."""
        config = LLMExtractionConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            config.validate()

        config = LLMExtractionConfig(temperature=2.1)
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            config.validate()

    def test_validate_invalid_max_retries(self):
        """Test validation with invalid max_retries."""
        config = LLMExtractionConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation with invalid batch_size."""
        config = LLMExtractionConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

        config = LLMExtractionConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_validate_invalid_max_workers(self):
        """Test validation with invalid max_workers."""
        config = LLMExtractionConfig(max_workers=0)
        with pytest.raises(ValueError, match="max_workers must be positive"):
            config.validate()

        config = LLMExtractionConfig(max_workers=-1)
        with pytest.raises(ValueError, match="max_workers must be positive"):
            config.validate()

    def test_validate_invalid_base_delay(self):
        """Test validation with invalid base_delay."""
        config = LLMExtractionConfig(base_delay=-0.1)
        with pytest.raises(ValueError, match="base_delay must be non-negative"):
            config.validate()

    def test_validate_invalid_track_cost(self):
        """Test validation with invalid track_cost."""
        config = LLMExtractionConfig(track_cost="True")
        with pytest.raises(ValueError, match="track_cost must be a boolean"):
            config.validate()

    def test_validate_max_budget_without_track_cost(self):
        """Test validation when max_budget is set but track_cost is False."""
        config = LLMExtractionConfig(track_cost=False, max_budget=100.0)
        with pytest.raises(ValueError, match="track_cost must be True if max_budget is specified"):
            config.validate()

    def test_validate_invalid_max_budget(self):
        """Test validation with invalid max_budget."""
        config = LLMExtractionConfig(track_cost=True, max_budget="100")
        with pytest.raises(ValueError, match="max_budget must be a number"):
            config.validate()

    def test_to_dict(self):
        """Test to_dict method."""
        config = LLMExtractionConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_retries=3,
            batch_size=5,
            max_workers=2,
            base_delay=0.5,
            track_cost=True,
            max_budget=50.0,
            model_input_cost_per_1M_tokens=10.0,
            model_output_cost_per_1M_tokens=30.0,
        )
        result = config.to_dict()
        expected = {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7,
            "max_retries": 3,
            "batch_size": 5,
            "max_workers": 2,
            "base_delay": 0.5,
            "dotenv_path": None,
            "track_cost": True,
            "max_budget": 50.0,
            "model_input_cost_per_1M_tokens": 10.0,
            "model_output_cost_per_1M_tokens": 30.0,
        }
        assert result == expected

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "provider": "anthropic",
            "name": "claude-3-sonnet",
            "temperature": 0.5,
            "max_retries": 5,
            "batch_size": 10,
            "max_workers": 4,
            "base_delay": 1.0,
            "track_cost": True,
            "max_budget": 100.0,
        }
        config = LLMExtractionConfig.from_dict(data)
        assert config.provider == "anthropic"
        assert config.name == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.batch_size == 10
        assert config.max_workers == 4
        assert config.base_delay == 1.0
        assert config.track_cost is True
        assert config.max_budget == 100.0


class TestSplittingConfig:
    """Test splitting configuration."""

    def test_initialization_default(self):
        """Test initialization with default values."""
        config = SplittingConfig()
        assert config.strategy is None

    def test_initialization_with_strategy(self):
        """Test initialization with a strategy."""
        strategy = ParagraphSplit()
        config = SplittingConfig(strategy=strategy)
        assert config.strategy == strategy

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = SplittingConfig()
        config.validate()  # Should not raise

        config = SplittingConfig(strategy=ParagraphSplit())
        config.validate()  # Should not raise

    def test_validate_invalid_strategy(self):
        """Test validation with invalid strategy."""
        config = SplittingConfig(strategy="invalid")
        with pytest.raises(ValueError, match="strategy must be a SplitStrategy instance"):
            config.validate()

    def test_to_dict_no_strategy(self):
        """Test to_dict with no strategy."""
        config = SplittingConfig()
        result = config.to_dict()
        assert result == {"type": "None"}

    def test_to_dict_with_strategy(self):
        """Test to_dict with a strategy."""
        strategy = FixedWindowSplit(window=100, stride=50)
        config = SplittingConfig(strategy=strategy)
        result = config.to_dict()
        expected = {
            "type": "FixedWindowSplit",
            "window": 100,
            "stride": 50,
        }
        assert result == expected

    def test_from_dict_none(self):
        """Test from_dict with None or empty dict."""
        config = SplittingConfig.from_dict({})
        assert config.strategy is None

        config = SplittingConfig.from_dict(None)
        assert config.strategy is None

    def test_from_dict_paragraph_split(self):
        """Test from_dict with ParagraphSplit."""
        data = {"type": "ParagraphSplit"}
        config = SplittingConfig.from_dict(data)
        assert isinstance(config.strategy, ParagraphSplit)

    def test_from_dict_fixed_window_split(self):
        """Test from_dict with FixedWindowSplit."""
        data = {
            "type": "FixedWindowSplit",
            "window": 200,
            "stride": 100,
        }
        config = SplittingConfig.from_dict(data)
        assert isinstance(config.strategy, FixedWindowSplit)
        assert config.strategy.window == 200
        assert config.strategy.stride == 100

    def test_from_dict_fixed_window_split_defaults(self):
        """Test from_dict with FixedWindowSplit using defaults."""
        data = {"type": "FixedWindowSplit"}
        config = SplittingConfig.from_dict(data)
        assert isinstance(config.strategy, FixedWindowSplit)
        assert config.strategy.window == DEFAULT_FIXED_WINDOW_SIZE
        assert config.strategy.stride == DEFAULT_FIXED_WINDOW_STRIDE

    def test_from_dict_regex_split(self):
        """Test from_dict with RegexSplit."""
        data = {
            "type": "RegexSplit",
            "pattern": r"\n\n",
        }
        config = SplittingConfig.from_dict(data)
        assert isinstance(config.strategy, RegexSplit)
        assert config.strategy.pattern_str == r"\n\n"

    def test_from_dict_regex_split_default(self):
        """Test from_dict with RegexSplit using default."""
        data = {"type": "RegexSplit"}
        config = SplittingConfig.from_dict(data)
        assert isinstance(config.strategy, RegexSplit)
        assert config.strategy.pattern_str == DEFAULT_REGEX_PATTERN

    def test_from_dict_unknown_strategy(self):
        """Test from_dict with unknown strategy."""
        data = {"type": "UnknownSplit"}
        with pytest.raises(ValueError, match="Unknown split strategy"):
            SplittingConfig.from_dict(data)


class TestScoringConfig:
    """Test scoring configuration."""

    def test_initialization_default(self):
        """Test initialization with default values."""
        config = ScoringConfig()
        assert config.scorer is None

    def test_initialization_with_scorer(self):
        """Test initialization with a scorer."""
        scorer = KeywordScorer(["test", "example"])
        config = ScoringConfig(scorer=scorer)
        assert config.scorer == scorer

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = ScoringConfig()
        config.validate()  # Should not raise

        scorer = KeywordScorer(["test"])
        config = ScoringConfig(scorer=scorer)
        config.validate()  # Should not raise

    def test_validate_invalid_scorer(self):
        """Test validation with invalid scorer."""
        config = ScoringConfig(scorer="invalid")
        with pytest.raises(ValueError, match="scorer must be a RelevanceScorer instance"):
            config.validate()

    def test_to_dict_no_scorer(self):
        """Test to_dict with no scorer."""
        config = ScoringConfig()
        result = config.to_dict()
        assert result == {"type": "None"}

    def test_to_dict_with_scorer(self):
        """Test to_dict with a scorer."""
        scorer = KeywordScorer(["test", "example"])
        config = ScoringConfig(scorer=scorer)
        result = config.to_dict()
        expected = {
            "type": "KeywordScorer",
            "keywords": ["test", "example"],
        }
        assert result == expected

    def test_from_dict_none(self):
        """Test from_dict with None or empty dict."""
        config = ScoringConfig.from_dict({})
        assert config.scorer is None

        config = ScoringConfig.from_dict(None)
        assert config.scorer is None

    def test_from_dict_keyword_scorer(self):
        """Test from_dict with KeywordScorer."""
        data = {
            "type": "KeywordScorer",
            "keywords": ["test", "example"],
        }
        config = ScoringConfig.from_dict(data)
        assert isinstance(config.scorer, KeywordScorer)
        assert config.scorer.keywords == ["test", "example"]

    def test_from_dict_keyword_scorer_empty_keywords(self):
        """Test from_dict with KeywordScorer and empty keywords."""
        data = {
            "type": "KeywordScorer",
            "keywords": [],
        }
        with pytest.raises(ValueError, match="KeywordScorer requires a non-empty keywords list"):
            ScoringConfig.from_dict(data)

    def test_from_dict_fuzzy_scorer(self):
        """Test from_dict with FuzzyScorer."""
        data = {
            "type": "FuzzyScorer",
            "keywords": ["test", "example"],
        }
        config = ScoringConfig.from_dict(data)
        assert isinstance(config.scorer, FuzzyScorer)
        assert config.scorer.keywords == ["test", "example"]

    def test_from_dict_fuzzy_scorer_empty_keywords(self):
        """Test from_dict with FuzzyScorer and empty keywords."""
        data = {
            "type": "FuzzyScorer",
            "keywords": [],
        }
        with pytest.raises(ValueError, match="FuzzyScorer requires a non-empty keywords list"):
            ScoringConfig.from_dict(data)

    def test_from_dict_unknown_scorer(self):
        """Test from_dict with unknown scorer."""
        data = {"type": "UnknownScorer"}
        with pytest.raises(ValueError, match="Unknown scorer type"):
            ScoringConfig.from_dict(data)


class TestDataPreprocessingConfig:
    """Test data preprocessing configuration."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        config = DataPreprocessingConfig()
        assert config.target_column == SYSTEM_RAW_DATA_COLUMN
        assert config.drop_target_column == DEFAULT_DROP_TARGET_COLUMN
        assert config.pandas_score_filter == DEFAULT_PANDAS_SCORE_FILTER
        assert config.preprocessed_data_path is None
        assert isinstance(config.splitting, SplittingConfig)
        assert isinstance(config.scoring, ScoringConfig)

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        config = DataPreprocessingConfig(
            target_column="custom_column",
            drop_target_column=True,
            pandas_score_filter="score > 0.5",
            preprocessed_data_path="data.feather",
        )
        assert config.target_column == "custom_column"
        assert config.drop_target_column is True
        assert config.pandas_score_filter == "score > 0.5"
        assert config.preprocessed_data_path == "data.feather"

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = DataPreprocessingConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_target_column(self):
        """Test validation with invalid target_column."""
        config = DataPreprocessingConfig(target_column="")
        with pytest.raises(ValueError, match="target_column must be a non-empty string"):
            config.validate()

        config = DataPreprocessingConfig(target_column=123)
        with pytest.raises(ValueError, match="target_column must be a non-empty string"):
            config.validate()

    def test_validate_invalid_drop_target_column(self):
        """Test validation with invalid drop_target_column."""
        config = DataPreprocessingConfig(drop_target_column="True")
        with pytest.raises(ValueError, match="drop_target_column must be a boolean"):
            config.validate()

    def test_validate_invalid_pandas_score_filter(self):
        """Test validation with invalid pandas_score_filter."""
        config = DataPreprocessingConfig(pandas_score_filter=123)
        with pytest.raises(ValueError, match="pandas_score_filter must be a string or None"):
            config.validate()

    def test_validate_invalid_pandas_query(self):
        """Test validation with invalid pandas query."""
        config = DataPreprocessingConfig(pandas_score_filter="invalid query")
        with pytest.raises(ValueError, match="pandas_score_filter is not a valid pandas query"):
            config.validate()

    def test_validate_valid_pandas_query(self):
        """Test validation with valid pandas query."""
        config = DataPreprocessingConfig(pandas_score_filter=f"{SYSTEM_SCORE_COLUMN} > 0.5")
        config.validate()  # Should not raise

    def test_validate_preprocessed_data_path_not_feather(self):
        """Test validation with non-feather preprocessed data path."""
        config = DataPreprocessingConfig(preprocessed_data_path="data.csv")
        with pytest.raises(ValueError, match="preprocessed_data_path must be a feather file"):
            config.validate()

    def test_validate_preprocessed_data_path_missing_columns(self):
        """Test validation with preprocessed data missing required columns."""
        # Create a temporary feather file with wrong columns
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_path = f.name
        
        try:
            # Create a DataFrame with wrong columns and save as feather
            df = pd.DataFrame({"wrong_column": [1]})
            df.to_feather(temp_path)
            
            config = DataPreprocessingConfig(preprocessed_data_path=temp_path)
            with pytest.raises(ValueError, match="Failed to read preprocessed data file"):
                config.validate()
        finally:
            Path(temp_path).unlink()

    @patch('pandas.read_feather')
    def test_validate_preprocessed_data_path_valid(self, mock_read_feather):
        """Test validation with valid preprocessed data path."""
        mock_df = pd.DataFrame({
            SYSTEM_CHUNK_COLUMN: ["chunk1"],
            SYSTEM_CHUNK_ID_COLUMN: [1],
        })
        mock_read_feather.return_value = mock_df
        
        config = DataPreprocessingConfig(preprocessed_data_path="data.feather")
        config.validate()  # Should not raise

    @patch('pandas.read_feather')
    def test_validate_preprocessed_data_conflicts(self, mock_read_feather):
        """Test validation when preprocessed data conflicts with other settings."""
        mock_df = pd.DataFrame({
            SYSTEM_CHUNK_COLUMN: ["chunk1"],
            SYSTEM_CHUNK_ID_COLUMN: [1],
        })
        mock_read_feather.return_value = mock_df
        
        config = DataPreprocessingConfig(
            preprocessed_data_path="data.feather",
            target_column="custom_column",
        )
        config._explicitly_set_fields = {"target_column"}
        
        with pytest.raises(ValueError, match="Cannot specify target_column when preprocessed_data_path is set"):
            config.validate()

    def test_to_dict_with_preprocessed_data(self):
        """Test to_dict with preprocessed data path."""
        config = DataPreprocessingConfig(preprocessed_data_path="data.feather")
        result = config.to_dict()
        assert result == {"preprocessed_data_path": "data.feather"}

    def test_to_dict_without_preprocessed_data(self):
        """Test to_dict without preprocessed data path."""
        config = DataPreprocessingConfig(
            target_column="custom_column",
            drop_target_column=True,
            pandas_score_filter="score > 0.5",
        )
        result = config.to_dict()
        expected = {
            "target_column": "custom_column",
            "drop_target_column": True,
            "pandas_score_filter": "score > 0.5",
            "splitting": {"type": "None"},
            "scoring": {"type": "None"},
        }
        assert result == expected

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "target_column": "custom_column",
            "drop_target_column": True,
            "pandas_score_filter": "score > 0.5",
            "splitting": {"type": "ParagraphSplit"},
            "scoring": {"type": "KeywordScorer", "keywords": ["test"]},
        }
        config = DataPreprocessingConfig.from_dict(data)
        assert config.target_column == "custom_column"
        assert config.drop_target_column is True
        assert config.pandas_score_filter == "score > 0.5"
        assert isinstance(config.splitting.strategy, ParagraphSplit)
        assert isinstance(config.scoring.scorer, KeywordScorer)
        assert config._explicitly_set_fields == set(data.keys())


class TestSchemaConfig:
    """Test schema configuration."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        config = SchemaConfig()
        assert config.spec_path == DEFAULT_SCHEMA_PATH
        assert config.prompt_template == DEFAULT_PROMPT_TEMPLATE
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        config = SchemaConfig(
            spec_path="custom_schema.yaml",
            prompt_template="Custom template: {data}",
            system_prompt="Custom system prompt",
        )
        assert config.spec_path == "custom_schema.yaml"
        assert config.prompt_template == "Custom template: {data}"
        assert config.system_prompt == "Custom system prompt"

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: data")
            temp_path = f.name
        
        try:
            config = SchemaConfig(spec_path=temp_path)
            config.validate()  # Should not raise
        finally:
            Path(temp_path).unlink()

    def test_validate_invalid_spec_path(self):
        """Test validation with invalid spec_path."""
        config = SchemaConfig(spec_path="")
        with pytest.raises(ValueError, match="spec_path must be a valid Path or string"):
            config.validate()

        config = SchemaConfig(spec_path=123)
        with pytest.raises(ValueError, match="spec_path must be a valid Path or string"):
            config.validate()

    def test_validate_nonexistent_file(self):
        """Test validation with nonexistent file."""
        config = SchemaConfig(spec_path="nonexistent.yaml")
        with pytest.raises(ValueError, match="Schema spec file does not exist"):
            config.validate()

    def test_validate_invalid_prompt_template(self):
        """Test validation with invalid prompt_template."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: data")
            temp_path = f.name
        
        try:
            config = SchemaConfig(spec_path=temp_path, prompt_template=123)
            with pytest.raises(ValueError, match="prompt_template must be a string"):
                config.validate()
        finally:
            Path(temp_path).unlink()

    def test_validate_invalid_system_prompt(self):
        """Test validation with invalid system_prompt."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: data")
            temp_path = f.name
        
        try:
            config = SchemaConfig(spec_path=temp_path, system_prompt=123)
            with pytest.raises(ValueError, match="system_prompt must be a string"):
                config.validate()
        finally:
            Path(temp_path).unlink()

    def test_to_dict(self):
        """Test to_dict method."""
        config = SchemaConfig(
            spec_path="test_schema.yaml",
            prompt_template="Test template",
            system_prompt="Test system prompt",
        )
        result = config.to_dict()
        expected = {
            "spec_path": "test_schema.yaml",
            "prompt_template": "Test template",
            "system_prompt": "Test system prompt",
        }
        assert result == expected

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "spec_path": "test_schema.yaml",
            "prompt_template": "Test template",
            "system_prompt": "Test system prompt",
        }
        config = SchemaConfig.from_dict(data)
        assert config.spec_path == Path("test_schema.yaml")
        assert config.prompt_template == "Test template"
        assert config.system_prompt == "Test system prompt"

    def test_from_dict_none(self):
        """Test from_dict with None."""
        config = SchemaConfig.from_dict(None)
        assert config.spec_path == Path("")
        assert config.prompt_template == DEFAULT_PROMPT_TEMPLATE
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT


class TestSemanticCacheConfig:
    """Test semantic cache configuration."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        config = SemanticCacheConfig()
        assert config.backend == DEFAULT_SEMANTIC_CACHE_BACKEND
        assert config.path == DEFAULT_SEMANTIC_CACHE_PATH
        assert config.max_size_mb == DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB
        assert config.synchronous == DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        config = SemanticCacheConfig(
            backend="lmdb",
            path="/custom/cache/path",
            max_size_mb=500,
            synchronous="full",
        )
        assert config.backend == "lmdb"
        assert config.path == "/custom/cache/path"
        assert config.max_size_mb == 500
        assert config.synchronous == "full"

    def test_resolve_path(self):
        """Test resolve_path method."""
        config = SemanticCacheConfig(path="~/cache")
        resolved = config.resolve_path()
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = SemanticCacheConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_backend(self):
        """Test validation with invalid backend."""
        config = SemanticCacheConfig(backend="invalid")
        with pytest.raises(ValueError, match="cache.backend must be 'sqlite', 'lmdb', or 'filesystem'"):
            config.validate()

    def test_validate_invalid_max_size_mb(self):
        """Test validation with invalid max_size_mb."""
        config = SemanticCacheConfig(max_size_mb=0)
        with pytest.raises(ValueError, match="cache.max_size_mb must be a positive integer"):
            config.validate()

        config = SemanticCacheConfig(max_size_mb=-1)
        with pytest.raises(ValueError, match="cache.max_size_mb must be a positive integer"):
            config.validate()

        config = SemanticCacheConfig(max_size_mb="100")
        with pytest.raises(ValueError, match="cache.max_size_mb must be a positive integer"):
            config.validate()

    def test_validate_invalid_synchronous_sqlite(self):
        """Test validation with invalid synchronous for SQLite."""
        config = SemanticCacheConfig(backend="sqlite", synchronous="invalid")
        with pytest.raises(ValueError, match="cache.synchronous must be 'normal' or 'full' for SQLite"):
            config.validate()

    def test_validate_valid_synchronous_sqlite(self):
        """Test validation with valid synchronous for SQLite."""
        config = SemanticCacheConfig(backend="sqlite", synchronous="normal")
        config.validate()  # Should not raise

        config = SemanticCacheConfig(backend="sqlite", synchronous="full")
        config.validate()  # Should not raise

    def test_to_dict(self):
        """Test to_dict method."""
        config = SemanticCacheConfig(
            backend="lmdb",
            path="/custom/cache/path",
            max_size_mb=500,
            synchronous="full",
        )
        result = config.to_dict()
        expected = {
            "backend": "lmdb",
            "path": "/custom/cache/path",
            "max_size_mb": 500,
            "synchronous": "full",
        }
        assert result == expected

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "backend": "lmdb",
            "path": "/custom/cache/path",
            "max_size_mb": 500,
            "synchronous": "full",
        }
        config = SemanticCacheConfig.from_dict(data)
        assert config.backend == "lmdb"
        assert config.path == "/custom/cache/path"
        assert config.max_size_mb == 500
        assert config.synchronous == "full"

    def test_from_dict_none(self):
        """Test from_dict with None."""
        config = SemanticCacheConfig.from_dict(None)
        assert config.backend == DEFAULT_SEMANTIC_CACHE_BACKEND
        assert config.path == DEFAULT_SEMANTIC_CACHE_PATH
        assert config.max_size_mb == DEFAULT_SEMANTIC_CACHE_MAX_SIZE_MB
        assert config.synchronous == DEFAULT_SEMANTIC_CACHE_SYNCHRONOUS


class TestDELMConfig:
    """Test complete DELM configuration."""

    def test_initialization(self):
        """Test initialization."""
        llm_config = LLMExtractionConfig()
        data_config = DataPreprocessingConfig()
        schema_config = SchemaConfig()
        cache_config = SemanticCacheConfig()
        
        config = DELMConfig(
            llm_extraction=llm_config,
            data_preprocessing=data_config,
            schema=schema_config,
            semantic_cache=cache_config,
        )
        
        assert config.llm_extraction == llm_config
        assert config.data_preprocessing == data_config
        assert config.schema == schema_config
        assert config.semantic_cache == cache_config

    def test_validate(self):
        """Test validation."""
        llm_config = LLMExtractionConfig()
        data_config = DataPreprocessingConfig()
        schema_config = SchemaConfig()
        cache_config = SemanticCacheConfig()
        
        config = DELMConfig(
            llm_extraction=llm_config,
            data_preprocessing=data_config,
            schema=schema_config,
            semantic_cache=cache_config,
        )
        
        # Should not raise if all sub-configs are valid
        # Note: This will fail if schema.spec_path doesn't exist, so we'll mock it
        with patch.object(schema_config, 'validate'):
            config.validate()

    def test_to_serialized_config_dict(self):
        """Test to_serialized_config_dict method."""
        llm_config = LLMExtractionConfig(provider="openai", name="gpt-4")
        data_config = DataPreprocessingConfig(target_column="custom_column")
        schema_config = SchemaConfig(spec_path="test.yaml")
        cache_config = SemanticCacheConfig(backend="sqlite")
        
        config = DELMConfig(
            llm_extraction=llm_config,
            data_preprocessing=data_config,
            schema=schema_config,
            semantic_cache=cache_config,
        )
        
        result = config.to_serialized_config_dict()
        expected = {
            "llm_extraction": llm_config.to_dict(),
            "data_preprocessing": data_config.to_dict(),
            "schema": schema_config.to_dict(),
            "semantic_cache": cache_config.to_dict(),
        }
        assert result == expected

    def test_to_serialized_schema_spec_dict_yaml(self):
        """Test to_serialized_schema_spec_dict with YAML file."""
        schema_data = {"test": "data", "nested": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml.dump(schema_data, f)
            temp_path = f.name
        
        try:
            schema_config = SchemaConfig(spec_path=temp_path)
            config = DELMConfig(
                llm_extraction=LLMExtractionConfig(),
                data_preprocessing=DataPreprocessingConfig(),
                schema=schema_config,
                semantic_cache=SemanticCacheConfig(),
            )
            
            result = config.to_serialized_schema_spec_dict()
            assert result == schema_data
        finally:
            Path(temp_path).unlink()

    def test_to_serialized_schema_spec_dict_json(self):
        """Test to_serialized_schema_spec_dict with JSON file."""
        import json
        schema_data = {"test": "data", "nested": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as f:
            json.dump(schema_data, f)
            temp_path = f.name
        
        try:
            schema_config = SchemaConfig(spec_path=temp_path)
            config = DELMConfig(
                llm_extraction=LLMExtractionConfig(),
                data_preprocessing=DataPreprocessingConfig(),
                schema=schema_config,
                semantic_cache=SemanticCacheConfig(),
            )
            
            result = config.to_serialized_schema_spec_dict()
            assert result == schema_data
        finally:
            Path(temp_path).unlink()

    def test_to_serialized_schema_spec_dict_none_path(self):
        """Test to_serialized_schema_spec_dict with None path."""
        schema_config = SchemaConfig(spec_path=None)
        config = DELMConfig(
            llm_extraction=LLMExtractionConfig(),
            data_preprocessing=DataPreprocessingConfig(),
            schema=schema_config,
            semantic_cache=SemanticCacheConfig(),
        )
        
        with pytest.raises(ValueError, match="Schema spec path is None"):
            config.to_serialized_schema_spec_dict()

    def test_to_serialized_schema_spec_dict_nonexistent_file(self):
        """Test to_serialized_schema_spec_dict with nonexistent file."""
        schema_config = SchemaConfig(spec_path="nonexistent.yaml")
        config = DELMConfig(
            llm_extraction=LLMExtractionConfig(),
            data_preprocessing=DataPreprocessingConfig(),
            schema=schema_config,
            semantic_cache=SemanticCacheConfig(),
        )
        
        with pytest.raises(FileNotFoundError, match="Schema spec file does not exist"):
            config.to_serialized_schema_spec_dict()

    def test_to_serialized_schema_spec_dict_unsupported_format(self):
        """Test to_serialized_schema_spec_dict with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            schema_config = SchemaConfig(spec_path=temp_path)
            config = DELMConfig(
                llm_extraction=LLMExtractionConfig(),
                data_preprocessing=DataPreprocessingConfig(),
                schema=schema_config,
                semantic_cache=SemanticCacheConfig(),
            )
            
            with pytest.raises(ValueError, match="Unsupported schema file format"):
                config.to_serialized_schema_spec_dict()
        finally:
            Path(temp_path).unlink()

    def test_to_dict_alias(self):
        """Test to_dict method (alias for to_serialized_config_dict)."""
        config = DELMConfig(
            llm_extraction=LLMExtractionConfig(),
            data_preprocessing=DataPreprocessingConfig(),
            schema=SchemaConfig(),
            semantic_cache=SemanticCacheConfig(),
        )
        
        result = config.to_dict()
        expected = config.to_serialized_config_dict()
        assert result == expected

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "llm_extraction": {
                "provider": "openai",
                "name": "gpt-4",
            },
            "data_preprocessing": {
                "target_column": "custom_column",
            },
            "schema": {
                "spec_path": "test.yaml",
            },
            "semantic_cache": {
                "backend": "sqlite",
            },
        }
        
        config = DELMConfig.from_dict(data)
        assert config.llm_extraction.provider == "openai"
        assert config.llm_extraction.name == "gpt-4"
        assert config.data_preprocessing.target_column == "custom_column"
        assert config.schema.spec_path == Path("test.yaml")
        assert config.semantic_cache.backend == "sqlite"

    def test_from_dict_none(self):
        """Test from_dict with None."""
        config = DELMConfig.from_dict(None)
        assert isinstance(config.llm_extraction, LLMExtractionConfig)
        assert isinstance(config.data_preprocessing, DataPreprocessingConfig)
        assert isinstance(config.schema, SchemaConfig)
        assert isinstance(config.semantic_cache, SemanticCacheConfig)

    def test_from_yaml(self):
        """Test from_yaml method."""
        config_data = {
            "llm_extraction": {
                "provider": "anthropic",
                "name": "claude-3-sonnet",
            },
            "data_preprocessing": {
                "target_column": "text_column",
            },
            "schema": {
                "spec_path": "schema.yaml",
            },
            "semantic_cache": {
                "backend": "lmdb",
            },
        }
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = DELMConfig.from_yaml(Path(temp_path))
            assert config.llm_extraction.provider == "anthropic"
            assert config.llm_extraction.name == "claude-3-sonnet"
            assert config.data_preprocessing.target_column == "text_column"
            assert config.schema.spec_path == Path("schema.yaml")
            assert config.semantic_cache.backend == "lmdb"
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_nonexistent_file(self):
        """Test from_yaml with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="YAML config file does not exist"):
            DELMConfig.from_yaml(Path("nonexistent.yaml"))

    def test_from_any_delm_config(self):
        """Test from_any with DELMConfig instance."""
        original_config = DELMConfig(
            llm_extraction=LLMExtractionConfig(),
            data_preprocessing=DataPreprocessingConfig(),
            schema=SchemaConfig(),
            semantic_cache=SemanticCacheConfig(),
        )
        
        result = DELMConfig.from_any(original_config)
        assert result is original_config

    def test_from_any_dict(self):
        """Test from_any with dictionary."""
        data = {
            "llm_extraction": {"provider": "openai"},
            "data_preprocessing": {"target_column": "text"},
            "schema": {"spec_path": "schema.yaml"},
            "semantic_cache": {"backend": "sqlite"},
        }
        
        config = DELMConfig.from_any(data)
        assert isinstance(config, DELMConfig)
        assert config.llm_extraction.provider == "openai"

    def test_from_any_yaml_path(self):
        """Test from_any with YAML file path."""
        config_data = {
            "llm_extraction": {"provider": "anthropic"},
            "data_preprocessing": {"target_column": "text"},
            "schema": {"spec_path": "schema.yaml"},
            "semantic_cache": {"backend": "lmdb"},
        }
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = DELMConfig.from_any(temp_path)
            assert isinstance(config, DELMConfig)
            assert config.llm_extraction.provider == "anthropic"
        finally:
            Path(temp_path).unlink()

    def test_from_any_invalid_type(self):
        """Test from_any with invalid type."""
        with pytest.raises(ValueError, match="config must be a DELMConfig, dict, or path to YAML"):
            DELMConfig.from_any(123) 