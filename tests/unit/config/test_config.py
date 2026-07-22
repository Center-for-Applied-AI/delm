"""
Unit tests for DELM configuration module.

NOTE: Most tests have been removed during API redesign.
"""

import pytest

from delm import DELMConfig, Schema
from delm.models import ExtractionVariable


# ============================================================================
# INTERNAL CONFIG CLASSES - TESTS DELETED
# ============================================================================
# The following test classes have been deleted because they test internal
# implementation details (LLMExtractionConfig, DataPreprocessingConfig,
# SemanticCacheConfig) that now require explicit parameters with no defaults.
#
# These classes are not part of the public API. Users interact with DELM via:
#   DELM(provider=..., model=..., temperature=..., target_column=..., ...)
#
# The DELM class provides sensible defaults and constructs these internal
# config objects. Testing should focus on the DELM class API, not internal
# config structures.
#
# Deleted test classes:
# - TestLLMExtractionConfig (~190 lines)
# - TestDataPreprocessingConfig (~180 lines)
# - TestSemanticCacheConfig (~120 lines)
#
# Total lines deleted: ~490
#
# See FEATURES_TODO.md for more details.
# ============================================================================


# ============================================================================
# DELMConfig - NEEDS REWRITE
# ============================================================================
# TODO: DELMConfig tests need to be rewritten for the new API.
# The new DELMConfig takes flat parameters and constructs sub-configs internally.
#
# Key changes:
# - schema parameter is required (accepts Schema, str, Path, or dict)
# - All LLM/cache/preprocessing params are now flat parameters
# - No more nested config objects (SchemaConfig removed entirely)
#
# See DELM.__init__ in delm.py for the new parameter structure.
# See FEATURES_TODO.md for detailed rewrite requirements.
# ============================================================================


class TestConfigPlaceholder:
    """Placeholder test class to prevent pytest from skipping this file entirely."""

    def test_placeholder(self):
        """Placeholder test that always passes."""
        assert True


@pytest.fixture
def schema_dict():
    schema = Schema.simple(
        variables_list=[
            ExtractionVariable(
                name="test_field",
                description="A test field",
                data_type="string",
            )
        ]
    )
    return schema.to_dict()


class TestDELMConfigFromDictForwardCompatibility:
    """from_dict must tolerate configs saved by older versions (issue #61)."""

    def test_schema_only_dict_uses_constructor_defaults(self, schema_dict):
        config = DELMConfig.from_dict({"schema": schema_dict})
        llm_cfg = config.llm_extraction_cfg
        assert llm_cfg.provider == "openai"
        assert llm_cfg.model == "gpt-4o-mini"
        assert llm_cfg.temperature == 0.0
        assert llm_cfg.track_cost is True
        assert config.data_preprocessing_cfg.target_column == "text"
        assert config.semantic_cache_cfg.backend == "sqlite"

    def test_partial_dict_overrides_only_given_keys(self, schema_dict):
        config = DELMConfig.from_dict(
            {"schema": schema_dict, "model": "gpt-4o", "temperature": 0.5}
        )
        assert config.llm_extraction_cfg.model == "gpt-4o"
        assert config.llm_extraction_cfg.temperature == 0.5
        assert config.llm_extraction_cfg.provider == "openai"

    def test_missing_schema_raises_value_error(self):
        with pytest.raises(ValueError, match="must contain a 'schema' key"):
            DELMConfig.from_dict({"model": "gpt-4o"})

    def test_unknown_keys_are_ignored(self, schema_dict):
        config = DELMConfig.from_dict(
            {"schema": schema_dict, "some_future_field": 123}
        )
        assert config.llm_extraction_cfg.model == "gpt-4o-mini"

    def test_full_round_trip_still_works(self, schema_dict):
        config = DELMConfig.from_dict({"schema": schema_dict, "batch_size": 42})
        restored = DELMConfig.from_dict(config.to_dict())
        assert restored.llm_extraction_cfg.batch_size == 42
        restored.validate()
