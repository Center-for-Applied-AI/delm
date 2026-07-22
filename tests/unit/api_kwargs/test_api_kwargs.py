"""
Unit tests for api_kwargs pass-through feature.

Verifies that api_kwargs flows from DELM -> DELMConfig -> LLMExtractionConfig
-> ExtractionManager -> instructor create_with_completion call.
"""

import pytest
from unittest.mock import patch, MagicMock

from delm import DELM, Schema, DELMConfig
from delm.config import LLMExtractionConfig
from delm.models import ExtractionVariable
from delm.utils.cost_tracker import get_tokenizer_for_model


@pytest.fixture
def simple_schema():
    return Schema.simple(
        variables_list=[
            ExtractionVariable(
                name="test_field",
                description="A test field",
                data_type="string",
            )
        ]
    )


class TestApiKwargsConfig:
    """Test api_kwargs in LLMExtractionConfig."""

    def test_default_api_kwargs_is_empty_dict(self, simple_schema):
        config = DELMConfig(schema=simple_schema)
        assert config.llm_extraction_cfg.api_kwargs == {}

    def test_api_kwargs_set_via_config(self, simple_schema):
        config = DELMConfig(schema=simple_schema, api_kwargs={"store": False})
        assert config.llm_extraction_cfg.api_kwargs == {"store": False}

    def test_api_kwargs_multiple_keys(self, simple_schema):
        kwargs = {"store": False, "custom_header": "value", "timeout": 30}
        config = DELMConfig(schema=simple_schema, api_kwargs=kwargs)
        assert config.llm_extraction_cfg.api_kwargs == kwargs

    def test_api_kwargs_serialization_round_trip(self, simple_schema):
        kwargs = {"store": False, "extra_param": 42}
        config = DELMConfig(schema=simple_schema, api_kwargs=kwargs)
        config_dict = config.to_dict()
        assert config_dict["api_kwargs"] == kwargs

        restored = DELMConfig.from_dict(config_dict)
        assert restored.llm_extraction_cfg.api_kwargs == kwargs

    def test_api_kwargs_validation_rejects_non_dict(self, simple_schema):
        config = DELMConfig(schema=simple_schema, api_kwargs={"valid": True})
        config.llm_extraction_cfg.api_kwargs = "not_a_dict"
        with pytest.raises(ValueError, match="api_kwargs must be a dict"):
            config.validate()

    def test_api_kwargs_none_defaults_to_empty(self, simple_schema):
        config = DELMConfig(schema=simple_schema, api_kwargs=None)
        assert config.llm_extraction_cfg.api_kwargs == {}


class TestApiKwargsDELM:
    """Test api_kwargs flows through DELM class."""

    def test_delm_passes_api_kwargs_to_config(self, simple_schema):
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.InMemoryExperimentManager"
        ), patch("delm.delm.CostTracker"), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):
            delm = DELM(
                schema=simple_schema,
                api_kwargs={"store": False},
                override_logging=False,
            )
            assert delm.config.llm_extraction_cfg.api_kwargs == {"store": False}

    def test_delm_default_api_kwargs(self, simple_schema):
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.InMemoryExperimentManager"
        ), patch("delm.delm.CostTracker"), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):
            delm = DELM(
                schema=simple_schema,
                override_logging=False,
            )
            assert delm.config.llm_extraction_cfg.api_kwargs == {}


class TestApiKwargsExtractionManager:
    """Test api_kwargs is unpacked into the instructor API call."""

    def test_api_kwargs_passed_to_create_with_completion(self, simple_schema):
        from delm.core.extraction_manager import ExtractionManager

        model_config = LLMExtractionConfig(
            provider="openai",
            model="gpt-4o-mini",
            base_url=None,
            mode=None,
            temperature=0.0,
            prompt_template="Extract {variables} from {text}",
            system_prompt="You are helpful.",
            max_retries=1,
            batch_size=10,
            max_workers=1,
            base_delay=1.0,
            rate_limit_tokens=None,
            rate_limit_requests=None,
            rate_limit_period_seconds=60.0,
            track_cost=False,
            max_budget=None,
            model_input_cost_per_1M_tokens=None,
            model_output_cost_per_1M_tokens=None,
            max_completion_tokens=4096,
            api_kwargs={"store": False, "extra": "value"},
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_completion = MagicMock()
        mock_completion.usage = None
        mock_client.chat.completions.create_with_completion.return_value = (
            mock_response,
            mock_completion,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        mock_rate_limiter = MagicMock()

        with patch("delm.core.extraction_manager.instructor"), patch(
            "delm.core.extraction_manager.is_pydantic_model", return_value=True
        ):
            manager = ExtractionManager.__new__(ExtractionManager)
            manager.model_config = model_config
            manager.temperature = model_config.temperature
            manager.client = mock_client
            manager.extraction_schema = simple_schema.schema
            manager.concurrent_processor = MagicMock()
            manager.retry_handler = MagicMock()
            manager.prompt_template = model_config.prompt_template
            manager.system_prompt = model_config.system_prompt
            manager.few_shot_selector = None
            manager.tokenizer = get_tokenizer_for_model(model_config.model)
            manager.cost_tracker = None
            manager.semantic_cache = mock_cache
            manager.rate_limiter = mock_rate_limiter
            manager.max_output_tokens = 0

            manager.retry_handler.execute_with_retry.side_effect = lambda fn: fn()

            manager._instructor_extract_with_retry("test text chunk")

            call_kwargs = mock_client.chat.completions.create_with_completion.call_args
            assert call_kwargs.kwargs["store"] is False
            assert call_kwargs.kwargs["extra"] == "value"
