"""
Unit tests for DELM main class.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from delm import DELM
from delm.config import (
    DELMConfig,
    LLMExtractionConfig,
    DataPreprocessingConfig,
    SchemaConfig,
    SemanticCacheConfig,
)


class TestDELMPreviewPrompt:
    """Test the DELM.preview_prompt method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DELMConfig."""
        config = Mock(spec=DELMConfig)

        # Mock data_preprocessing config
        data_preprocessing = Mock(spec=DataPreprocessingConfig)
        data_preprocessing.target_column = "text_column"
        config.data_preprocessing = data_preprocessing

        # Mock llm_extraction config
        llm_extraction = Mock(spec=LLMExtractionConfig)
        llm_extraction.provider = "openai"
        llm_extraction.name = "gpt-4"
        llm_extraction.track_cost = False
        llm_extraction.batch_size = 32
        config.llm_extraction = llm_extraction

        # Mock schema config
        schema = Mock(spec=SchemaConfig)
        schema.spec_path = "tests/unit/schemas/test_data/simple_schema.yaml"
        schema.prompt_template = "Extract the following from {text}: {fields}"
        schema.system_prompt = "You are a helpful assistant."
        config.schema = schema

        # Mock semantic_cache config
        semantic_cache = Mock(spec=SemanticCacheConfig)
        semantic_cache.backend = "none"
        config.semantic_cache = semantic_cache

        # Mock validate method
        config.validate = Mock()

        return config

    @pytest.fixture
    def mock_schema_manager(self):
        """Create a mock SchemaManager."""
        schema_manager = Mock()

        # Mock extraction schema with create_prompt method
        extraction_schema = Mock()
        extraction_schema.create_prompt = Mock(return_value="Mocked compiled prompt")
        schema_manager.extraction_schema = extraction_schema

        # Mock prompt_template
        schema_manager.prompt_template = "Extract the following from {text}: {fields}"

        return schema_manager

    def test_preview_prompt_with_text(self, mock_config, mock_schema_manager):
        """Test preview_prompt with custom text provided."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test with custom text
            custom_text = "This is my custom text for extraction"
            result = delm.preview_prompt(text=custom_text)

            # Verify create_prompt was called with the custom text
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text=custom_text,
                prompt_template=mock_schema_manager.prompt_template,
            )

            # Verify the result
            assert result == "Mocked compiled prompt"

    def test_preview_prompt_without_text(self, mock_config, mock_schema_manager):
        """Test preview_prompt without text (should use placeholder)."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test without text (should use placeholder)
            result = delm.preview_prompt()

            # Verify create_prompt was called with placeholder text
            expected_placeholder = f"<{mock_config.data_preprocessing.target_column}>"
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text=expected_placeholder,
                prompt_template=mock_schema_manager.prompt_template,
            )

            # Verify the result
            assert result == "Mocked compiled prompt"

    def test_preview_prompt_with_none_text(self, mock_config, mock_schema_manager):
        """Test preview_prompt with explicit None text."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test with explicit None
            result = delm.preview_prompt(text=None)

            # Verify create_prompt was called with placeholder text
            expected_placeholder = f"<{mock_config.data_preprocessing.target_column}>"
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text=expected_placeholder,
                prompt_template=mock_schema_manager.prompt_template,
            )

            # Verify the result
            assert result == "Mocked compiled prompt"

    def test_preview_prompt_with_empty_string(self, mock_config, mock_schema_manager):
        """Test preview_prompt with empty string (should use empty string, not placeholder)."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test with empty string
            result = delm.preview_prompt(text="")

            # Verify create_prompt was called with empty string (not placeholder)
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text="",
                prompt_template=mock_schema_manager.prompt_template,
            )

            # Verify the result
            assert result == "Mocked compiled prompt"

    def test_preview_prompt_with_multiline_text(self, mock_config, mock_schema_manager):
        """Test preview_prompt with multiline text."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test with multiline text
            multiline_text = """This is line 1
This is line 2
This is line 3"""
            result = delm.preview_prompt(text=multiline_text)

            # Verify create_prompt was called with multiline text
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text=multiline_text,
                prompt_template=mock_schema_manager.prompt_template,
            )

            # Verify the result
            assert result == "Mocked compiled prompt"

    def test_preview_prompt_with_special_characters(
        self, mock_config, mock_schema_manager
    ):
        """Test preview_prompt with special characters in text."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test with special characters
            special_text = "Text with special chars: @#$%^&*()_+-={}[]|\\:;<>?,./~`"
            result = delm.preview_prompt(text=special_text)

            # Verify create_prompt was called with special characters
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text=special_text,
                prompt_template=mock_schema_manager.prompt_template,
            )

            # Verify the result
            assert result == "Mocked compiled prompt"

    def test_preview_prompt_uses_correct_target_column(
        self, mock_config, mock_schema_manager
    ):
        """Test that preview_prompt uses the correct target column from config."""
        # Set a specific target column name
        mock_config.data_preprocessing.target_column = "my_custom_column"

        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            # Test without text - should use custom target column in placeholder
            result = delm.preview_prompt()

            # Verify create_prompt was called with correct placeholder
            expected_placeholder = "<my_custom_column>"
            mock_schema_manager.extraction_schema.create_prompt.assert_called_once_with(
                text=expected_placeholder,
                prompt_template=mock_schema_manager.prompt_template,
            )

    def test_preview_prompt_returns_string(self, mock_config, mock_schema_manager):
        """Test that preview_prompt returns a string."""
        with patch("delm.delm.DataProcessor"), patch(
            "delm.delm.SchemaManager", return_value=mock_schema_manager
        ), patch("delm.delm.DiskExperimentManager"), patch(
            "delm.delm.CostTracker"
        ), patch(
            "delm.delm.SemanticCacheFactory"
        ), patch(
            "delm.delm.ExtractionManager"
        ), patch(
            "delm.delm._configure_logging"
        ):

            delm = DELM(
                config=mock_config,
                experiment_name="test_experiment",
                experiment_directory=Path("/tmp/test_experiment"),
                override_logging=False,
            )

            result = delm.preview_prompt(text="Test text")

            # Verify result is a string
            assert isinstance(result, str)
            assert result == "Mocked compiled prompt"
