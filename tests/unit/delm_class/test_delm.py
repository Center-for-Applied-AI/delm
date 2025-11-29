"""
Unit tests for DELM main class.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from delm import DELM, Schema
from delm.models import ExtractionVariable


class TestDELMPreviewPrompt:
    """Test the DELM.preview_prompt method."""

    @pytest.fixture
    def simple_schema(self):
        """Create a simple schema for testing."""
        return Schema.simple(
            variables_list=[
                ExtractionVariable(
                    name="test_field",
                    description="A test field",
                    data_type="string",
                )
            ]
        )

    def test_preview_prompt_with_text(self, simple_schema):
        """Test preview_prompt with custom text provided."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            # Test with custom text
            custom_text = "This is my custom text for extraction"
            result = delm.preview_prompt(text=custom_text)

            # Verify the result is a string and contains the text
            assert isinstance(result, str)
            assert "test_field" in result
            assert custom_text in result

    def test_preview_prompt_without_text(self, simple_schema):
        """Test preview_prompt without text (should use placeholder)."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            # Test without text (should use placeholder)
            result = delm.preview_prompt()

            # Verify the result contains placeholder
            assert isinstance(result, str)
            assert "<text_column>" in result

    def test_preview_prompt_with_none_text(self, simple_schema):
        """Test preview_prompt with explicit None text."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            # Test with explicit None
            result = delm.preview_prompt(text=None)

            # Verify the result contains placeholder
            assert isinstance(result, str)
            assert "<text_column>" in result

    def test_preview_prompt_with_empty_string(self, simple_schema):
        """Test preview_prompt with empty string (should use empty string, not placeholder)."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            # Test with empty string
            result = delm.preview_prompt(text="")

            # Verify the result is a string
            assert isinstance(result, str)
            # Should not contain placeholder when empty string provided
            assert "<text_column>" not in result or result == ""

    def test_preview_prompt_with_multiline_text(self, simple_schema):
        """Test preview_prompt with multiline text."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            # Test with multiline text
            multiline_text = """This is line 1
This is line 2
This is line 3"""
            result = delm.preview_prompt(text=multiline_text)

            # Verify the result contains the multiline text
            assert isinstance(result, str)
            assert "This is line 1" in result
            assert "This is line 2" in result
            assert "This is line 3" in result

    def test_preview_prompt_with_special_characters(self, simple_schema):
        """Test preview_prompt with special characters in text."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            # Test with special characters
            special_text = "Text with special chars: @#$%^&*()_+-={}[]|\\:;<>?,./~`"
            result = delm.preview_prompt(text=special_text)

            # Verify the result contains special characters
            assert isinstance(result, str)
            assert "@#$%" in result or special_text in result

    def test_preview_prompt_uses_correct_target_column(self, simple_schema):
        """Test that preview_prompt uses the correct target column from config."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="my_custom_column",
                override_logging=False,
            )

            # Test without text - should use custom target column in placeholder
            result = delm.preview_prompt()

            # Verify placeholder uses correct column name
            assert isinstance(result, str)
            assert "<my_custom_column>" in result

    def test_preview_prompt_returns_string(self, simple_schema):
        """Test that preview_prompt returns a string."""
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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                override_logging=False,
            )

            result = delm.preview_prompt(text="Test text")

            # Verify result is a string
            assert isinstance(result, str)
            assert len(result) > 0

    def test_preview_prompt_with_custom_prompt_template(self, simple_schema):
        """Test preview_prompt with custom prompt template."""
        custom_template = "Custom template: Extract {variables} from:\n{text}"

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
                provider="openai",
                model="gpt-4o-mini",
                target_column="text_column",
                prompt_template=custom_template,
                override_logging=False,
            )

            result = delm.preview_prompt(text="Test text")

            # Verify custom template is used
            assert isinstance(result, str)
            assert "Custom template" in result or "Extract" in result
