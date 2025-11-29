"""
Unit tests for DELM schemas.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel
from typing import Dict, Any, List

from delm.schemas.schemas import (
    ExtractionSchema,
    SimpleSchema,
    NestedSchema,
    MultipleSchema,
    _ann_and_field,
    _validate_type_safe,
)
from delm.models import ExtractionVariable


class TestUtilities:
    """Test utility functions."""

    def test_make_enum(self):
        """Test enum creation with safe names.

        NOTE: _make_enum function no longer exists in the new API.
        Enum handling is now done internally via allowed_values parameter.
        """
        pytest.skip(
            "_make_enum function no longer exists - enums handled internally via allowed_values"
        )

    def test_ann_and_field_scalar(self):
        """Test annotation and field creation for scalar types."""
        ann, field, is_list = _ann_and_field("string", True, "Test description")
        assert str(ann) == "typing.Optional[str]"
        assert field.description == "Test description"
        assert is_list is False

    def test_ann_and_field_list(self):
        """Test annotation and field creation for list types."""
        ann, field, is_list = _ann_and_field("[string]", True, "Test description")
        assert "List" in str(ann)
        assert "Optional" in str(ann)
        assert field.description == "Test description"
        assert is_list is True

    def test_validate_type_safe_valid(self):
        """Test type validation with valid types."""
        assert _validate_type_safe("test", "string", "test") is True
        assert _validate_type_safe(42, "integer", "test") is True
        assert _validate_type_safe(3.14, "number", "test") is True
        assert _validate_type_safe(True, "boolean", "test") is True

    def test_validate_type_safe_invalid(self):
        """Test type validation with invalid types."""
        assert _validate_type_safe(42, "string", "test") is False
        assert _validate_type_safe("test", "integer", "test") is False
        assert _validate_type_safe("test", "number", "test") is False
        assert _validate_type_safe("test", "boolean", "test") is False


class TestExtractionSchema:
    """Test the abstract base class."""

    def test_abstract_methods(self):
        """Test that ExtractionSchema is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ExtractionSchema({})


class TestSimpleSchema:
    """Test the SimpleSchema class."""

    def test_initialization(self):
        """Test SimpleSchema initialization."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            ),
            ExtractionVariable(
                name="tags",
                description="The tags",
                data_type="[string]",
                required=False,
            ),
        ]

        schema = SimpleSchema(variables)
        assert len(schema.variables) == 2
        assert schema.variables[0].name == "title"
        assert schema.variables[1].name == "tags"
        # Verify that tags is identified as a list type
        assert schema.variables[1].is_list()

    def test_variables_property(self):
        """Test variables property."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)
        variables_result = schema.variables
        assert len(variables_result) == 1
        assert isinstance(variables_result[0], ExtractionVariable)
        assert variables_result[0].name == "title"

    def test_create_pydantic_schema(self):
        """Test Pydantic schema creation."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            ),
            ExtractionVariable(
                name="count",
                description="The count",
                data_type="integer",
                required=False,
            ),
        ]

        schema = SimpleSchema(variables)
        pydantic_schema = schema.create_pydantic_schema()

        assert issubclass(pydantic_schema, BaseModel)
        assert "title" in pydantic_schema.__annotations__
        assert "count" in pydantic_schema.__annotations__

    def test_create_prompt(self):
        """Test prompt creation."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)
        prompt_template = (
            "Extract from: {text}\nVariables:\n{variables}\nContext: {context}"
        )

        result = schema.create_prompt("Sample text", prompt_template, {"key": "value"})

        assert "Sample text" in result
        assert "title" in result
        assert "The title" in result
        assert "{'key': 'value'}" in result

    def test_get_variables_text(self):
        """Test variables text generation."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            ),
            ExtractionVariable(
                name="tags",
                description="The tags",
                data_type="[string]",
                required=False,
                allowed_values=["tag1", "tag2"],
            ),
        ]

        schema = SimpleSchema(variables)
        text = schema.get_variables_text()

        assert "title: The title (string) [REQUIRED]" in text
        assert "tags: The tags ([string])" in text
        assert 'allowed values: "tag1", "tag2"' in text

    def test_validate_and_parse_response_to_dict_valid(self):
        """Test response validation and parsing with valid data."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)
        pydantic_schema = schema.create_pydantic_schema()

        # Create a valid response
        response = pydantic_schema(title="Test Title")

        result = schema.validate_and_parse_response_to_dict(
            response, "Sample text with Test Title"
        )

        assert result == {"title": "Test Title"}

    def test_validate_and_parse_response_to_dict_invalid(self):
        """Test response validation and parsing with invalid data."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)
        pydantic_schema = schema.create_pydantic_schema()

        # Create an invalid response (missing required field)
        response = pydantic_schema(title=None)

        result = schema.validate_and_parse_response_to_dict(response, "Sample text")

        assert result == {}

    def test_is_valid_json_dict_valid(self):
        """Test JSON dict validation with valid data."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            ),
            ExtractionVariable(
                name="count",
                description="The count",
                data_type="integer",
                required=False,
            ),
        ]

        schema = SimpleSchema(variables)
        data = {"title": "Test Title", "count": 42}

        assert schema.is_valid_json_dict(data) is True

    def test_is_valid_json_dict_invalid_missing_required(self):
        """Test JSON dict validation with missing required field."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)
        data = {}  # Missing required title

        assert schema.is_valid_json_dict(data) is False

    def test_is_valid_json_dict_invalid_wrong_type(self):
        """Test JSON dict validation with wrong type."""
        variables = [
            ExtractionVariable(
                name="count",
                description="The count",
                data_type="integer",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)
        data = {"count": "not an integer"}

        assert schema.is_valid_json_dict(data) is False

    def test_is_valid_json_dict_list_type(self):
        """Test JSON dict validation with list types."""
        variables = [
            ExtractionVariable(
                name="tags",
                description="The tags",
                data_type="[string]",
                required=True,
            )
        ]

        schema = SimpleSchema(variables)

        # Valid list
        data_valid = {"tags": ["tag1", "tag2"]}
        assert schema.is_valid_json_dict(data_valid) is True

        # Invalid - not a list
        data_invalid = {"tags": "not a list"}
        assert schema.is_valid_json_dict(data_invalid) is False


class TestNestedSchema:
    """Test the NestedSchema class."""

    def test_initialization(self):
        """Test NestedSchema initialization."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        assert schema.container_name == "books"
        assert len(schema.variables) == 1
        assert schema.variables[0].name == "title"

    def test_container_name_property(self):
        """Test container_name property."""
        schema = NestedSchema(container_name="custom_container", variables=[])
        assert schema.container_name == "custom_container"

    def test_create_pydantic_schema(self):
        """Test Pydantic schema creation."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        pydantic_schema = schema.create_pydantic_schema()

        assert issubclass(pydantic_schema, BaseModel)
        assert "books" in pydantic_schema.__annotations__

    def test_create_prompt(self):
        """Test prompt creation."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        prompt_template = (
            "Extract from: {text}\nVariables:\n{variables}\nContext: {context}"
        )

        result = schema.create_prompt("Sample text", prompt_template, {"key": "value"})

        assert "Sample text" in result
        assert "title" in result
        assert "key: value" in result

    def test_validate_and_parse_response_to_dict_valid(self):
        """Test response validation and parsing with valid data."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        pydantic_schema = schema.create_pydantic_schema()

        # Create a valid response - the container expects a list of dicts, not Pydantic models
        items = [{"title": "Book 1"}, {"title": "Book 2"}]
        response = pydantic_schema(books=items)

        result = schema.validate_and_parse_response_to_dict(
            response, "Sample text with Book 1 and Book 2"
        )

        assert "books" in result
        assert len(result["books"]) == 2
        assert result["books"][0]["title"] == "Book 1"
        assert result["books"][1]["title"] == "Book 2"

    def test_validate_and_parse_response_to_dict_invalid(self):
        """Test response validation and parsing with invalid data."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        pydantic_schema = schema.create_pydantic_schema()

        # Create an invalid response (empty list)
        response = pydantic_schema(books=[])

        result = schema.validate_and_parse_response_to_dict(response, "Sample text")

        assert result == {}

    def test_is_valid_json_dict_valid(self):
        """Test JSON dict validation with valid data."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        data = {"books": [{"title": "Book 1"}, {"title": "Book 2"}]}

        assert schema.is_valid_json_dict(data) is True

    def test_is_valid_json_dict_invalid_missing_container(self):
        """Test JSON dict validation with missing container."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        data = {}  # Missing books container

        assert schema.is_valid_json_dict(data) is False

    def test_is_valid_json_dict_invalid_not_list(self):
        """Test JSON dict validation with non-list container."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        data = {"books": "not a list"}

        assert schema.is_valid_json_dict(data) is False

    def test_is_valid_json_dict_with_override_container_name(self):
        """Test JSON dict validation with override container name."""
        variables = [
            ExtractionVariable(
                name="title",
                description="The title",
                data_type="string",
                required=True,
            )
        ]

        schema = NestedSchema(container_name="books", variables=variables)
        data = {"custom_container": [{"title": "Book 1"}]}

        assert (
            schema.is_valid_json_dict(data, override_container_name="custom_container")
            is True
        )


class TestMultipleSchema:
    """Test the MultipleSchema class."""

    def test_initialization(self):
        """Test MultipleSchema initialization."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            ),
            "nested_schema": NestedSchema(
                container_name="books",
                variables=[
                    ExtractionVariable(
                        name="author",
                        description="The author",
                        data_type="string",
                        required=True,
                    )
                ],
            ),
        }

        schema = MultipleSchema(schemas_dict)
        assert len(schema.schemas) == 2
        assert "simple_schema" in schema.schemas
        assert "nested_schema" in schema.schemas
        assert isinstance(schema.schemas["simple_schema"], SimpleSchema)
        assert isinstance(schema.schemas["nested_schema"], NestedSchema)

    def test_variables_property(self):
        """Test variables property combines all sub-schemas."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            ),
            "nested_schema": NestedSchema(
                container_name="books",
                variables=[
                    ExtractionVariable(
                        name="author",
                        description="The author",
                        data_type="string",
                        required=True,
                    )
                ],
            ),
        }

        schema = MultipleSchema(schemas_dict)
        variables = schema.variables

        assert len(variables) == 2
        variable_names = [v.name for v in variables]
        assert "title" in variable_names
        assert "author" in variable_names

    def test_create_pydantic_schema(self):
        """Test Pydantic schema creation."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            )
        }

        schema = MultipleSchema(schemas_dict)
        pydantic_schema = schema.create_pydantic_schema()

        assert issubclass(pydantic_schema, BaseModel)
        assert "simple_schema" in pydantic_schema.__annotations__

    def test_create_prompt(self):
        """Test prompt creation."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            )
        }

        schema = MultipleSchema(schemas_dict)
        prompt_template = (
            "Extract from: {text}\nVariables:\n{variables}\nContext: {context}"
        )

        result = schema.create_prompt("Sample text", prompt_template, {"key": "value"})

        assert "Sample text" in result
        assert "SIMPLE_SCHEMA" in result
        assert "title" in result

    def test_validate_and_parse_response_to_dict_simple(self):
        """Test response validation and parsing with simple sub-schema."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            )
        }

        schema = MultipleSchema(schemas_dict)
        pydantic_schema = schema.create_pydantic_schema()

        # Create a valid response - pass the dict directly
        response = pydantic_schema(simple_schema={"title": "Test Title"})

        result = schema.validate_and_parse_response_to_dict(
            response, "Sample text with Test Title"
        )

        assert "simple_schema" in result
        assert result["simple_schema"] == {"title": "Test Title"}

    def test_validate_and_parse_response_to_dict_nested(self):
        """Test response validation and parsing with nested sub-schema."""
        schemas_dict = {
            "nested_schema": NestedSchema(
                container_name="books",
                variables=[
                    ExtractionVariable(
                        name="author",
                        description="The author",
                        data_type="string",
                        required=True,
                    )
                ],
            )
        }

        schema = MultipleSchema(schemas_dict)
        pydantic_schema = schema.create_pydantic_schema()

        # Create a valid response - pass the dict directly
        response = pydantic_schema(
            nested_schema={"books": [{"author": "Author 1"}, {"author": "Author 2"}]}
        )

        result = schema.validate_and_parse_response_to_dict(
            response, "Sample text with Author 1 and Author 2"
        )

        assert "nested_schema" in result
        assert len(result["nested_schema"]) == 2
        assert result["nested_schema"][0]["author"] == "Author 1"
        assert result["nested_schema"][1]["author"] == "Author 2"

    def test_is_valid_json_dict_valid(self):
        """Test JSON dict validation with valid data."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            ),
            "nested_schema": NestedSchema(
                container_name="books",
                variables=[
                    ExtractionVariable(
                        name="author",
                        description="The author",
                        data_type="string",
                        required=True,
                    )
                ],
            ),
        }

        schema = MultipleSchema(schemas_dict)
        data = {
            "simple_schema": {"title": "Test Title"},
            "nested_schema": [{"author": "Author 1"}, {"author": "Author 2"}],
        }

        assert schema.is_valid_json_dict(data) is True

    def test_is_valid_json_dict_invalid_missing_key(self):
        """Test JSON dict validation with missing key."""
        schemas_dict = {
            "simple_schema": SimpleSchema(
                [
                    ExtractionVariable(
                        name="title",
                        description="The title",
                        data_type="string",
                        required=True,
                    )
                ]
            )
        }

        schema = MultipleSchema(schemas_dict)
        data = {}  # Missing simple_schema key

        assert schema.is_valid_json_dict(data) is False
