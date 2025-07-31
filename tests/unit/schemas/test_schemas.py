"""
Unit tests for DELM schemas.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel
from typing import Dict, Any, List

from delm.schemas.schemas import (
    BaseSchema, SimpleSchema, NestedSchema, MultipleSchema, SchemaRegistry,
    _make_enum, _ann_and_field, _validate_type_safe
)
from delm.models import ExtractionVariable


class TestUtilities:
    """Test utility functions."""
    
    def test_make_enum(self):
        """Test enum creation with safe names."""
        enum = _make_enum("TestEnum", ["value 1", "value-2", "value3"])
        assert enum.value_1.value == "value 1"
        assert enum.value_2.value == "value-2"
        assert enum.value3.value == "value3"
    
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


class TestBaseSchema:
    """Test the abstract base class."""
    
    def test_abstract_methods(self):
        """Test that BaseSchema is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSchema({})


class TestSimpleSchema:
    """Test the SimpleSchema class."""
    
    def test_initialization(self):
        """Test SimpleSchema initialization."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                },
                {
                    "name": "tags",
                    "description": "The tags",
                    "data_type": "[string]",
                    "required": False
                }
            ]
        }
        
        schema = SimpleSchema(config)
        assert len(schema.variables) == 2
        assert schema.variables[0].name == "title"
        assert schema.variables[1].name == "tags"
        assert schema._list_vars == ["tags"]
    
    def test_variables_property(self):
        """Test variables property."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = SimpleSchema(config)
        variables = schema.variables
        assert len(variables) == 1
        assert isinstance(variables[0], ExtractionVariable)
        assert variables[0].name == "title"
    
    def test_create_pydantic_schema(self):
        """Test Pydantic schema creation."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                },
                {
                    "name": "count",
                    "description": "The count",
                    "data_type": "integer",
                    "required": False
                }
            ]
        }
        
        schema = SimpleSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        assert issubclass(pydantic_schema, BaseModel)
        assert "title" in pydantic_schema.__annotations__
        assert "count" in pydantic_schema.__annotations__
    
    def test_create_prompt(self):
        """Test prompt creation."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
    
        schema = SimpleSchema(config)
        prompt_template = "Extract from: {text}\nVariables:\n{variables}\nContext: {context}"
    
        result = schema.create_prompt("Sample text", prompt_template, {"key": "value"})
    
        assert "Sample text" in result
        assert "title" in result
        assert "The title" in result
        assert "{'key': 'value'}" in result
    
    def test_get_variables_text(self):
        """Test variables text generation."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                },
                {
                    "name": "tags",
                    "description": "The tags",
                    "data_type": "[string]",
                    "required": False,
                    "allowed_values": ["tag1", "tag2"]
                }
            ]
        }
        
        schema = SimpleSchema(config)
        text = schema.get_variables_text()
        
        assert "title: The title (string) [REQUIRED]" in text
        assert "tags: The tags ([string])" in text
        assert "allowed values: \"tag1\", \"tag2\"" in text
    
    def test_validate_and_parse_response_to_dict_valid(self):
        """Test response validation and parsing with valid data."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = SimpleSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        # Create a valid response
        response = pydantic_schema(title="Test Title")
        
        result = schema.validate_and_parse_response_to_dict(response, "Sample text with Test Title")
        
        assert result == {"title": "Test Title"}
    
    def test_validate_and_parse_response_to_dict_invalid(self):
        """Test response validation and parsing with invalid data."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = SimpleSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        # Create an invalid response (missing required field)
        response = pydantic_schema(title=None)
        
        result = schema.validate_and_parse_response_to_dict(response, "Sample text")
        
        assert result == {}
    
    def test_is_valid_json_dict_valid(self):
        """Test JSON dict validation with valid data."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                },
                {
                    "name": "count",
                    "description": "The count",
                    "data_type": "integer",
                    "required": False
                }
            ]
        }
        
        schema = SimpleSchema(config)
        data = {"title": "Test Title", "count": 42}
        
        assert schema.is_valid_json_dict(data) is True
    
    def test_is_valid_json_dict_invalid_missing_required(self):
        """Test JSON dict validation with missing required field."""
        config = {
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = SimpleSchema(config)
        data = {}  # Missing required title
        
        assert schema.is_valid_json_dict(data) is False
    
    def test_is_valid_json_dict_invalid_wrong_type(self):
        """Test JSON dict validation with wrong type."""
        config = {
            "variables": [
                {
                    "name": "count",
                    "description": "The count",
                    "data_type": "integer",
                    "required": True
                }
            ]
        }
        
        schema = SimpleSchema(config)
        data = {"count": "not an integer"}
        
        assert schema.is_valid_json_dict(data) is False
    
    def test_is_valid_json_dict_list_type(self):
        """Test JSON dict validation with list types."""
        config = {
            "variables": [
                {
                    "name": "tags",
                    "description": "The tags",
                    "data_type": "[string]",
                    "required": True
                }
            ]
        }
        
        schema = SimpleSchema(config)
        
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
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        assert schema.container_name == "books"
        assert len(schema.variables) == 1
        assert schema.variables[0].name == "title"
    
    def test_container_name_property(self):
        """Test container_name property."""
        config = {
            "container_name": "custom_container",
            "variables": []
        }
        
        schema = NestedSchema(config)
        assert schema.container_name == "custom_container"
    
    def test_create_pydantic_schema(self):
        """Test Pydantic schema creation."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        assert issubclass(pydantic_schema, BaseModel)
        assert "books" in pydantic_schema.__annotations__
    
    def test_create_prompt(self):
        """Test prompt creation."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        prompt_template = "Extract from: {text}\nVariables:\n{variables}\nContext: {context}"
        
        result = schema.create_prompt("Sample text", prompt_template, {"key": "value"})
        
        assert "Sample text" in result
        assert "title" in result
        assert "key: value" in result
    
    def test_validate_and_parse_response_to_dict_valid(self):
        """Test response validation and parsing with valid data."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        # Create a valid response - the container expects a list of dicts, not Pydantic models
        items = [{"title": "Book 1"}, {"title": "Book 2"}]
        response = pydantic_schema(books=items)
        
        result = schema.validate_and_parse_response_to_dict(response, "Sample text with Book 1 and Book 2")
        
        assert "books" in result
        assert len(result["books"]) == 2
        assert result["books"][0]["title"] == "Book 1"
        assert result["books"][1]["title"] == "Book 2"
    
    def test_validate_and_parse_response_to_dict_invalid(self):
        """Test response validation and parsing with invalid data."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        # Create an invalid response (empty list)
        response = pydantic_schema(books=[])
        
        result = schema.validate_and_parse_response_to_dict(response, "Sample text")
        
        assert result == {}
    
    def test_is_valid_json_dict_valid(self):
        """Test JSON dict validation with valid data."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        data = {
            "books": [
                {"title": "Book 1"},
                {"title": "Book 2"}
            ]
        }
        
        assert schema.is_valid_json_dict(data) is True
    
    def test_is_valid_json_dict_invalid_missing_container(self):
        """Test JSON dict validation with missing container."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        data = {}  # Missing books container
        
        assert schema.is_valid_json_dict(data) is False
    
    def test_is_valid_json_dict_invalid_not_list(self):
        """Test JSON dict validation with non-list container."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        data = {"books": "not a list"}
        
        assert schema.is_valid_json_dict(data) is False
    
    def test_is_valid_json_dict_with_override_container_name(self):
        """Test JSON dict validation with override container name."""
        config = {
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = NestedSchema(config)
        data = {
            "custom_container": [
                {"title": "Book 1"}
            ]
        }
        
        assert schema.is_valid_json_dict(data, override_container_name="custom_container") is True


class TestMultipleSchema:
    """Test the MultipleSchema class."""
    
    def test_initialization(self):
        """Test MultipleSchema initialization."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            },
            "nested_schema": {
                "schema_type": "nested",
                "container_name": "books",
                "variables": [
                    {
                        "name": "author",
                        "description": "The author",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        assert len(schema.schemas) == 2
        assert "simple_schema" in schema.schemas
        assert "nested_schema" in schema.schemas
        assert isinstance(schema.schemas["simple_schema"], SimpleSchema)
        assert isinstance(schema.schemas["nested_schema"], NestedSchema)
    
    def test_variables_property(self):
        """Test variables property combines all sub-schemas."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            },
            "nested_schema": {
                "schema_type": "nested",
                "container_name": "books",
                "variables": [
                    {
                        "name": "author",
                        "description": "The author",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        variables = schema.variables
        
        assert len(variables) == 2
        variable_names = [v.name for v in variables]
        assert "title" in variable_names
        assert "author" in variable_names
    
    def test_create_pydantic_schema(self):
        """Test Pydantic schema creation."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        assert issubclass(pydantic_schema, BaseModel)
        assert "simple_schema" in pydantic_schema.__annotations__
    
    def test_create_prompt(self):
        """Test prompt creation."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        prompt_template = "Extract from: {text}\nVariables:\n{variables}\nContext: {context}"
        
        result = schema.create_prompt("Sample text", prompt_template, {"key": "value"})
        
        assert "Sample text" in result
        assert "SIMPLE_SCHEMA" in result
        assert "title" in result
    
    def test_validate_and_parse_response_to_dict_simple(self):
        """Test response validation and parsing with simple sub-schema."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        # Create a valid response - pass the dict directly
        response = pydantic_schema(simple_schema={"title": "Test Title"})
        
        result = schema.validate_and_parse_response_to_dict(response, "Sample text with Test Title")
        
        assert "simple_schema" in result
        assert result["simple_schema"] == {"title": "Test Title"}
    
    def test_validate_and_parse_response_to_dict_nested(self):
        """Test response validation and parsing with nested sub-schema."""
        config = {
            "nested_schema": {
                "schema_type": "nested",
                "container_name": "books",
                "variables": [
                    {
                        "name": "author",
                        "description": "The author",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        pydantic_schema = schema.create_pydantic_schema()
        
        # Create a valid response - pass the dict directly
        response = pydantic_schema(nested_schema={"books": [{"author": "Author 1"}, {"author": "Author 2"}]})
        
        result = schema.validate_and_parse_response_to_dict(response, "Sample text with Author 1 and Author 2")
        
        assert "nested_schema" in result
        assert len(result["nested_schema"]) == 2
        assert result["nested_schema"][0]["author"] == "Author 1"
        assert result["nested_schema"][1]["author"] == "Author 2"
    
    def test_is_valid_json_dict_valid(self):
        """Test JSON dict validation with valid data."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            },
            "nested_schema": {
                "schema_type": "nested",
                "container_name": "books",
                "variables": [
                    {
                        "name": "author",
                        "description": "The author",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
    
        schema = MultipleSchema(config)
        data = {
            "simple_schema": {"title": "Test Title"},
            "nested_schema": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    
        assert schema.is_valid_json_dict(data) is True
    
    def test_is_valid_json_dict_invalid_missing_key(self):
        """Test JSON dict validation with missing key."""
        config = {
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = MultipleSchema(config)
        data = {}  # Missing simple_schema key
        
        assert schema.is_valid_json_dict(data) is False


class TestSchemaRegistry:
    """Test the SchemaRegistry class."""
    
    def test_initialization(self):
        """Test SchemaRegistry initialization."""
        registry = SchemaRegistry()
        assert len(registry._reg) >= 3
        assert "simple" in registry._reg
        assert "nested" in registry._reg
        assert "multiple" in registry._reg
    
    def test_register(self):
        """Test registering a custom schema type."""
        registry = SchemaRegistry()
        
        class CustomSchema(BaseSchema):
            def __init__(self, config):
                pass
            
            @property
            def variables(self):
                return []
            
            def create_pydantic_schema(self):
                return type("CustomSchema", (BaseModel,), {})
            
            def create_prompt(self, text, prompt_template, context=None):
                return prompt_template.format(text=text, variables="", context=context or "")
            
            def validate_and_parse_response_to_dict(self, response, text_chunk):
                return {}
            
            def is_valid_json_dict(self, data, path="root", override_container_name=None):
                return True
        
        registry.register("custom", CustomSchema)
        assert "custom" in registry._reg
        assert registry._reg["custom"] == CustomSchema
    
    def test_create_simple(self):
        """Test creating a simple schema."""
        registry = SchemaRegistry()
        config = {
            "schema_type": "simple",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = registry.create(config)
        assert isinstance(schema, SimpleSchema)
        assert len(schema.variables) == 1
    
    def test_create_nested(self):
        """Test creating a nested schema."""
        registry = SchemaRegistry()
        config = {
            "schema_type": "nested",
            "container_name": "books",
            "variables": [
                {
                    "name": "title",
                    "description": "The title",
                    "data_type": "string",
                    "required": True
                }
            ]
        }
        
        schema = registry.create(config)
        assert isinstance(schema, NestedSchema)
        assert schema.container_name == "books"
    
    def test_create_multiple(self):
        """Test creating a multiple schema."""
        registry = SchemaRegistry()
        config = {
            "schema_type": "multiple",
            "simple_schema": {
                "schema_type": "simple",
                "variables": [
                    {
                        "name": "title",
                        "description": "The title",
                        "data_type": "string",
                        "required": True
                    }
                ]
            }
        }
        
        schema = registry.create(config)
        assert isinstance(schema, MultipleSchema)
        assert len(schema.schemas) == 1
    
    def test_create_unknown_type(self):
        """Test creating an unknown schema type."""
        registry = SchemaRegistry()
        config = {"schema_type": "unknown"}
        
        with pytest.raises(ValueError, match="Unknown schema_type"):
            registry.create(config)
    
    def test_list_available(self):
        """Test listing available schema types."""
        registry = SchemaRegistry()
        available = registry.list_available()
        
        assert isinstance(available, list)
        assert "simple" in available
        assert "nested" in available
        assert "multiple" in available 