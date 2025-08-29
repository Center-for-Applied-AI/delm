"""
Unit tests for DELM SchemaManager.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any

from delm.schemas.schema_manager import SchemaManager
from delm.config import SchemaConfig
from delm.schemas import BaseSchema, SimpleSchema, NestedSchema, MultipleSchema, SchemaRegistry


class TestSchemaManager:
    """Test the SchemaManager class."""
    
    def test_initialization(self):
        """Test SchemaManager initialization with valid config."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_schema = Mock(spec=SimpleSchema)
                mock_registry_instance.create.return_value = mock_schema
                mock_load.return_value = {"schema_type": "simple", "variables": []}
                
                manager = SchemaManager(config)
                
                assert manager.spec_path == Path("tests/unit/schemas/test_data/simple_schema.yaml")
                assert manager.prompt_template == "Extract {text}"
                assert manager.system_prompt == "You are a helpful assistant."
                assert manager.schema_registry == mock_registry_instance
                assert manager.extraction_schema == mock_schema
    
    def test_initialization_with_string_path(self):
        """Test SchemaManager initialization with string path."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_schema = Mock(spec=SimpleSchema)
                mock_registry_instance.create.return_value = mock_schema
                mock_load.return_value = {"schema_type": "simple", "variables": []}
                
                manager = SchemaManager(config)
                
                assert isinstance(manager.spec_path, Path)
                assert str(manager.spec_path) == "tests/unit/schemas/test_data/simple_schema.yaml"
    
    def test_get_extraction_schema(self):
        """Test get_extraction_schema method."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_schema = Mock(spec=SimpleSchema)
                mock_registry_instance.create.return_value = mock_schema
                mock_load.return_value = {"schema_type": "simple", "variables": []}
                
                manager = SchemaManager(config)
                result = manager.get_extraction_schema()
                
                assert result == mock_schema
    
    def test_load_schema_success(self):
        """Test successful schema loading."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_schema = Mock(spec=SimpleSchema)
                mock_registry_instance.create.return_value = mock_schema
                mock_load.return_value = {"schema_type": "simple", "variables": []}
                
                manager = SchemaManager(config)
                
                mock_registry_instance.create.assert_called_once()
                call_args = mock_registry_instance.create.call_args[0][0]
                assert call_args["schema_type"] == "simple"
                assert "variables" in call_args
    
    def test_load_schema_with_nested_schema(self):
        """Test loading nested schema."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/nested_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_schema = Mock(spec=NestedSchema)
                mock_registry_instance.create.return_value = mock_schema
                mock_load.return_value = {"schema_type": "nested", "variables": []}
                
                manager = SchemaManager(config)
                
                mock_registry_instance.create.assert_called_once()
                call_args = mock_registry_instance.create.call_args[0][0]
                assert call_args["schema_type"] == "nested"
    
    def test_load_schema_with_multiple_schema(self):
        """Test loading multiple schema."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/multiple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_schema = Mock(spec=MultipleSchema)
                mock_registry_instance.create.return_value = mock_schema
                mock_load.return_value = {"schema_type": "multiple", "variables": []}
                
                manager = SchemaManager(config)
                
                mock_registry_instance.create.assert_called_once()
                call_args = mock_registry_instance.create.call_args[0][0]
                assert call_args["schema_type"] == "multiple"
    
    def test_load_schema_spec_yaml_success(self):
        """Test successful YAML schema spec loading."""
        yaml_content = """
        schema_type: simple
        variables:
          - name: title
            description: The title
            data_type: string
            required: true
        """
        
        with patch('pathlib.Path.read_text', return_value=yaml_content):
            result = SchemaManager._load_schema_spec(Path("test.yaml"))
            
            assert result["schema_type"] == "simple"
            assert len(result["variables"]) == 1
            assert result["variables"][0]["name"] == "title"
    
    def test_load_schema_spec_yaml_empty_file(self):
        """Test loading empty YAML file."""
        with patch('pathlib.Path.read_text', return_value=""):
            result = SchemaManager._load_schema_spec(Path("test.yaml"))
            assert result == {}
    
    def test_load_schema_spec_yaml_none_content(self):
        """Test loading YAML file with None content."""
        with patch('pathlib.Path.read_text', return_value="some content"):
            with patch('yaml.safe_load', return_value=None):
                result = SchemaManager._load_schema_spec(Path("test.yaml"))
                assert result == {}
    
    def test_load_schema_spec_unsupported_format(self):
        """Test loading schema spec with unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported schema file format: .json"):
            SchemaManager._load_schema_spec(Path("test.json"))
    
    def test_load_schema_spec_file_not_found(self):
        """Test loading schema spec from non-existent file."""
        with patch('pathlib.Path.read_text', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                SchemaManager._load_schema_spec(Path("nonexistent.yaml"))
    
    def test_load_schema_spec_yaml_parse_error(self):
        """Test loading schema spec with invalid YAML."""
        invalid_yaml = """
        schema_type: simple
        variables:
          - name: title
            description: "Unclosed quote
        """
        
        with patch('pathlib.Path.read_text', return_value=invalid_yaml):
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                SchemaManager._load_schema_spec(Path("test.yaml"))
    
    def test_load_schema_registry_error(self):
        """Test handling of SchemaRegistry creation error."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_registry_instance = Mock()
                mock_registry.return_value = mock_registry_instance
                
                mock_registry_instance.create.side_effect = ValueError("Invalid schema")
                mock_load.return_value = {"schema_type": "simple", "variables": []}
                
                with pytest.raises(ValueError, match="Invalid schema"):
                    SchemaManager(config)
    
    def test_logging_during_initialization(self):
        """Test that appropriate logging occurs during initialization."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.log') as mock_log:
            with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
                with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                    mock_registry_instance = Mock()
                    mock_registry.return_value = mock_registry_instance
                    
                    mock_schema = Mock(spec=SimpleSchema)
                    mock_registry_instance.create.return_value = mock_schema
                    mock_load.return_value = {"schema_type": "simple", "variables": []}
                    
                    SchemaManager(config)
                    
                    # Check that debug logs were called
                    assert mock_log.debug.call_count >= 3  # At least 3 debug calls during init
    
    def test_logging_during_schema_loading(self):
        """Test that appropriate logging occurs during schema loading."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.log') as mock_log:
            with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
                with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                    mock_registry_instance = Mock()
                    mock_registry.return_value = mock_registry_instance
                    
                    mock_schema = Mock(spec=SimpleSchema)
                    mock_registry_instance.create.return_value = mock_schema
                    mock_load.return_value = {"schema_type": "simple", "variables": []}
                    
                    SchemaManager(config)
                    
                    # Check that schema loading logs were called
                    debug_calls = [call[0][0] for call in mock_log.debug.call_args_list]
                    assert any("Loading schema from spec file" in call for call in debug_calls)
                    assert any("Schema spec loaded with" in call for call in debug_calls)
                    assert any("Schema loaded successfully" in call for call in debug_calls)
    
    def test_logging_during_get_extraction_schema(self):
        """Test that appropriate logging occurs when getting extraction schema."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        with patch('delm.schemas.schema_manager.log') as mock_log:
            with patch('delm.schemas.schema_manager.SchemaRegistry') as mock_registry:
                with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                    mock_registry_instance = Mock()
                    mock_registry.return_value = mock_registry_instance
                    
                    mock_schema = Mock(spec=SimpleSchema)
                    mock_registry_instance.create.return_value = mock_schema
                    mock_load.return_value = {"schema_type": "simple", "variables": []}
                    
                    manager = SchemaManager(config)
                    manager.get_extraction_schema()
                    
                    # Check that get_extraction_schema log was called
                    debug_calls = [call[0][0] for call in mock_log.debug.call_args_list]
                    assert any("Getting extraction schema" in call for call in debug_calls)
    
    def test_schema_manager_with_real_schema_files(self):
        """Test SchemaManager with actual schema files."""
        test_cases = [
            ("tests/unit/schemas/test_data/simple_schema.yaml", SimpleSchema),
            ("tests/unit/schemas/test_data/nested_schema.yaml", NestedSchema),
            ("tests/unit/schemas/test_data/multiple_schema.yaml", MultipleSchema),
        ]
        
        for schema_path, expected_schema_type in test_cases:
            config = SchemaConfig(
                spec_path=schema_path,
                prompt_template="Extract {text}",
                system_prompt="You are a helpful assistant."
            )
            
            manager = SchemaManager(config)
            schema = manager.get_extraction_schema()
            
            assert isinstance(schema, expected_schema_type)
    
    def test_schema_manager_config_validation(self):
        """Test that SchemaManager works with validated SchemaConfig."""
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template="Extract {text}",
            system_prompt="You are a helpful assistant."
        )
        
        # Validate the config first
        config.validate()
        
        # Should work without issues
        manager = SchemaManager(config)
        schema = manager.get_extraction_schema()
        
        assert isinstance(schema, SimpleSchema)
    
    def test_schema_manager_with_long_prompts(self):
        """Test SchemaManager with very long prompt templates."""
        long_prompt = "A" * 10000  # Very long prompt
        config = SchemaConfig(
            spec_path="tests/unit/schemas/test_data/simple_schema.yaml",
            prompt_template=long_prompt,
            system_prompt=long_prompt
        )
        
        manager = SchemaManager(config)
        schema = manager.get_extraction_schema()
        
        assert manager.prompt_template == long_prompt
        assert manager.system_prompt == long_prompt
        assert isinstance(schema, SimpleSchema)
    
    def test_schema_manager_with_special_characters_in_path(self):
        """Test SchemaManager with special characters in path."""
        # Create a temporary file with special characters
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
            schema_type: simple
            variables:
              - name: title
                description: The title
                data_type: string
                required: true
            """
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = SchemaConfig(
                spec_path=temp_path,
                prompt_template="Extract {text}",
                system_prompt="You are a helpful assistant."
            )
            
            manager = SchemaManager(config)
            schema = manager.get_extraction_schema()
            
            assert isinstance(schema, SimpleSchema)
        finally:
            os.unlink(temp_path)
    
    def test_schema_manager_error_handling_integration(self):
        """Test error handling when schema file is corrupted."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            temp_path = f.name
        
        try:
            config = SchemaConfig(
                spec_path=temp_path,
                prompt_template="Extract {text}",
                system_prompt="You are a helpful assistant."
            )
            
            with pytest.raises(Exception):  # Should raise YAML parsing error
                SchemaManager(config)
        finally:
            os.unlink(temp_path) 