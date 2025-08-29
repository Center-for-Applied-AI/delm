"""
Unit tests for post_processing module.
"""

import pytest
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch

from delm.utils.post_processing import (
    merge_jsons_for_record,
    explode_json_results,
    _majority_vote
)
from delm.schemas.schemas import SimpleSchema, NestedSchema, MultipleSchema, ExtractionVariable
from delm.constants import SYSTEM_EXTRACTED_DATA_JSON_COLUMN


class TestMajorityVote:
    """Test the _majority_vote helper function."""
    
    def test_majority_vote_basic(self):
        """Test basic majority vote functionality."""
        values = ["A", "B", "A", "C", "A"]
        result = _majority_vote(values)
        assert result == "A"
    
    def test_majority_vote_tie(self):
        """Test majority vote with a tie - should return first winner."""
        values = ["A", "B", "A", "B"]
        result = _majority_vote(values)
        assert result == "A"  # First winner wins
    
    def test_majority_vote_empty_list(self):
        """Test majority vote with empty list."""
        result = _majority_vote([])
        assert result is None
    
    def test_majority_vote_single_value(self):
        """Test majority vote with single value."""
        values = ["A"]
        result = _majority_vote(values)
        assert result == "A"
    
    def test_majority_vote_numbers(self):
        """Test majority vote with numbers."""
        values = [1, 2, 1, 3, 1, 2]
        result = _majority_vote(values)
        assert result == 1


class TestMergeJsonsForRecord:
    """Test the merge_jsons_for_record function."""
    
    def setup_method(self):
        """Set up test schemas."""
        self.simple_schema = SimpleSchema({
            "variables": [
                {"name": "source", "data_type": "string", "required": True, "description": ""},
                {"name": "ratings", "data_type": "[integer]", "required": False, "description": ""},
                {"name": "price", "data_type": "number", "required": False, "description": ""},
            ]
        })
        
        self.nested_schema = NestedSchema({
            "container_name": "books",
            "variables": [
                {"name": "title", "data_type": "string", "required": True, "description": ""},
                {"name": "author", "data_type": "string", "required": True, "description": ""},
                {"name": "sales", "data_type": "[integer]", "required": False, "description": ""},
            ]
        })
        
        self.multiple_schema = MultipleSchema({
            "info": {
                "schema_type": "simple",
                "variables": [
                    {"name": "source", "data_type": "string", "required": True, "description": ""},
                    {"name": "ratings", "data_type": "[integer]", "required": False, "description": ""},
                ]
            },
            "books": {
                "schema_type": "nested",
                "container_name": "entries",
                "variables": [
                    {"name": "title", "data_type": "string", "required": True, "description": ""},
                    {"name": "author", "data_type": "string", "required": True, "description": ""},
                    {"name": "sales", "data_type": "[integer]", "required": False, "description": ""},
                ]
            }
        })
    
    def test_merge_simple_schema_scalars(self):
        """Test merging simple schema with scalar values."""
        input_jsons = [
            {"source": "A", "price": 100},
            {"source": "B", "price": 200},
            {"source": "B", "price": 200},
        ]
        merged = merge_jsons_for_record(input_jsons, self.simple_schema)
        
        assert merged["source"] == "B"  # Majority vote
        assert merged["price"] == 200   # Majority vote
        assert merged["ratings"] == []  # Empty list for missing field
    
    def test_merge_simple_schema_lists(self):
        """Test merging simple schema with list values."""
        input_jsons = [
            {"source": "A", "ratings": [1, 2]},
            {"source": "B", "ratings": [3, 4]},
            {"source": "A", "ratings": [5]},
        ]
        merged = merge_jsons_for_record(input_jsons, self.simple_schema)
        
        assert merged["source"] == "A"  # First winner in tie
        assert merged["ratings"] == [1, 2, 3, 4, 5]  # Concatenated lists
    
    def test_merge_simple_schema_mixed(self):
        """Test merging simple schema with mixed scalar and list values."""
        input_jsons = [
            {"source": "A", "price": 100, "ratings": [1]},
            {"source": "B", "price": 200, "ratings": [2, 3]},
            {"source": "B", "price": 200, "ratings": [4]},
        ]
        merged = merge_jsons_for_record(input_jsons, self.simple_schema)
        
        assert merged["source"] == "B"  # Majority vote
        assert merged["price"] == 200   # Majority vote
        assert merged["ratings"] == [1, 2, 3, 4]  # Concatenated lists
    
    def test_merge_nested_schema(self):
        """Test merging nested schema."""
        input_jsons = [
            {"books": [
                {"title": "Book A", "author": "Author X", "sales": [100]},
                {"title": "Book B", "author": "Author Y", "sales": [200]},
            ]},
            {"books": [
                {"title": "Book C", "author": "Author Z", "sales": [300]}
            ]},
        ]
        merged = merge_jsons_for_record(input_jsons, self.nested_schema)
        
        expected_books = [
            {"title": "Book A", "author": "Author X", "sales": [100]},
            {"title": "Book B", "author": "Author Y", "sales": [200]},
            {"title": "Book C", "author": "Author Z", "sales": [300]},
        ]
        assert merged["books"] == expected_books
    
    def test_merge_multiple_schema(self):
        """Test merging multiple schema."""
        input_jsons = [
            {
                "info": {"source": "A", "ratings": [1]},
                "books": [
                    {"title": "Book A", "author": "Author X", "sales": [100]},
                    {"title": "Book B", "author": "Author Y", "sales": [200]},
                ]
            },
            {
                "info": {"source": "A", "ratings": [2]},
                "books": [
                    {"title": "Book C", "author": "Author Z", "sales": [300]},
                ]
            }
        ]
        merged = merge_jsons_for_record(input_jsons, self.multiple_schema)
        
        assert merged["info"]["source"] == "A"
        assert merged["info"]["ratings"] == [1, 2]
        
        expected_books = [
            {"title": "Book A", "author": "Author X", "sales": [100]},
            {"title": "Book B", "author": "Author Y", "sales": [200]},
            {"title": "Book C", "author": "Author Z", "sales": [300]},
        ]
        assert merged["books"] == expected_books
    
    def test_merge_empty_json_list(self):
        """Test merging with empty JSON list."""
        merged = merge_jsons_for_record([], self.simple_schema)
        assert merged == {"source": None, "ratings": [], "price": None}
    
    def test_merge_json_strings(self):
        """Test merging with JSON strings instead of dicts."""
        input_jsons = [
            '{"source": "A", "price": 100}',
            '{"source": "B", "price": 200}',
        ]
        merged = merge_jsons_for_record(input_jsons, self.simple_schema)
        
        assert merged["source"] == "A"  # First value in tie
        assert merged["price"] == 100   # First value in tie
    
    def test_merge_unknown_schema_type(self):
        """Test merging with unknown schema type."""
        mock_schema = Mock()
        mock_schema.schema_type = "unknown"
        
        with pytest.raises(ValueError, match="Unknown schema type: unknown"):
            merge_jsons_for_record([{"test": "data"}], mock_schema)


class TestExplodeJsonResults:
    """Test the explode_json_results function."""
    
    def setup_method(self):
        """Set up test schemas."""
        self.simple_schema = SimpleSchema({
            "variables": [
                {"name": "company", "description": "Company name", "data_type": "string", "required": True},
                {"name": "price", "description": "Price value", "data_type": "number", "required": False},
                {"name": "tags", "description": "Tags", "data_type": "[string]", "required": False},
            ]
        })
        
        self.nested_schema = NestedSchema({
            "container_name": "books",
            "variables": [
                {"name": "title", "description": "Book title", "data_type": "string", "required": True},
                {"name": "author", "description": "Book author", "data_type": "string", "required": True},
                {"name": "price", "description": "Book price", "data_type": "number", "required": False},
                {"name": "genres", "description": "Book genres", "data_type": "[string]", "required": False},
            ]
        })
        
        self.multiple_schema = MultipleSchema({
            "books": {
                "schema_type": "nested",
                "container_name": "books",
                "variables": [
                    {"name": "title", "description": "Book title", "data_type": "string", "required": True},
                    {"name": "author", "description": "Book author", "data_type": "string", "required": True},
                ]
            },
            "authors": {
                "schema_type": "nested",
                "container_name": "authors",
                "variables": [
                    {"name": "name", "description": "Author name", "data_type": "string", "required": True},
                    {"name": "genre", "description": "Author genre", "data_type": "string", "required": False},
                ]
            }
        })
    
    def test_explode_simple_schema(self):
        """Test exploding simple schema results."""
        input_df = pd.DataFrame({
            "chunk_id": [1, 2],
            "json": [
                '{"company": "Apple", "price": 150.0, "tags": ["tech", "hardware"]}',
                '{"company": "Microsoft", "price": 300.0, "tags": ["tech", "software", "cloud"]}'
            ]
        })
        
        result = explode_json_results(input_df, self.simple_schema, json_column="json")
        
        assert len(result) == 2
        assert result.iloc[0]["company"] == "Apple"
        assert result.iloc[0]["price"] == 150.0
        assert result.iloc[0]["tags"] == ["tech", "hardware"]
        assert result.iloc[1]["company"] == "Microsoft"
        assert result.iloc[1]["price"] == 300.0
        assert result.iloc[1]["tags"] == ["tech", "software", "cloud"]
        assert "chunk_id" in result.columns
    
    def test_explode_nested_schema(self):
        """Test exploding nested schema results."""
        input_df = pd.DataFrame({
            "chunk_id": [1, 2],
            "json": [
                '{"books": [{"title": "Python Guide", "author": "Alice", "price": 29.99, "genres": ["programming", "education"]}, {"title": "Data Science", "author": "Bob", "price": 39.99, "genres": ["programming", "science"]}]}',
                '{"books": [{"title": "Machine Learning", "author": "Carol", "price": 49.99, "genres": ["AI", "programming"]}]}'
            ]
        })
        
        result = explode_json_results(input_df, self.nested_schema, json_column="json")
        
        assert len(result) == 3
        assert result.iloc[0]["title"] == "Python Guide"
        assert result.iloc[0]["author"] == "Alice"
        assert result.iloc[0]["price"] == 29.99
        assert result.iloc[0]["genres"] == ["programming", "education"]
        assert result.iloc[1]["title"] == "Data Science"
        assert result.iloc[1]["author"] == "Bob"
        assert result.iloc[2]["title"] == "Machine Learning"
        assert result.iloc[2]["author"] == "Carol"
    
    def test_explode_multiple_schema(self):
        """Test exploding multiple schema results."""
        input_df = pd.DataFrame({
            "chunk_id": [1, 2],
            "json": [
                '{"books": {"books": [{"title": "Python Guide", "author": "Alice"}, {"title": "Data Science", "author": "Bob"}]}, "authors": {"authors": [{"name": "Alice", "genre": "programming"}, {"name": "Bob", "genre": "science"}]}}',
                '{"books": {"books": [{"title": "Machine Learning", "author": "Carol"}]}, "authors": {"authors": [{"name": "Carol", "genre": "AI"}]}}'
            ]
        })
        
        result = explode_json_results(input_df, self.multiple_schema, json_column="json")
        
        assert len(result) == 6  # 3 books + 3 authors
        assert "schema_name" in result.columns
        
        # Check books
        books = result[result["schema_name"] == "books"]
        assert len(books) == 3
        assert books.iloc[0]["books_title"] == "Python Guide"
        assert books.iloc[0]["books_author"] == "Alice"
        
        # Check authors
        authors = result[result["schema_name"] == "authors"]
        assert len(authors) == 3
        assert authors.iloc[0]["authors_name"] == "Alice"
        assert authors.iloc[0]["authors_genre"] == "programming"
    
    def test_explode_with_dict_json(self):
        """Test exploding with dict JSON instead of strings."""
        input_df = pd.DataFrame({
            "chunk_id": [1],
            "json": [{"company": "Apple", "price": 150.0, "tags": ["tech", "hardware"]}]
        })
        
        result = explode_json_results(input_df, self.simple_schema, json_column="json")
        
        assert len(result) == 1
        assert result.iloc[0]["company"] == "Apple"
        assert result.iloc[0]["price"] == 150.0
    
    def test_explode_empty_dataframe(self):
        """Test exploding empty DataFrame."""
        input_df = pd.DataFrame(columns=["chunk_id", "json"])
        
        result = explode_json_results(input_df, self.simple_schema, json_column="json")
        
        assert len(result) == 0
    
    def test_explode_missing_json_column(self):
        """Test exploding with missing JSON column."""
        input_df = pd.DataFrame({"chunk_id": [1]})
        
        with pytest.raises(ValueError, match="Column json not found in input DataFrame"):
            explode_json_results(input_df, self.simple_schema, json_column="json")
    
    def test_explode_with_schema_path(self):
        """Test exploding with schema path instead of schema object."""
        input_df = pd.DataFrame({
            "chunk_id": [1],
            "json": ['{"company": "Apple", "price": 150.0, "tags": ["tech", "hardware"]}']
        })
        
        # Create a temporary schema file
        schema_content = """
schema_type: simple
variables:
  - name: company
    description: Company name
    data_type: string
    required: true
  - name: price
    description: Price value
    data_type: number
    required: false
  - name: tags
    description: Tags
    data_type: "[string]"
    required: false
"""
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = schema_content
            
            with patch('delm.schemas.schema_manager.SchemaManager._load_schema_spec') as mock_load:
                mock_load.return_value = {
                    "schema_type": "simple",
                    "variables": [
                        {"name": "company", "description": "Company name", "data_type": "string", "required": True},
                        {"name": "price", "description": "Price value", "data_type": "number", "required": False},
                        {"name": "tags", "description": "Tags", "data_type": "[string]", "required": False},
                    ]
                }
                
                with patch('delm.schemas.schemas.SchemaRegistry.create') as mock_create:
                    mock_create.return_value = self.simple_schema
                    
                    result = explode_json_results(input_df, "schema.yaml", json_column="json")
                    
                    assert len(result) == 1
                    assert result.iloc[0]["company"] == "Apple"


class TestExtractionVariableIsList:
    """Test the is_list method of ExtractionVariable."""
    
    def test_is_list_with_list_type(self):
        """Test is_list method with list data type."""
        var = ExtractionVariable(name="test", data_type="[string]", required=True, description="")
        assert var.is_list() is True
    
    def test_is_list_with_scalar_type(self):
        """Test is_list method with scalar data type."""
        var = ExtractionVariable(name="test", data_type="string", required=True, description="")
        assert var.is_list() is False
    
    def test_is_list_with_number_type(self):
        """Test is_list method with number data type."""
        var = ExtractionVariable(name="test", data_type="number", required=True, description="")
        assert var.is_list() is False
    
    def test_is_list_with_integer_list_type(self):
        """Test is_list method with integer list data type."""
        var = ExtractionVariable(name="test", data_type="[integer]", required=True, description="")
        assert var.is_list() is True 