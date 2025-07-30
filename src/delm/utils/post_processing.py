import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
from delm.schemas.schemas import BaseSchema, SchemaRegistry, SimpleSchema, NestedSchema, MultipleSchema
from delm.schemas.schema_manager import SchemaManager
from delm.constants import SYSTEM_EXTRACTED_DATA_JSON_COLUMN

# Module-level logger
log = logging.getLogger(__name__)


def explode_json_results(input_df: pd.DataFrame, schema: Union[BaseSchema, str, Path], json_column: str = SYSTEM_EXTRACTED_DATA_JSON_COLUMN) -> pd.DataFrame:
    """
    Explode JSON results according to the schema structure.
    
    This function handles all schema types:
    - Simple: Explodes list fields, keeps other fields as-is
    - Nested: Explodes the container list, then explodes any list fields within items
    - Multiple: Explodes each sub-schema separately and combines with schema_name column
    
    Args:
        input_df: DataFrame with JSON results
        schema: The schema object or path to schema file (YAML/JSON)
        json_column: Name of column containing JSON data (either JSON string or Python dict)
        
    Returns:
        DataFrame with exploded results where each extracted item gets its own row
    """
    log.debug("Exploding JSON column: %s of %d rows", json_column, len(input_df))
    
    if json_column not in input_df.columns:
        raise ValueError(f"Column {json_column} not found in input DataFrame")

    # Load schema if path is provided
    if isinstance(schema, (str, Path)):
        schema_config = SchemaManager._load_schema_spec(schema)
        schema = SchemaRegistry().create(schema_config)

    df = input_df.copy()
    
    # Convert JSON strings to Python objects if needed
    if df[json_column].dtype == 'object' and isinstance(df[json_column].iloc[0], str):
        df[json_column] = df[json_column].apply(lambda x: json.loads(x) if x else {})
    
    exploded_rows = []
    
    for idx, row in df.iterrows():
        json_data = row[json_column]
        if not json_data:
            continue
            
        # Get system columns (non-JSON data)
        system_cols = {col: row[col] for col in row.index if col != json_column}
        
        if isinstance(schema, SimpleSchema):
            # For simple schema, data is already flat
            # Just need to explode any list fields
            exploded_rows.extend(_explode_simple_schema_row(json_data, system_cols, schema))
            
        elif isinstance(schema, NestedSchema):
            # For nested schema, explode the container list
            container_name = schema.container_name
            container_data = json_data.get(container_name, [])
            
            if isinstance(container_data, list):
                for item in container_data:
                    exploded_rows.extend(_explode_simple_schema_row(item, system_cols, schema))
            else:
                # Single item case
                exploded_rows.extend(_explode_simple_schema_row(container_data, system_cols, schema))
                
        elif isinstance(schema, MultipleSchema):
            # For multiple schema, explode each sub-schema separately
            for schema_name, sub_schema in schema.schemas.items():
                sub_data = json_data.get(schema_name, {})
                
                if isinstance(sub_schema, NestedSchema):
                    # Handle nested sub-schema
                    container_name = sub_schema.container_name
                    container_data = sub_data.get(container_name, [])
                    
                    if isinstance(container_data, list):
                        for item in container_data:
                            row_data = _explode_simple_schema_row(item, system_cols, sub_schema, schema_name)
                            for r in row_data:
                                r['schema_name'] = schema_name
                            exploded_rows.extend(row_data)
                    else:
                        # Single item case
                        row_data = _explode_simple_schema_row(container_data, system_cols, sub_schema, schema_name)
                        for r in row_data:
                            r['schema_name'] = schema_name
                        exploded_rows.extend(row_data)
                else:
                    # Handle simple sub-schema
                    row_data = _explode_simple_schema_row(sub_data, system_cols, sub_schema, schema_name)
                    for r in row_data:
                        r['schema_name'] = schema_name
                    exploded_rows.extend(row_data)
    
    if not exploded_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(exploded_rows)

def _explode_simple_schema_row(data: Dict[str, Any], system_cols: Dict[str, Any], schema: BaseSchema, schema_prefix: str = "") -> List[Dict[str, Any]]:
    """
    Explode a single row from a simple schema (or nested schema item).
    Keeps list fields as lists, does not explode them.
    
    Args:
        data: The data dictionary to explode
        system_cols: System columns to include in each row
        schema: The schema object
        schema_prefix: Optional prefix to add to column names (for multiple schemas)
    """
    if not data:
        return []
    
    # Create a single row with all data (including lists as-is)
    row = {**system_cols}
    
    # Add all fields with prefix if provided
    for var in schema.variables:
        col_name = f"{schema_prefix}_{var.name}" if schema_prefix else var.name
        row[col_name] = data.get(var.name)
    
    return [row]

def explode_json_results_in_place(input_df: pd.DataFrame, json_column: str) -> pd.DataFrame:
    """
    Simple explode of JSON column (deprecated - use explode_json_results with schema instead).
    """
    return input_df.explode(json_column, ignore_index=True)

if __name__ == "__main__":
    print("=== JSON EXPLOSION TESTING ===")
    print()
    
    # Test 1: Simple Schema
    print("1. SIMPLE SCHEMA TEST")
    print("-" * 40)
    simple_df = pd.DataFrame({
        "chunk_id": [1, 2],
        "json": [
            '{"company": "Apple", "price": 150.0, "tags": ["tech", "hardware"]}',
            '{"company": "Microsoft", "price": 300.0, "tags": ["tech", "software", "cloud"]}'
        ]
    })
    
    simple_schema = SimpleSchema({
        "variables": [
            {"name": "company", "description": "Company name", "data_type": "string", "required": True},
            {"name": "price", "description": "Price value", "data_type": "number", "required": False},
            {"name": "tags", "description": "Tags", "data_type": "[string]", "required": False},
        ]
    })
    
    print("Original DataFrame:")
    print(simple_df)
    print()
    
    result = explode_json_results(simple_df, simple_schema, json_column="json")
    print("Exploded DataFrame:")
    print(result)
    print()
    
    # Test 2: Nested Schema
    print("2. NESTED SCHEMA TEST")
    print("-" * 40)
    nested_df = pd.DataFrame({
        "chunk_id": [1, 2],
        "json": [
            '{"books": [{"title": "Python Guide", "author": "Alice", "price": 29.99, "genres": ["programming", "education"]}, {"title": "Data Science", "author": "Bob", "price": 39.99, "genres": ["programming", "science"]}]}',
            '{"books": [{"title": "Machine Learning", "author": "Carol", "price": 49.99, "genres": ["AI", "programming"]}]}'
        ]
    })
    
    nested_schema = NestedSchema({
        "container_name": "books",
        "variables": [
            {"name": "title", "description": "Book title", "data_type": "string", "required": True},
            {"name": "author", "description": "Book author", "data_type": "string", "required": True},
            {"name": "price", "description": "Book price", "data_type": "number", "required": False},
            {"name": "genres", "description": "Book genres", "data_type": "[string]", "required": False},
        ]
    })
    
    print("Original DataFrame:")
    print(nested_df)
    print()
    
    result = explode_json_results(nested_df, nested_schema, json_column="json")
    print("Exploded DataFrame:")
    print(result)
    print()
    
    # Test 3: Multiple Schema
    print("3. MULTIPLE SCHEMA TEST")
    print("-" * 40)
    multiple_df = pd.DataFrame({
        "chunk_id": [1, 2],
        "json": [
            '{"books": {"books": [{"title": "Python Guide", "author": "Alice"}, {"title": "Data Science", "author": "Bob"}]}, "authors": {"authors": [{"name": "Alice", "genre": "programming"}, {"name": "Bob", "genre": "science"}]}}',
            '{"books": {"books": [{"title": "Machine Learning", "author": "Carol"}]}, "authors": {"authors": [{"name": "Carol", "genre": "AI"}]}}'
        ]
    })
    
    multiple_schema = MultipleSchema({
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
    
    print("Original DataFrame:")
    print(multiple_df)
    print()
    
    result = explode_json_results(multiple_df, multiple_schema, json_column="json")
    print("Exploded DataFrame:")
    print(result)
    print()
    
    print("=== ALL TESTS COMPLETED ===")