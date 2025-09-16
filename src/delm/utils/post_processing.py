"""Post‑processing helpers for DELM extraction outputs."""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
from collections import Counter
from delm.schemas.schemas import BaseSchema, SchemaRegistry, SimpleSchema, NestedSchema, MultipleSchema, ExtractionVariable
from delm.schemas.schema_manager import SchemaManager
from delm.constants import SYSTEM_EXTRACTED_DATA_JSON_COLUMN

# Module-level logger
log = logging.getLogger(__name__)

def _majority_vote(values: List[Any]) -> Any:
    """Perform a majority vote on a list of values.

    Args:
        values: A list of values to vote on.

    Returns:
        The value with the highest count.
    """
    log.debug("Performing majority vote on %d values", len(values))
    if not values:
        log.debug("No values for majority vote, returning None")
        return None
    
    counts = Counter(values)
    top = max(counts.values())
    log.debug("Majority vote counts: %s, top count: %d", dict(counts), top)
    
    for v in values:           # first winner wins
        if counts[v] == top:
            log.debug("Majority vote winner: %s (count: %d)", v, counts[v])
            return v
    
    log.debug("No clear winner, returning first value: %s", values[0])
    return values[0]
    # TODO: should return the first value of the top count, not the first value in the list


def merge_jsons_for_record(json_list: List[Dict[str, Any]], schema: BaseSchema):
    """
    Consolidate multiple extraction results for a single record, obeying:
      • Scalars  → majority vote (ties → first encountered)
      • List-types → concatenate, keep duplicates

    Args:
        json_list: A list of JSON dictionaries to merge.
        schema: The schema to use for merging.

    Returns:
        A dictionary of the merged JSONs.
        
    Raises:
        ValueError: If the schema type is unknown.
    """
    log.debug("Merging %d JSON records for schema type: %s", len(json_list), type(schema).__name__)
    
    if not json_list:
        log.debug("Empty JSON list, using empty list")
        json_list = []
    if json_list and isinstance(json_list[0], str):
        log.debug("Converting %d JSON strings to dicts", len(json_list))
        json_list = [json.loads(j) for j in json_list] # type: ignore

    schema_type = getattr(schema, "schema_type", type(schema).__name__).lower()
    log.debug("Schema type: %s", schema_type)

    # SIMPLE
    if schema_type == "simpleschema":
        log.debug("Processing SimpleSchema with %d variables", len(schema.variables))
        merged_simple: Dict[str, Any] = {}
        for schema_var in schema.variables:
            log.debug("Processing variable: %s", schema_var.name)
            bucket: List[Any] = []
            for json_item in json_list:
                val = json_item.get(schema_var.name)
                if val is None:
                    continue
                elif schema_var.is_list():
                    log.debug("Extending bucket with list value for variable '%s'", schema_var.name)
                    bucket.extend(val) 
                else:
                    log.debug("Appending scalar value for variable '%s'", schema_var.name)
                    bucket.append(val)
            
            if schema_var.is_list():
                merged_simple[schema_var.name] = bucket
                log.debug("Variable '%s' merged as list with %d items", schema_var.name, len(bucket))
            else:
                merged_simple[schema_var.name] = _majority_vote(bucket)
                log.debug("Variable '%s' merged with majority vote from %d values", schema_var.name, len(bucket))
        
        log.debug("SimpleSchema merge completed with %d variables", len(merged_simple))
        return merged_simple

    # NESTED
    if schema_type == "nestedschema":
        nested_container_name = schema.container_name
        log.debug("Processing NestedSchema with container: %s", nested_container_name)
        merged_nested: List[Dict[str, Any]] = []
        for json_item in json_list:
            items = json_item.get(nested_container_name, [])
            if items:
                log.debug("Adding %d items from container '%s'", len(items), nested_container_name)
                merged_nested.extend(items)
        
        log.debug("NestedSchema merge completed: %d total items in container '%s'", len(merged_nested), nested_container_name)
        return {nested_container_name: merged_nested}

    # MULTIPLE
    if schema_type == "multipleschema":
        log.debug("Processing MultipleSchema with %d sub-schemas", len(schema.schemas))
        merged_multiple: Dict[str, Any] = {}
        for sub_schema_spec_name, sub_schema in schema.schemas.items():
            log.debug("Processing sub-schema: %s", sub_schema_spec_name)
            sub_schema_type = getattr(sub_schema, "schema_type", type(sub_schema).__name__).lower()
            nested_container_name = getattr(sub_schema, "container_name", None)
            log.debug("Sub-schema type: %s, container: %s", sub_schema_type, nested_container_name)
            
            sub_jsons = []
            for json_item in json_list:
                if sub_schema_type == "simpleschema":
                    sub_jsons.append(json_item[sub_schema_spec_name])
                elif sub_schema_type == "nestedschema":
                    nested_json_item = {}
                    if sub_schema_spec_name in json_item:
                        nested_json_item[nested_container_name] = json_item[sub_schema_spec_name]
                    sub_jsons.append(nested_json_item)
            
            log.debug("Recursively merging %d sub-jsons for sub-schema '%s'", len(sub_jsons), sub_schema_spec_name)
            merged_jsons = merge_jsons_for_record(sub_jsons, sub_schema)
            
            if sub_schema_type == "simpleschema":
                merged_multiple[sub_schema_spec_name] = merged_jsons
                log.debug("Sub-schema '%s' merged as simple schema", sub_schema_spec_name)
            elif sub_schema_type == "nestedschema": 
                merged_multiple[sub_schema_spec_name] = merged_jsons.get(nested_container_name, []) # type: ignore
                log.debug("Sub-schema '%s' merged as nested schema", sub_schema_spec_name)
        
        log.debug("MultipleSchema merge completed with %d sub-schemas", len(merged_multiple))
        return merged_multiple

    log.error("Unknown schema type: %s", schema_type)
    raise ValueError(f"Unknown schema type: {schema_type}")


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

    Raises:
        ValueError: If the JSON column is not found in the input DataFrame.
    """
    log.debug("Exploding JSON column: %s of %d rows", json_column, len(input_df))
    
    if json_column not in input_df.columns:
        raise ValueError(f"Column {json_column} not found in input DataFrame")

    # Load schema if path is provided
    if isinstance(schema, (str, Path)):
        schema_config = SchemaManager._load_schema_spec(schema)
        schema = SchemaRegistry().create(schema_config)

    df = input_df.copy()
    
    # Handle empty DataFrame
    if len(df) == 0:
        return pd.DataFrame()
    
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
    """Explode a single row for simple schema or nested item without list explosion.

    Args:
        data: The data dictionary to explode.
        system_cols: System columns to include in each row.
        schema: The schema object.
        schema_prefix: Optional prefix to add to column names (for multiple schemas).

    Returns:
        A list with a single row mapping columns to values.
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