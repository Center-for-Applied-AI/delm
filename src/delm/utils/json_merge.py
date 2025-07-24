"""
Merging utilities for DELM extracted JSONs.

Includes logic for majority vote and list concatenation merging.
"""

import json
from collections import Counter
from typing import Any, Dict, List
from delm.schemas.schemas import BaseSchema, ExtractionVariable


def _is_list_type(var: ExtractionVariable) -> bool:
    """Return True if the ExtractionVariable describes a list."""
    return isinstance(var.data_type, str) and var.data_type.startswith("[")


def _majority_vote(values: List[Any]) -> Any:
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    for v in values:           # first winner wins
        if counts[v] == top:
            return v
    return values[0]


def merge_jsons_for_record(json_list: List[Dict[str, Any]], schema: BaseSchema):
    """
    Consolidate multiple extraction results for a single record, obeying:
      • Scalars  → majority vote (ties → first encountered)
      • List-types → concatenate, keep duplicates
    """
    if not json_list:
        json_list = []
    if json_list and isinstance(json_list[0], str):
        json_list = [json.loads(j) for j in json_list] # type: ignore

    schema_type = getattr(schema, "schema_type", type(schema).__name__).lower()

    # SIMPLE
    if schema_type == "simpleschema":
        merged_simple: Dict[str, Any] = {}
        for schema_var in schema.variables:
            bucket: List[Any] = []
            for json_item in json_list:
                val = json_item.get(schema_var.name)
                if val is None:
                    continue
                elif _is_list_type(schema_var):
                    bucket.extend(val) 
                else:
                    bucket.append(val)
            merged_simple[schema_var.name] = bucket if _is_list_type(schema_var) else _majority_vote(bucket)
        return merged_simple

    # NESTED
    if schema_type == "nestedschema":
        nested_container_name = schema.container_name
        merged_nested: List[Dict[str, Any]] = []
        for json_item in json_list:
            items = json_item.get(nested_container_name, [])
            if items:
                merged_nested.extend(items)
        return {nested_container_name: merged_nested}

    # MULTIPLE
    if schema_type == "multipleschema":
        merged_multiple: Dict[str, Any] = {}
        for sub_schema_spec_name, sub_schema in schema.schemas.items():
            sub_schema_type = getattr(sub_schema, "schema_type", type(sub_schema).__name__).lower()
            nested_container_name = getattr(sub_schema, "container_name", None)
            sub_jsons = []
            for json_item in json_list:
                if sub_schema_type == "simpleschema":
                    sub_jsons.append(json_item[sub_schema_spec_name])
                elif sub_schema_type == "nestedschema":
                    nested_json_item = {}
                    if sub_schema_spec_name in json_item:
                        nested_json_item[nested_container_name] = json_item[sub_schema_spec_name]
                    sub_jsons.append(nested_json_item)
            merged_jsons = merge_jsons_for_record(sub_jsons, sub_schema)
            if sub_schema_type == "simpleschema":
                merged_multiple[sub_schema_spec_name] = merged_jsons
            elif sub_schema_type == "nestedschema": 
                merged_multiple[sub_schema_spec_name] = merged_jsons.get(nested_container_name, []) # type: ignore
        return merged_multiple

    raise ValueError(f"Unknown schema type: {schema_type}") 