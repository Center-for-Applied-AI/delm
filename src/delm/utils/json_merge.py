"""
Merging utilities for DELM extracted JSONs.

Includes logic for majority vote and list concatenation merging.
"""

import logging
import json
from collections import Counter
from typing import Any, Dict, List
from delm.schemas.schemas import BaseSchema, ExtractionVariable

# Module-level logger
log = logging.getLogger(__name__)


def _is_list_type(var: ExtractionVariable) -> bool:
    """Return True if the ExtractionVariable describes a list.
    
    Args:
        var: The ExtractionVariable to check.

    Returns:
        True if the ExtractionVariable describes a list, False otherwise.
    """
    is_list = isinstance(var.data_type, str) and var.data_type.startswith("[")
    log.debug("Checking if variable '%s' is list type: %s (data_type: %s)", var.name, is_list, var.data_type)
    return is_list


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
                elif _is_list_type(schema_var):
                    log.debug("Extending bucket with list value for variable '%s'", schema_var.name)
                    bucket.extend(val) 
                else:
                    log.debug("Appending scalar value for variable '%s'", schema_var.name)
                    bucket.append(val)
            
            if _is_list_type(schema_var):
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