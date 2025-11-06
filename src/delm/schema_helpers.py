"""Schema helper functions for simplified DELM API.

This module provides convenience functions for creating schemas without
having to work with config dictionaries directly.
"""

from typing import Any, Dict, List, Optional, Union
from delm.models import ExtractionVariable


def variable(
    name: str,
    description: str,
    data_type: str,
    required: bool = False,
    allowed_values: Optional[List[str]] = None,
    validate_in_text: bool = False,
) -> ExtractionVariable:
    """Create a variable for extraction.

    Args:
        name: Variable name.
        description: Variable description.
        data_type: Variable data type (e.g., "string", "number", "[string]").
        required: Whether the variable is required.
        allowed_values: Optional list of allowed values.
        validate_in_text: Whether to validate that string values appear in the text.

    Returns:
        An ExtractionVariable instance.

    Example:
        >>> var = variable("company", "Company name", "string", required=True)
    """
    return ExtractionVariable(
        name=name,
        description=description,
        data_type=data_type,
        required=required,
        allowed_values=allowed_values,
        validate_in_text=validate_in_text,
    )


def simple_schema(*variables: ExtractionVariable) -> Dict[str, Any]:
    """Create a simple (flat) schema from variables.

    Args:
        *variables: Variable definitions.

    Returns:
        A schema configuration dictionary.

    Example:
        >>> schema = simple_schema(
        ...     variable("company", "Company name", "string"),
        ...     variable("price", "Price value", "number"),
        ... )
    """
    return {
        "schema_type": "simple",
        "variables": [
            {
                "name": v.name,
                "description": v.description,
                "data_type": v.data_type,
                "required": v.required,
                "allowed_values": v.allowed_values,
                "validate_in_text": v.validate_in_text,
            }
            for v in variables
        ],
    }


def nested_schema(
    container_name: str,
    *variables: ExtractionVariable
) -> Dict[str, Any]:
    """Create a nested schema (list of items) from variables.

    Args:
        container_name: Name of the container field.
        *variables: Variable definitions for items in the container.

    Returns:
        A schema configuration dictionary.

    Example:
        >>> schema = nested_schema(
        ...     "companies",
        ...     variable("name", "Company name", "string", required=True),
        ...     variable("revenue", "Revenue", "number"),
        ... )
    """
    return {
        "schema_type": "nested",
        "container_name": container_name,
        "variables": [
            {
                "name": v.name,
                "description": v.description,
                "data_type": v.data_type,
                "required": v.required,
                "allowed_values": v.allowed_values,
                "validate_in_text": v.validate_in_text,
            }
            for v in variables
        ],
    }


def multiple_schema(**schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Create a multiple schema from sub-schemas.

    Args:
        **schemas: Named sub-schemas (each should be a simple or nested schema).

    Returns:
        A schema configuration dictionary.

    Example:
        >>> products = nested_schema(
        ...     "products",
        ...     variable("name", "Product name", "string"),
        ... )
        >>> companies = nested_schema(
        ...     "companies",
        ...     variable("name", "Company name", "string"),
        ... )
        >>> schema = multiple_schema(products=products, companies=companies)
    """
    result = {"schema_type": "multiple"}
    result.update(schemas)
    return result

