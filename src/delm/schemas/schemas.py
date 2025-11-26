"""
DELM Schema System
==================
A **single‑file rewrite** that unifies handling of scalars vs lists, guarantees
proper DataFrame *explosion* for every schema, and cleans up dynamic Pydantic
model generation so type‑checkers (Pyright/Mypy) no longer complain about
`Field` overloads.

> Updated  2025‑07‑22
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Union, Optional, Dict, List, Sequence, Type

from pydantic import BaseModel, Field
import yaml

from delm.constants import LLM_NULL_WORDS_LOWERCASE
from delm.models import ExtractionVariable

# Module-level logger
log = logging.getLogger(__name__)

###############################################################################
# Utilities
###############################################################################
_Mapping: Dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "date": str,
}


def _ann_and_field(dtype: str, required: bool, desc: str):
    """Return (<annotation>, <FieldInfo>, <is_list_flag>)."""
    is_list = dtype.startswith("[") and dtype.endswith("]")
    base_key = dtype[1:-1] if is_list else dtype
    py_base = _Mapping.get(base_key, str)

    ann = List[py_base] if is_list else py_base  # noqa: F821 – Forward ref ok
    # Always make fields Optional in Pydantic schema to accept None from LLM
    # We'll handle "required" logic in our cleaning phase
    ann = Optional[ann]

    # --- build FieldInfo
    # Always allow None values from LLM, handle required logic in cleaning
    if is_list:
        fld = Field(default_factory=list, description=desc)
    else:
        fld = Field(default=None, description=desc)
    return ann, fld, is_list


def _validate_type_safe(val, data_type, path) -> bool:
    """Safe version of _validate_type that returns boolean instead of raising exceptions."""
    log.debug(
        f"Validating type at {path}: {type(val).__name__} ({val!r}) should be {data_type}"
    )
    if data_type == "number":
        if not isinstance(val, float):
            log.warning(
                f"Type validation failed at {path}: Expected float (number), got {type(val).__name__} ({val!r})"
            )
            return False
    elif data_type == "integer":
        if not isinstance(val, int):
            log.warning(
                f"Type validation failed at {path}: Expected integer, got {type(val).__name__} ({val!r})"
            )
            return False
    elif data_type == "string":
        if not isinstance(val, str):
            log.warning(
                f"Type validation failed at {path}: Expected string, got {type(val).__name__} ({val!r})"
            )
            return False
    elif data_type == "boolean":
        if not isinstance(val, bool):
            log.warning(
                f"Type validation failed at {path}: Expected boolean, got {type(val).__name__} ({val!r})"
            )
            return False
    log.debug(f"Type validation passed at {path}")
    return True


###############################################################################
# Abstract base
###############################################################################
class ExtractionSchema(ABC):
    """Common surface for Simple, Nested, Multiple schemas."""

    # Required interface -----------------------------------------------------
    @property
    @abstractmethod
    def variables(self) -> List[ExtractionVariable]:
        """Get the variables for the schema.

        Returns:
            A list of variables for the schema.
        """
        ...

    @abstractmethod
    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create a Pydantic schema for the schema.

        Returns:
            A Pydantic schema for the schema.
        """
        ...

    @abstractmethod
    def create_prompt(self, text: str, prompt_template: str) -> str:
        """Create a prompt for the schema.

        Args:
            text: The text to create the prompt from.
            prompt_template: The prompt template to use.

        Returns:
            A prompt for the schema.
        """
        ...

    @abstractmethod
    def validate_and_parse_response_to_dict(
        self, response: BaseModel, text_chunk: str
    ) -> dict:
        """Validate and parse the response to a dictionary.

        Args:
            response: The response to validate and parse.
            text_chunk: The text chunk that was used to generate the response.

        Returns:
            A dictionary containing the extracted data. If the response is None, returns an empty dictionary.
        """
        ...

    @abstractmethod
    def is_valid_json_dict(
        self,
        data: Dict[str, Any],
        path: str = "root",
        override_container_name: Optional[str] = None,
    ) -> bool:
        """Validate JSON data against schema. Returns True if valid, False if invalid.

        Logs warnings for validation issues but doesn't raise exceptions.
        Used primarily for validating expected/ground truth data in performance estimation.

        Args:
            data: The data to validate.
            path: The path to the data.
            override_container_name: The name of the container to override.

        Returns:
            True if the data is valid, False otherwise.
        """
        ...

    # Convenience ------------------------------------------------------------
    @property
    def container_name(self) -> str:
        return getattr(self, "_container_name", "instances")

    @property
    def schemas(self) -> Dict[str, "ExtractionSchema"]:
        return getattr(self, "_schemas", {})

    # ---------------------------------------------------------------------
    def get_variables_text(self) -> str:
        """Get the variables text for the schema.

        Returns:
            A string containing the variables text.
        """
        lines: List[str] = []
        for v in self.variables:
            s = f"- {v.name}: {v.description} ({v.data_type})"
            if v.required:
                s += " [REQUIRED]"
            if v.allowed_values:
                allowed = ", ".join(f'"{x}"' for x in v.allowed_values)
                s += f" (allowed values: {allowed})"
            lines.append(s)
        return "\n".join(lines)


###############################################################################
# Simple (flat) schema
###############################################################################
class SimpleSchema(ExtractionSchema):
    def __init__(self, variables: List[ExtractionVariable]):
        log.debug("Initializing SimpleSchema")
        self._variables = variables
        self._pydantic_schema: Optional[Type[BaseModel]] = None

    # ---- interface impl ----------------------------------------------------
    @property
    def variables(self) -> List[ExtractionVariable]:
        return self._variables

    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create and cache the Pydantic schema for extraction."""
        if self._pydantic_schema is not None:
            return self._pydantic_schema

        log.debug("Creating Pydantic schema for SimpleSchema")
        annotations, fields = {}, {}
        for v in self.variables:
            ann, fld, _ = _ann_and_field(v.data_type, v.required, v.description)
            annotations[v.name] = ann
            fields[v.name] = fld
        log.debug(
            f"SimpleSchema Pydantic schema created with {len(annotations)} fields"
        )
        self._pydantic_schema = type(
            "DynamicExtractSchema",
            (BaseModel,),
            {"__annotations__": annotations, **fields},
        )
        return self._pydantic_schema

    def create_prompt(
        self, text: str, prompt_template: str, context: Dict[str, Any] | None = None
    ) -> str:
        log.debug("Creating prompt for SimpleSchema")
        variables_text = self.get_variables_text()
        log.debug(
            f"SimpleSchema prompt created with {len(self.variables)} variables, text length: {len(text)}"
        )
        return prompt_template.format(
            text=text, variables=variables_text, context=context or ""
        )

    # ---- validation helpers ------------------------------------------------
    def _clean(self, response: BaseModel, text_chunk: str) -> Optional[BaseModel]:
        """Clean and validate extraction response."""
        instance_dict = response.model_dump()
        cleaned: Dict[str, Any] = {}
        text_lwr = text_chunk.lower()

        for v in self.variables:
            raw = instance_dict.get(v.name)
            items = raw if isinstance(raw, list) else [raw]
            items = [i for i in items if i is not None]

            if "string" in v.data_type:
                # Filter out NONE strings from LLM unless they're explicitly allowed
                if v.allowed_values is None:
                    nones_to_filter = LLM_NULL_WORDS_LOWERCASE
                else:
                    nones_to_filter = [
                        i for i in LLM_NULL_WORDS_LOWERCASE if i not in v.allowed_values
                    ]
                if len(nones_to_filter) > 0:
                    items = [i for i in items if i.lower() not in nones_to_filter]

            if v.allowed_values:
                items = [i for i in items if i in v.allowed_values]
            if v.validate_in_text:
                items = [
                    i for i in items if isinstance(i, str) and i.lower() in text_lwr
                ]
            if v.required and not items:
                return None  # whole response invalid
            cleaned[v.name] = (
                items if v.data_type.startswith("[") else (items[0] if items else None)
            )

        Schema = self.create_pydantic_schema()
        return Schema(**cleaned)

    # ---- public validate/parse --------------------------------------------
    def validate_and_parse_response_to_dict(
        self, response: Any, text_chunk: str
    ) -> dict:
        """Validate and parse response to dict."""
        model = self._clean(response, text_chunk)
        return {} if model is None else model.model_dump(mode="json")

    def is_valid_json_dict(self, data: Dict[str, Any], path: str = "root") -> bool:
        log.debug(
            f"Validating SimpleSchema JSON dict at path '{path}' with {len(self.variables)} variables"
        )
        for var in self.variables:
            if var.required and var.name not in data:
                log.warning(f"Required field '{var.name}' missing at {path}")
                return False
            if var.name in data:
                val = data[var.name]
                log.debug(f"Validating variable '{var.name}' at {path}.{var.name}")
                if var.data_type.startswith("["):
                    if not isinstance(val, list):
                        log.warning(
                            f"Expected list for '{var.name}' at {path}.{var.name}, got {type(val).__name__}"
                        )
                        return False
                    for i, item in enumerate(val):
                        if not _validate_type_safe(
                            item, var.data_type[1:-1], f"{path}.{var.name}[{i}]"
                        ):
                            return False
                else:
                    if isinstance(val, list):
                        log.warning(
                            f"Expected scalar for '{var.name}' at {path}.{var.name}, got list"
                        )
                        return False
                    if not _validate_type_safe(
                        val, var.data_type, f"{path}.{var.name}"
                    ):
                        return False
        log.debug(
            f"SimpleSchema JSON dict validation completed successfully at '{path}'"
        )
        return True


###############################################################################
# Nested schema (container of items)
###############################################################################
class NestedSchema(ExtractionSchema):
    def __init__(self, container_name: str, variables: List[ExtractionVariable]):
        log.debug("Initializing NestedSchema")
        self._container_name = container_name
        self._variables = variables
        self._item_schema_cached: Optional[Type[BaseModel]] = None
        self._pydantic_schema: Optional[Type[BaseModel]] = None
        log.debug(
            f"NestedSchema initialized with container '{self._container_name}', {len(self._variables)} variables"
        )

    # ---- interface ---------------------------------------------------------
    @property
    def variables(self) -> List[ExtractionVariable]:
        return self._variables

    @property
    def container_name(self) -> str:  # noqa: D401 – property overrides base
        return self._container_name

    # ---- dynamic schema ----------------------------------------------------
    def _item_schema(self) -> Type[BaseModel]:
        """Create and cache the item schema for nested extraction."""
        if self._item_schema_cached is not None:
            return self._item_schema_cached

        ann, flds = {}, {}
        for v in self.variables:
            a, fld, _ = _ann_and_field(v.data_type, v.required, v.description)
            ann[v.name] = a
            flds[v.name] = fld
        self._item_schema_cached = type(
            "DynamicItem", (BaseModel,), {"__annotations__": ann, **flds}
        )
        return self._item_schema_cached

    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create and cache the Pydantic schema for extraction."""
        if self._pydantic_schema is not None:
            return self._pydantic_schema

        log.debug(
            f"Creating Pydantic schema for NestedSchema with container '{self.container_name}'"
        )
        Item = self._item_schema()
        ann = {self.container_name: List[Item]}  # noqa: F821 – forward ref ok
        flds = {
            self.container_name: Field(
                default_factory=list, description=f"list of {Item.__name__}"
            )
        }
        log.debug(
            f"NestedSchema Pydantic schema created with container '{self.container_name}'"
        )
        self._pydantic_schema = type(
            "DynamicContainer", (BaseModel,), {"__annotations__": ann, **flds}
        )
        return self._pydantic_schema

    # ---- prompt ------------------------------------------------------------
    def create_prompt(
        self, text: str, prompt_template: str, context: Dict[str, Any] | None = None
    ) -> str:  # noqa: D401 – simple name
        log.debug(
            f"Creating prompt for NestedSchema with container '{self.container_name}'"
        )
        ctx = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
        variables_text = self.get_variables_text()
        log.debug(
            f"NestedSchema prompt created with container '{self.container_name}', {len(self.variables)} variables, text length: {len(text)}"
        )
        return prompt_template.format(text=text, variables=variables_text, context=ctx)

    # ---- validation --------------------------------------------------------
    def _clean_item(
        self, raw_item: Dict[str, Any], text_lwr: str
    ) -> Optional[Dict[str, Any]]:
        """Clean a single item."""
        cleaned: Dict[str, Any] = {}
        for v in self.variables:
            val = raw_item.get(v.name)
            items = val if isinstance(val, list) else [val]
            items = [i for i in items if i is not None]

            if "string" in v.data_type:
                # Filter out NONE strings from LLM unless they're explicitly allowed
                if v.allowed_values is None:
                    nones_to_filter = LLM_NULL_WORDS_LOWERCASE
                else:
                    nones_to_filter = [
                        i for i in LLM_NULL_WORDS_LOWERCASE if i not in v.allowed_values
                    ]
                if len(nones_to_filter) > 0:
                    items = [i for i in items if i.lower() not in nones_to_filter]

            if v.allowed_values:
                items = [i for i in items if i in v.allowed_values]
            if v.validate_in_text:
                items = [
                    i for i in items if isinstance(i, str) and i.lower() in text_lwr
                ]
            if v.required and not items:
                return None
            cleaned[v.name] = (
                items if v.data_type.startswith("[") else (items[0] if items else None)
            )
        return cleaned

    def _clean(self, response: BaseModel, text_chunk: str) -> Optional[BaseModel]:
        """Clean nested response."""
        items = getattr(response, self.container_name, [])
        text_lwr = text_chunk.lower()
        cleaned_items = [
            ci
            for itm in items
            if (ci := self._clean_item(itm.model_dump(), text_lwr)) is not None
        ]
        if not cleaned_items:
            return None
        Schema = self.create_pydantic_schema()
        return Schema(**{self.container_name: cleaned_items})

    # ---- public parse ------------------------------------------------------
    def validate_and_parse_response_to_dict(
        self, response: Any, text_chunk: str
    ) -> dict:
        """Validate and parse response to dict. Hot path - no debug logging."""
        model = self._clean(response, text_chunk)
        return {} if model is None else model.model_dump(mode="json")

    def is_valid_json_dict(
        self,
        data: Dict[str, Any],
        path: str = "root",
        override_container_name: Optional[str] = None,
    ) -> bool:
        container = override_container_name or self.container_name
        log.debug(
            f"Validating NestedSchema JSON dict at path '{path}' with container '{container}' and {len(self.variables)} variables"
        )
        if container not in data:
            log.warning(f"Missing container '{container}' in nested schema at {path}")
            return False
        items = data[container]
        if not isinstance(items, list):
            log.warning(
                f"Expected list for container '{container}' at {path}.{container}, got {type(items).__name__}"
            )
            return False
        log.debug(f"Validating {len(items)} items in container '{container}'")
        for i, item in enumerate(items):
            log.debug(f"Validating item {i} in container '{container}'")
            for var in self.variables:
                if var.required and var.name not in item:
                    log.warning(
                        f"Required field '{var.name}' missing at {path}.{container}[{i}]"
                    )
                    return False
                if var.name in item:
                    val = item[var.name]
                    log.debug(
                        f"Validating variable '{var.name}' at {path}.{container}[{i}].{var.name}"
                    )
                    if var.data_type.startswith("["):
                        if not isinstance(val, list):
                            log.warning(
                                f"Expected list for '{var.name}' at {path}.{container}[{i}].{var.name}, got {type(val).__name__}"
                            )
                            return False
                        for j, subitem in enumerate(val):
                            if not _validate_type_safe(
                                subitem,
                                var.data_type[1:-1],
                                f"{path}.{container}[{i}].{var.name}[{j}]",
                            ):
                                return False
                    else:
                        if isinstance(val, list):
                            log.warning(
                                f"Expected scalar for '{var.name}' at {path}.{container}[{i}].{var.name}, got list"
                            )
                            return False
                        if not _validate_type_safe(
                            val, var.data_type, f"{path}.{container}[{i}].{var.name}"
                        ):
                            return False
        log.debug(
            f"NestedSchema JSON dict validation completed successfully at '{path}' with container '{container}'"
        )
        return True


###############################################################################
# Multiple schema – orchestrates several sub‑schemas
###############################################################################
class MultipleSchema(ExtractionSchema):
    def __init__(self, schemas: Dict[str, ExtractionSchema]):
        log.debug("Initializing MultipleSchema")
        for schema in schemas.values():
            if isinstance(schema, MultipleSchema):
                raise ValueError(f"Cannot nest MultipleSchema")
        self._schemas = schemas
        self._pydantic_schema: Optional[Type[BaseModel]] = None

    # ---- interface ---------------------------------------------------------
    @property
    def schemas(self) -> Dict[str, ExtractionSchema]:
        return self._schemas

    @property
    def variables(self) -> List[ExtractionVariable]:
        vars_: List[ExtractionVariable] = []
        for sch in self.schemas.values():
            vars_.extend(sch.variables)
        return vars_

    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create and cache the Pydantic schema for extraction."""
        if self._pydantic_schema is not None:
            return self._pydantic_schema

        log.debug("Creating Pydantic schema for MultipleSchema")
        ann, flds = {}, {}
        for name, sch in self.schemas.items():
            log.debug(f"Creating Pydantic schema for sub-schema '{name}'")
            ann[name] = sch.create_pydantic_schema()
            flds[name] = Field(..., description=f"results for {name}")
        log.debug(f"MultipleSchema Pydantic schema created with {len(ann)} sub-schemas")
        self._pydantic_schema = type(
            "MultipleExtract", (BaseModel,), {"__annotations__": ann, **flds}
        )
        return self._pydantic_schema

    def create_prompt(
        self, text: str, prompt_template: str, context: Dict[str, Any] | None = None
    ) -> str:  # noqa: D401
        log.debug("Creating prompt for MultipleSchema")
        parts = []
        for name, sch in self.schemas.items():
            log.debug(f"Creating prompt for sub-schema '{name}'")
            parts.append(
                f"## {name.upper()}\n"
                + sch.create_prompt(text, prompt_template, context)
            )
        log.debug(
            f"MultipleSchema prompt created with {len(parts)} sub-schema sections, text length: {len(text)}"
        )
        return "\n\n".join(parts)

    # ---- parse -------------------------------------------------------------
    def validate_and_parse_response_to_dict(
        self, response: Any, text_chunk: str
    ) -> dict:  # noqa: D401
        """Validate and parse response to dict. Hot path - no debug logging."""
        out: Dict[str, Any] = {}
        for name, sch in self.schemas.items():
            sub_resp = (
                getattr(response, name, None) if hasattr(response, name) else None
            )
            val = sch.validate_and_parse_response_to_dict(sub_resp, text_chunk)
            if (
                getattr(sch, "schema_type", type(sch).__name__).lower()
                == "nestedschema"
            ):
                # Unwrap the container
                container = sch.container_name
                unwrapped_val = val.get(container, []) if isinstance(val, dict) else val
                out[name] = unwrapped_val
            else:
                out[name] = val
        return out

    def is_valid_json_dict(self, data: Dict[str, Any], path: str = "root") -> bool:
        log.debug(
            f"Validating MultipleSchema JSON dict at path '{path}' with {len(self.schemas)} sub-schemas"
        )
        for name, sub_schema in self.schemas.items():
            log.debug(f"Validating sub-schema '{name}' at {path}.{name}")
            if name not in data:
                log.warning(f"Missing key '{name}' in multiple schema at {path}")
                return False
            if isinstance(sub_schema, NestedSchema):
                # We need to wrap the data in a dict with the name as the key so
                # that the nested schema can validate it. This is so we expect
                # the data to look like {books: [...]} and not {books: {entries: [...]}}
                #  for example.
                log.debug(
                    f"Sub-schema '{name}' is NestedSchema, wrapping data for validation"
                )
                if not sub_schema.is_valid_json_dict(
                    {name: data[name]},
                    path=f"{path}.{name}",
                    override_container_name=name,
                ):
                    return False
            else:
                log.debug(
                    f"Sub-schema '{name}' is {type(sub_schema).__name__}, validating directly"
                )
                if not sub_schema.is_valid_json_dict(data[name], path=f"{path}.{name}"):
                    return False
        log.debug(
            f"MultipleSchema JSON dict validation completed successfully at '{path}'"
        )
        return True


@dataclass
class Schema:
    """User-facing unified schema API."""

    schema: ExtractionSchema

    @classmethod
    def simple(
        cls,
        *variables: ExtractionVariable,
        variables_list: Optional[List[ExtractionVariable]] = None,
    ) -> "Schema":
        """Create a simple (flat) extraction schema.

        Args:
            *variables: Variable definitions (positional args)
            variables_list: Variable definitions (as list, alternative to *variables)

        Examples:
            Positional args:
            >>> schema = Schema.simple(
            ...     ExtractionVariable("company", "Company name", "string"),
            ...     ExtractionVariable("revenue", "Revenue amount", "number"),
            ... )

            Or as a list:
            >>> vars = [
            ...     ExtractionVariable("company", "Company name", "string"),
            ...     ExtractionVariable("revenue", "Revenue amount", "number"),
            ... ]
            >>> schema = Schema.simple(variables_list=vars)
        """
        # Use positional args if provided, otherwise use variables_list
        vars_list: List[ExtractionVariable] = (
            list(variables) if variables else (variables_list or [])
        )

        if not vars_list:
            raise ValueError("Must provide at least one variable")

        return cls(schema=SimpleSchema(variables=vars_list))

    @classmethod
    def nested(
        cls,
        container_name: str,
        *variables: ExtractionVariable,
        variables_list: Optional[List[ExtractionVariable]] = None,
    ) -> "Schema":
        """Create a nested (list) extraction schema.

        Args:
            container_name: Name for the list container (e.g., "products", "companies")
            *variables: Variable definitions for each object in the list (positional)
            variables_list: Variable definitions (as list, alternative to *variables)

        Examples:
            >>> schema = Schema.nested(
            ...     "products",
            ...     ExtractionVariable("name", "Product name", "string", required=True),
            ...     ExtractionVariable("price", "Product price", "number"),
            ... )
        """
        vars_list: List[ExtractionVariable] = (
            list(variables) if variables else (variables_list or [])
        )

        if not vars_list:
            raise ValueError("Must provide at least one variable")

        return cls(
            schema=NestedSchema(container_name=container_name, variables=vars_list)
        )

    @classmethod
    def multiple(cls, **schemas: "Schema") -> "Schema":
        """Create a multiple schema for extracting several independent structures.

        Args:
            **schemas: Named Schema objects to extract independently

        Examples:
            >>> products_schema = Schema.nested(
            ...     "products",
            ...     ExtractionVariable("name", "Product name", "string")
            ... )
            >>> companies_schema = Schema.nested(
            ...     "companies",
            ...     ExtractionVariable("name", "Company name", "string")
            ... )
            >>> schema = Schema.multiple(
            ...     products=products_schema,
            ...     companies=companies_schema
            ... )
        """
        if not schemas:
            raise ValueError("Must provide at least one sub-schema")

        # Unwrap Schema wrappers to get internal schema objects
        internal_schemas: Dict[str, ExtractionSchema] = {
            name: s.schema for name, s in schemas.items()
        }
        return cls(schema=MultipleSchema(schemas=internal_schemas))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """Create a schema from a dictionary.

        Args:
            data: Dictionary containing schema specification with:
                - schema_type: "simple", "nested", or "multiple" (default: "simple")
                - variables: List of variable definitions (for simple/nested)
                - container_name: Container name (for nested only)
                - For multiple: sub-schema definitions as additional keys

        Returns:
            Schema instance

        Examples:
            Simple schema:
            >>> schema = Schema.from_dict({
            ...     "schema_type": "simple",
            ...     "variables": [
            ...         {"name": "price", "description": "Price", "data_type": "number"}
            ...     ]
            ... })

            Nested schema:
            >>> schema = Schema.from_dict({
            ...     "schema_type": "nested",
            ...     "container_name": "products",
            ...     "variables": [
            ...         {"name": "name", "description": "Product name", "data_type": "string"}
            ...     ]
            ... })

            Multiple schema:
            >>> schema = Schema.from_dict({
            ...     "schema_type": "multiple",
            ...     "commodities": {
            ...         "schema_type": "nested",
            ...         "container_name": "commodities",
            ...         "variables": [...]
            ...     },
            ...     "companies": {
            ...         "schema_type": "simple",
            ...         "variables": [...]
            ...     }
            ... })
        """
        schema_type = data.get("schema_type", "simple").lower()
        log.debug(f"Creating schema from dict with schema_type: {schema_type}")

        if schema_type == "simple":
            variables = [
                ExtractionVariable.from_dict(v) for v in data.get("variables", [])
            ]
            log.debug(f"Created simple schema with {len(variables)} variables")
            return cls.simple(variables_list=variables)

        elif schema_type == "nested":
            container_name = data.get("container_name", "instances")
            variables = [
                ExtractionVariable.from_dict(v) for v in data.get("variables", [])
            ]
            log.debug(
                f"Created nested schema with container '{container_name}' and {len(variables)} variables"
            )
            return cls.nested(container_name, variables_list=variables)

        elif schema_type == "multiple":
            # For multiple schemas, each key (except schema_type) is a sub-schema
            schemas = {}
            for key, value in data.items():
                if key != "schema_type":
                    log.debug(f"Parsing sub-schema '{key}' for multiple schema")
                    schemas[key] = cls.from_dict(value)
            log.debug(
                f"Created multiple schema with {len(schemas)} sub-schemas: {list(schemas.keys())}"
            )
            return cls.multiple(**schemas)

        else:
            available_types = ["simple", "nested", "multiple"]
            log.error(
                f"Unknown schema_type '{schema_type}', available types: {available_types}"
            )
            raise ValueError(
                f"Unknown schema_type '{schema_type}'. Valid types: {', '.join(available_types)}"
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Schema":
        """Create a schema from a YAML file.

        Args:
            path: Path to YAML file containing schema specification

        Returns:
            Schema instance

        Raises:
            ValueError: If file format is not YAML or file is empty
        """
        path = Path(path) if isinstance(path, str) else path
        log.debug(f"Loading schema from YAML file: {path}")

        if path.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError(f"Unsupported schema file format: {path.suffix}")

        log.debug("Loading YAML schema specification")
        content = yaml.safe_load(path.read_text()) or {}

        if not content:
            raise ValueError("YAML schema specification is empty")

        return cls.from_dict(content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary.

        Returns:
            Dictionary representation of the schema that can be used with from_dict()

        Examples:
            >>> schema = Schema.simple(
            ...     ExtractionVariable("price", "Price value", "number")
            ... )
            >>> schema_dict = schema.to_dict()
            >>> # schema_dict == {"schema_type": "simple", "variables": [...]}
        """
        log.debug(f"Converting schema to dict: {type(self.schema).__name__}")

        if isinstance(self.schema, SimpleSchema):
            return {
                "schema_type": "simple",
                "variables": [v.to_dict() for v in self.schema.variables],
            }

        elif isinstance(self.schema, NestedSchema):
            return {
                "schema_type": "nested",
                "container_name": self.schema.container_name,
                "variables": [v.to_dict() for v in self.schema.variables],
            }

        elif isinstance(self.schema, MultipleSchema):
            result: Dict[str, Any] = {"schema_type": "multiple"}
            for name, sub_schema in self.schema.schemas.items():
                # Recursively convert sub-schemas
                sub_schema_wrapper = Schema(schema=sub_schema)
                result[name] = sub_schema_wrapper.to_dict()
            log.debug(
                f"Converted multiple schema with {len(result) - 1} sub-schemas to dict"
            )
            return result

        else:
            raise ValueError(f"Unknown schema type: {type(self.schema).__name__}")
