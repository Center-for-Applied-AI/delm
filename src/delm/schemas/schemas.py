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
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union, Optional, Dict, List, Sequence, Type

from pydantic import BaseModel, Field  # <- real Field, returns FieldInfo

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


def _make_enum(name: str, allowed: Sequence[str]) -> Enum:
    """Create a *safe* Enum from arbitrary strings (spaces / dashes removed)."""
    log.debug(f"Creating enum '{name}' with {len(allowed)} allowed values")
    safe_members = {str(v).replace(" ", "_").replace("-", "_"): v for v in allowed}
    log.debug(f"Enum '{name}' created with {len(safe_members)} safe members")
    return Enum(name, safe_members)


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
class BaseSchema(ABC):
    """Common surface for Simple, Nested, Multiple schemas."""

    def __init__(self, config: Dict[str, Any]):
        pass

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
    def create_prompt(
        self, text: str, prompt_template: str, context: Dict[str, Any] | None = None
    ) -> str:
        """Create a prompt for the schema.

        Args:
            text: The text to create the prompt from.
            prompt_template: The prompt template to use.
            context: The context to inject into the prompt.

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
    def schemas(self) -> Dict[str, "BaseSchema"]:
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
class SimpleSchema(BaseSchema):
    def __init__(self, config: Dict[str, Any]):
        log.debug("Initializing SimpleSchema")
        self._variables = [
            ExtractionVariable.from_dict(v) for v in config.get("variables", [])
        ]
        log.debug(f"SimpleSchema initialized with {len(self._variables)} variables")

        # derived – which variables are lists?
        self._list_vars = [
            v.name for v in self._variables if v.data_type.startswith("[")
        ]
        if self._list_vars:
            log.debug(
                f"SimpleSchema has {len(self._list_vars)} list variables: {self._list_vars}"
            )

    # ---- interface impl ----------------------------------------------------
    @property
    def variables(self) -> List[ExtractionVariable]:
        return self._variables

    def create_pydantic_schema(self) -> Type[BaseModel]:
        log.debug("Creating Pydantic schema for SimpleSchema")
        annotations, fields = {}, {}
        for v in self.variables:
            ann, fld, _ = _ann_and_field(v.data_type, v.required, v.description)
            annotations[v.name] = ann
            fields[v.name] = fld
        log.debug(
            f"SimpleSchema Pydantic schema created with {len(annotations)} fields"
        )
        return type(
            "DynamicExtractSchema",
            (BaseModel,),
            {"__annotations__": annotations, **fields},
        )

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
        log.debug("Cleaning SimpleSchema response")
        instance_dict = response.model_dump()
        cleaned: Dict[str, Any] = {}
        text_lwr = text_chunk.lower()
        log.debug(f"Cleaning {len(self.variables)} variables from response")

        for v in self.variables:
            raw = instance_dict.get(v.name)
            items = raw if isinstance(raw, list) else [raw]
            items = [i for i in items if i is not None]
            log.debug(f"Variable '{v.name}': {len(items)} items before filtering")

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
                    log.debug(
                        f"Variable '{v.name}': {len(items)} items after null filtering"
                    )

            if v.allowed_values:
                items = [i for i in items if i in v.allowed_values]
                log.debug(
                    f"Variable '{v.name}': {len(items)} items after allowed values filtering"
                )
            if v.validate_in_text:
                items = [
                    i for i in items if isinstance(i, str) and i.lower() in text_lwr
                ]
                log.debug(
                    f"Variable '{v.name}': {len(items)} items after text validation"
                )
            if v.required and not items:
                log.debug(
                    f"Required variable '{v.name}' has no valid items, returning None"
                )
                return None  # whole response invalid
            cleaned[v.name] = (
                items if v.data_type.startswith("[") else (items[0] if items else None)
            )

        Schema = self.create_pydantic_schema()
        log.debug(f"SimpleSchema cleaned response with {len(cleaned)} variables")
        return Schema(**cleaned)

    # ---- public validate/parse --------------------------------------------
    def validate_and_parse_response_to_dict(
        self, response: Any, text_chunk: str
    ) -> dict:
        log.debug("Validating and parsing SimpleSchema response to dict")
        model = self._clean(response, text_chunk)
        result = {} if model is None else model.model_dump(mode="json")
        log.debug(f"SimpleSchema dict result has {len(result)} keys")
        return result

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
class NestedSchema(BaseSchema):
    def __init__(self, config: Dict[str, Any]):
        log.debug("Initializing NestedSchema")
        self._container_name = config.get("container_name", "instances")
        self._variables = [
            ExtractionVariable.from_dict(v) for v in config.get("variables", [])
        ]
        self._list_vars = [
            v.name for v in self._variables if v.data_type.startswith("[")
        ]
        log.debug(
            f"NestedSchema initialized with container '{self._container_name}', {len(self._variables)} variables"
        )
        if self._list_vars:
            log.debug(
                f"NestedSchema has {len(self._list_vars)} list variables: {self._list_vars}"
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
        ann, flds = {}, {}
        for v in self.variables:
            a, fld, _ = _ann_and_field(v.data_type, v.required, v.description)
            ann[v.name] = a
            flds[v.name] = fld
        return type("DynamicItem", (BaseModel,), {"__annotations__": ann, **flds})

    def create_pydantic_schema(self) -> Type[BaseModel]:
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
        return type("DynamicContainer", (BaseModel,), {"__annotations__": ann, **flds})

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
        log.debug(f"Cleaning NestedSchema item with {len(self.variables)} variables")
        cleaned: Dict[str, Any] = {}
        for v in self.variables:
            val = raw_item.get(v.name)
            items = val if isinstance(val, list) else [val]
            items = [i for i in items if i is not None]
            log.debug(f"Variable '{v.name}': {len(items)} items before filtering")

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
                    log.debug(
                        f"Variable '{v.name}': {len(items)} items after null filtering"
                    )

            if v.allowed_values:
                items = [i for i in items if i in v.allowed_values]
                log.debug(
                    f"Variable '{v.name}': {len(items)} items after allowed values filtering"
                )
            if v.validate_in_text:
                items = [
                    i for i in items if isinstance(i, str) and i.lower() in text_lwr
                ]
                log.debug(
                    f"Variable '{v.name}': {len(items)} items after text validation"
                )
            if v.required and not items:
                log.debug(
                    f"Required variable '{v.name}' has no valid items, skipping item"
                )
                return None
            cleaned[v.name] = (
                items if v.data_type.startswith("[") else (items[0] if items else None)
            )
        log.debug(f"NestedSchema item cleaned with {len(cleaned)} variables")
        return cleaned

    def _clean(self, response: BaseModel, text_chunk: str) -> Optional[BaseModel]:
        log.debug(
            f"Cleaning NestedSchema response with container '{self.container_name}'"
        )
        items = getattr(response, self.container_name, [])
        log.debug(
            f"NestedSchema found {len(items)} items in container '{self.container_name}'"
        )
        text_lwr = text_chunk.lower()
        cleaned_items = [
            ci
            for itm in items
            if (ci := self._clean_item(itm.model_dump(), text_lwr)) is not None
        ]
        log.debug(
            f"NestedSchema cleaned {len(cleaned_items)} valid items from {len(items)} total items"
        )
        if not cleaned_items:
            log.debug(
                f"NestedSchema no valid items found in container '{self.container_name}', returning None"
            )
            return None
        Schema = self.create_pydantic_schema()
        log.debug(f"NestedSchema created cleaned model with {len(cleaned_items)} items")
        return Schema(**{self.container_name: cleaned_items})

    # ---- public parse ------------------------------------------------------
    def validate_and_parse_response_to_dict(
        self, response: Any, text_chunk: str
    ) -> dict:
        log.debug(
            f"Validating and parsing NestedSchema response to dict with container '{self.container_name}'"
        )
        model = self._clean(response, text_chunk)
        result = {} if model is None else model.model_dump(mode="json")
        if model is not None:
            items = result.get(self.container_name, [])
            log.debug(
                f"NestedSchema dict result has container '{self.container_name}' with {len(items)} items"
            )
        else:
            log.debug("NestedSchema dict result is empty")
        return result

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
class MultipleSchema(BaseSchema):
    def __init__(self, config: Dict[str, Any]):
        log.debug("Initializing MultipleSchema")
        self._schemas: Dict[str, BaseSchema] = {}
        for schema_name, sub_schema_config in config.items():
            if schema_name != "schema_type":  # Skip the schema_type key in the spec
                log.debug(f"Creating sub-schema '{schema_name}'")
                self._schemas[schema_name] = SchemaRegistry().create(sub_schema_config)
        log.debug(
            f"MultipleSchema initialized with {len(self._schemas)} sub-schemas: {list(self._schemas.keys())}"
        )

    # ---- interface ---------------------------------------------------------
    @property
    def schemas(self) -> Dict[str, "BaseSchema"]:
        return self._schemas

    @property
    def variables(self) -> List[ExtractionVariable]:
        vars_: List[ExtractionVariable] = []
        for sch in self.schemas.values():
            vars_.extend(sch.variables)
        return vars_

    def create_pydantic_schema(self) -> Type[BaseModel]:
        log.debug("Creating Pydantic schema for MultipleSchema")
        ann, flds = {}, {}
        for name, sch in self.schemas.items():
            log.debug(f"Creating Pydantic schema for sub-schema '{name}'")
            ann[name] = sch.create_pydantic_schema()
            flds[name] = Field(..., description=f"results for {name}")
        log.debug(f"MultipleSchema Pydantic schema created with {len(ann)} sub-schemas")
        return type("MultipleExtract", (BaseModel,), {"__annotations__": ann, **flds})

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
        log.debug("Validating and parsing MultipleSchema response to dict")
        out: Dict[str, Any] = {}
        for name, sch in self.schemas.items():
            log.debug(f"Processing sub-schema '{name}' for dict output")
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
                log.debug(
                    f"Sub-schema '{name}' (nested) unwrapped container '{container}' with {len(unwrapped_val) if isinstance(unwrapped_val, list) else 'scalar'} items"
                )
            else:
                out[name] = val
                log.debug(
                    f"Sub-schema '{name}' (simple) with {len(val) if isinstance(val, dict) else 'scalar'} items"
                )
        log.debug(f"MultipleSchema dict result has {len(out)} sub-schemas")
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


###############################################################################
# Schema registry
###############################################################################
class SchemaRegistry:
    def __init__(self):
        log.debug("Initializing SchemaRegistry")
        self._reg: Dict[str, Type[BaseSchema]] = {}
        self._reg.update(
            {
                "simple": SimpleSchema,
                "nested": NestedSchema,
                "multiple": MultipleSchema,
            }
        )
        log.debug(
            f"SchemaRegistry initialized with {len(self._reg)} schema types: {list(self._reg.keys())}"
        )

    def register(self, name: str, cls: Type[BaseSchema]):
        log.debug(f"Registering schema type '{name}' with class {cls.__name__}")
        self._reg[name] = cls

    def create(self, cfg: Dict[str, Any]) -> BaseSchema:
        typ = cfg.get("schema_type", "simple")
        log.debug(f"Creating schema with type '{typ}'")
        if typ not in self._reg:
            log.error(
                f"Unknown schema_type '{typ}', available types: {list(self._reg.keys())}"
            )
            raise ValueError(
                f"Unknown schema_type {typ}, available types: {list(self._reg.keys())}"
            )
        schema = self._reg[typ](cfg)
        log.debug(f"Successfully created schema of type '{typ}'")
        return schema

    def list_available(self) -> List[str]:
        available = list(self._reg.keys())
        log.debug(f"Available schema types: {available}")
        return available
