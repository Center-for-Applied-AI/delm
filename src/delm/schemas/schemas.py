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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Type

import pandas as pd
from pydantic import BaseModel, Field  # <- real Field, returns FieldInfo

from ..constants import DEFAULT_PROMPT_TEMPLATE
from ..exceptions import SchemaError
from ..models import ExtractionVariable

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
    safe_members = {str(v).replace(" ", "_").replace("-", "_"): v for v in allowed}
    return Enum(name, safe_members)

def _ann_and_field(dtype: str, required: bool, desc: str):
    """Return (<annotation>, <FieldInfo>, <is_list_flag>)."""
    is_list = dtype.startswith("[") and dtype.endswith("]")
    base_key = dtype[1:-1] if is_list else dtype
    py_base = _Mapping.get(base_key, str)

    ann = List[py_base] if is_list else py_base  # noqa: F821 – Forward ref ok
    if not required:
        ann = Optional[ann]

    # --- build FieldInfo
    if required:
        fld = Field(..., description=desc)
    else:
        if is_list:
            fld = Field(default_factory=list, description=desc)
        else:
            fld = Field(default=None, description=desc)
    return ann, fld, is_list

def _explode_df(df: pd.DataFrame, list_cols: Sequence[str]) -> pd.DataFrame:
    """Explode each *present* list column so every row is atomic."""
    for col in list_cols:
        if col in df.columns:
            df = df.explode(col, ignore_index=True)
    return df

def _validate_type(val, data_type, path):
    if data_type == "number":
        if not isinstance(val, float):
            raise ValueError(f"{path}: Expected float (number), got {type(val).__name__} ({val!r})")
    elif data_type == "integer":
        if not isinstance(val, int):
            raise ValueError(f"{path}: Expected integer, got {type(val).__name__} ({val!r})")
    elif data_type == "string":
        if not isinstance(val, str):
            raise ValueError(f"{path}: Expected string, got {type(val).__name__} ({val!r})")
    elif data_type == "boolean":
        if not isinstance(val, bool):
            raise ValueError(f"{path}: Expected boolean, got {type(val).__name__} ({val!r})")
    # Add more types as needed

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
        ...

    @abstractmethod
    def create_pydantic_schema(self) -> Type[BaseModel]:
        ...

    @abstractmethod
    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:
        ...

    @abstractmethod
    def validate_and_parse_response_to_exploded_dataframe(
        self, response: Any, text_chunk: str
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def validate_and_parse_response_to_dict(self, response: Any, text_chunk: str) -> dict:
        ...

    @abstractmethod
    def validate_json_dict(self, data: Dict[str, Any], path: str = "root", override_container_name: str | None = None) -> None:
        """Raise ValueError if data is invalid, with a clear path."""
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
        self._variables = [ExtractionVariable.from_dict(v) for v in config.get("variables", [])]
        self.prompt_template = config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

        # derived – which variables are lists?
        self._list_vars = [v.name for v in self._variables if v.data_type.startswith("[")]

    # ---- interface impl ----------------------------------------------------
    @property
    def variables(self) -> List[ExtractionVariable]:
        return self._variables

    def create_pydantic_schema(self) -> Type[BaseModel]:
        annotations, fields = {}, {}
        for v in self.variables:
            ann, fld, _ = _ann_and_field(v.data_type, v.required, v.description)
            annotations[v.name] = ann
            fields[v.name] = fld
        return type("DynamicExtractSchema", (BaseModel,), {"__annotations__": annotations, **fields})

    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:
        return self.prompt_template.format(text=text, variables=self.get_variables_text(), context=context or "")

    # ---- validation helpers ------------------------------------------------
    def _clean(self, response: Any, text_chunk: str) -> Optional[BaseModel]:
        if response is None:
            return None
        instance_dict = response.model_dump() if isinstance(response, BaseModel) else response
        cleaned: Dict[str, Any] = {}
        text_lwr = text_chunk.lower()
        for v in self.variables:
            raw = instance_dict.get(v.name)
            items = raw if isinstance(raw, list) else [raw]
            items = [i for i in items if i is not None]
            if v.allowed_values:
                items = [i for i in items if i in v.allowed_values]
            if v.validate_in_text:
                items = [i for i in items if isinstance(i, str) and i.lower() in text_lwr]
            if v.required and not items:
                return None  # whole response invalid
            cleaned[v.name] = items if v.data_type.startswith("[") else (items[0] if items else None)
        Schema = self.create_pydantic_schema()
        return Schema(**cleaned)

    # ---- public validate/parse --------------------------------------------
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        model = self._clean(response, text_chunk)
        if model is None:
            return pd.DataFrame()
        df = pd.DataFrame([model.model_dump()])
        return _explode_df(df, self._list_vars)

    def validate_and_parse_response_to_dict(self, response: Any, text_chunk: str) -> dict:  # noqa: D401 – simple name
        model = self._clean(response, text_chunk)
        return {} if model is None else model.model_dump(mode="json")

    def validate_json_dict(self, data: Dict[str, Any], path: str = "root") -> None:
        for var in self.variables:
            if var.required and var.name not in data:
                raise ValueError(f"{path}.{var.name}: Required field missing")
            if var.name in data:
                val = data[var.name]
                if var.data_type.startswith("["):
                    if not isinstance(val, list):
                        raise ValueError(f"{path}.{var.name}: Expected list, got {type(val).__name__}")
                    for i, item in enumerate(val):
                        _validate_type(item, var.data_type[1:-1], f"{path}.{var.name}[{i}]")
                else:
                    if isinstance(val, list):
                        raise ValueError(f"{path}.{var.name}: Expected scalar, got list")
                    _validate_type(val, var.data_type, f"{path}.{var.name}")

###############################################################################
# Nested schema (container of items)
###############################################################################
class NestedSchema(BaseSchema):
    def __init__(self, config: Dict[str, Any]):
        self._container_name = config.get("container_name", "instances")
        self._variables = [ExtractionVariable.from_dict(v) for v in config.get("variables", [])]
        self.prompt_template = config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self._list_vars = [v.name for v in self._variables if v.data_type.startswith("[")]

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
        Item = self._item_schema()
        ann = {self.container_name: List[Item]}  # noqa: F821 – forward ref ok
        flds = {self.container_name: Field(default_factory=list, description=f"list of {Item.__name__}")}
        return type("DynamicContainer", (BaseModel,), {"__annotations__": ann, **flds})

    # ---- prompt ------------------------------------------------------------
    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:  # noqa: D401 – simple name
        ctx = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
        return self.prompt_template.format(text=text, variables=self.get_variables_text(), context=ctx)

    # ---- validation --------------------------------------------------------
    def _clean_item(self, raw_item: Dict[str, Any], text_lwr: str) -> Optional[Dict[str, Any]]:
        cleaned: Dict[str, Any] = {}
        for v in self.variables:
            val = raw_item.get(v.name)
            items = val if isinstance(val, list) else [val]
            items = [i for i in items if i is not None]
            if v.allowed_values:
                items = [i for i in items if i in v.allowed_values]
            if v.validate_in_text:
                items = [i for i in items if isinstance(i, str) and i.lower() in text_lwr]
            if v.required and not items:
                return None
            cleaned[v.name] = items if v.data_type.startswith("[") else (items[0] if items else None)
        return cleaned

    def _clean(self, response: Any, text_chunk: str) -> Optional[BaseModel]:
        if response is None:
            return None
        items = getattr(response, self.container_name, [])
        text_lwr = text_chunk.lower()
        cleaned_items = [ci for itm in items if (ci := self._clean_item(itm.model_dump(), text_lwr)) is not None]
        if not cleaned_items:
            return None
        Schema = self.create_pydantic_schema()
        return Schema(**{self.container_name: cleaned_items})

    # ---- public parse ------------------------------------------------------
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        model = self._clean(response, text_chunk)
        if model is None:
            return pd.DataFrame()
        df = pd.DataFrame(getattr(model, self.container_name))
        return _explode_df(df, self._list_vars)

    def validate_and_parse_response_to_dict(self, response: Any, text_chunk: str) -> dict:
        model = self._clean(response, text_chunk)
        return {} if model is None else model.model_dump(mode="json")

    def validate_json_dict(self, data: Dict[str, Any], path: str = "root", override_container_name: str | None = None) -> None:
        container = override_container_name or self.container_name
        if container not in data:
            raise ValueError(f"{path}: Missing container '{container}' in nested schema")
        items = data[container]
        if not isinstance(items, list):
            raise ValueError(f"{path}.{container}: Expected list, got {type(items).__name__}")
        for i, item in enumerate(items):
            for var in self.variables:
                if var.required and var.name not in item:
                    raise ValueError(f"{path}.{container}[{i}].{var.name}: Required field missing")
                if var.name in item:
                    val = item[var.name]
                    if var.data_type.startswith("["):
                        if not isinstance(val, list):
                            raise ValueError(f"{path}.{container}[{i}].{var.name}: Expected list, got {type(val).__name__}")
                        for j, subitem in enumerate(val):
                            _validate_type(subitem, var.data_type[1:-1], f"{path}.{container}[{i}].{var.name}[{j}]")
                    else:
                        if isinstance(val, list):
                            raise ValueError(f"{path}.{container}[{i}].{var.name}: Expected scalar, got list")
                        _validate_type(val, var.data_type, f"{path}.{container}[{i}].{var.name}")

###############################################################################
# Multiple schema – orchestrates several sub‑schemas
###############################################################################
class MultipleSchema(BaseSchema):
    def __init__(self, config: Dict[str, Any]):
        self._schemas: Dict[str, BaseSchema] = {}
        for schema_name, sub_schema_config in config.items():
            if schema_name != "schema_type": # Skip the schema_type key in the spec
                self._schemas[schema_name] = SchemaRegistry().create(sub_schema_config)

    # ---- interface ---------------------------------------------------------
    @property
    def schemas(self) -> Dict[str, "BaseSchema"]:  # type: ignore[override]
        return self._schemas

    @property
    def variables(self) -> List[ExtractionVariable]:
        vars_: List[ExtractionVariable] = []
        for sch in self.schemas.values():
            vars_.extend(sch.variables)
        return vars_

    def create_pydantic_schema(self) -> Type[BaseModel]:
        ann, flds = {}, {}
        for name, sch in self.schemas.items():
            ann[name] = sch.create_pydantic_schema()
            flds[name] = Field(..., description=f"results for {name}")
        return type("MultipleExtract", (BaseModel,), {"__annotations__": ann, **flds})

    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:  # noqa: D401
        parts = []
        for name, sch in self.schemas.items():
            parts.append(f"## {name.upper()}\n" + sch.create_prompt(text, context))
        return "\n\n".join(parts)

    # ---- parse -------------------------------------------------------------
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for name, sch in self.schemas.items():
            sub_resp = getattr(response, name, None) if hasattr(response, name) else None
            df = sch.validate_and_parse_response_to_exploded_dataframe(sub_resp, text_chunk)
            if not df.empty:
                df.insert(0, "schema_type", name)  # prepend column
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def validate_and_parse_response_to_dict(self, response: Any, text_chunk: str) -> dict:  # noqa: D401
        out: Dict[str, Any] = {}
        for name, sch in self.schemas.items():
            sub_resp = getattr(response, name, None) if hasattr(response, name) else None
            val = sch.validate_and_parse_response_to_dict(sub_resp, text_chunk)
            if getattr(sch, "schema_type", type(sch).__name__).lower() == "nestedschema":
                # Unwrap the container
                container = sch.container_name
                out[name] = val.get(container, []) if isinstance(val, dict) else val
            else:
                out[name] = val
        return out

    def validate_json_dict(self, data: Dict[str, Any], path: str = "root") -> None:
        for name, sub_schema in self.schemas.items():
            if name not in data:
                raise ValueError(f"{path}: Missing key '{name}' in multiple schema")
            if isinstance(sub_schema, NestedSchema):
                # We need to wrap the data in a dict with the name as the key so 
                # that the nested schema can validate it. This is so we expect 
                # the data to look like {books: [...]} and not {books: {entries: [...]}}
                #  for example.
                sub_schema.validate_json_dict({name: data[name]}, path=f"{path}.{name}", override_container_name=name)
            else:
                sub_schema.validate_json_dict(data[name], path=f"{path}.{name}")

###############################################################################
# Schema registry
###############################################################################
class SchemaRegistry:
    def __init__(self):
        self._reg: Dict[str, Type[BaseSchema]] = {}
        self._reg.update({
            "simple": SimpleSchema,
            "nested": NestedSchema,
            "multiple": MultipleSchema,
        })

    def register(self, name: str, cls: Type[BaseSchema]):
        self._reg[name] = cls

    def create(self, cfg: Dict[str, Any]) -> BaseSchema:
        typ = cfg.get("schema_type", "simple")
        if typ not in self._reg:
            raise SchemaError(
                f"Unknown schema_type {typ}",
                {"schema_type": typ, "available": list(self._reg.keys())},
            )
        return self._reg[typ](cfg)

    def list_available(self) -> List[str]:
        return list(self._reg.keys())
