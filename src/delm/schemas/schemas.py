"""
DELM Schema System
==================
Unified schema system for data extraction with extensible registry pattern.
Supports simple, nested, and multiple schema types with progressive complexity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from ..models import ExtractionVariable
from ..exceptions import SchemaError
from ..constants import DEFAULT_PROMPT_TEMPLATE


def make_enum(name, allowed_values):
    # Enum names must be valid identifiers, so replace spaces and special chars
    return Enum(name, {str(v).replace(' ', '_').replace('-', '_'): v for v in allowed_values})


class BaseSchema(ABC):
    """Base interface for all schema types."""
    
    @property
    @abstractmethod
    def variables(self) -> List[ExtractionVariable]:
        """Get variables for this schema."""
        pass
    
    @property
    def container_name(self) -> str:
        """Get container name for nested schemas."""
        return getattr(self, '_container_name', 'instances')
    
    @property
    def schemas(self) -> Dict[str, 'BaseSchema']:
        """Get sub-schemas for multiple schemas."""
        return getattr(self, '_schemas', {})
    
    @abstractmethod
    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create Pydantic model for extraction."""
        pass
    
    @abstractmethod
    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:
        """Create extraction prompt."""
        pass
    
    @abstractmethod
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        """Validate the response (removing invalid elements) and parse it into a DataFrame of valid rows."""
        pass
    
    @abstractmethod
    def validate_and_parse_response_to_json(self, response: Any, text_chunk: str) -> dict:
        """Validate the response and return a JSON-serializable dict of valid data, or None if nothing is valid."""
        pass
    
    def get_variables_text(self) -> str:
        """Get the formatted variables text without the full prompt."""
        variable_descriptions = []
        for var in self.variables:
            desc = f"- {var.name}: {var.description} ({var.data_type})"
            if var.required:
                desc += " [REQUIRED]"
            
            # Add allowed values to description
            if var.allowed_values:
                allowed_list = ", ".join([f'"{v}"' for v in var.allowed_values])
                desc += f" (allowed values: {allowed_list})"
            
            variable_descriptions.append(desc)
        
        return "\n".join(variable_descriptions)

    def _is_row_valid(self, instance_dict, text_chunk: str) -> bool:
        text_chunk_lower = text_chunk.lower()
        for var in self.variables:
            value = instance_dict.get(var.name)
            if value is None and var.required:
                return False
            if value is not None and var.validate_in_text and isinstance(value, str) and value.lower() not in text_chunk_lower:
                if var.required:
                    return False
        return True

    def _validate_pydantic_output_schema(self, response, text_chunk: str):
        """Return a cleaned/validated version of the response, with invalid elements removed. Return None if nothing is valid."""
        raise NotImplementedError


class SimpleSchema(BaseSchema):
    """Simple key-value extraction (current DELM behavior)."""
    
    def __init__(self, config: Dict[str, Any]):
        self._variables = [ExtractionVariable.from_dict(v) for v in config.get("variables", [])]
        self.prompt_template = config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    
    @property
    def variables(self) -> List[ExtractionVariable]:
        return self._variables
    
    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create simple Pydantic schema from variables."""
        fields = {}
        annotations = {}
        for var in self.variables:
            if var.data_type == 'string' or var.allowed_values:
                base_type = str
            elif var.data_type == 'number':
                base_type = float
            elif var.data_type == 'integer':
                base_type = int
            elif var.data_type == 'boolean':
                base_type = bool
            elif var.data_type == 'date':
                base_type = str
            else:
                base_type = str

            field_type = List[base_type]
            if var.required:
                fields[var.name] = Field(..., description=var.description)
            else:
                fields[var.name] = Field(default_factory=list, description=var.description)

            annotations[var.name] = field_type

        DynamicExtractSchema = type(
            'DynamicExtractSchema',
            (BaseModel,),
            {
                '__annotations__': annotations,
                **fields
            }
        )
        return DynamicExtractSchema
    
    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:
        """Create extraction prompt from template and variables."""
        variables_text = self.get_variables_text()
        return self.prompt_template.format(
            text=text,
            variables=variables_text
        )
    
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        """Validate the response and parse it into a DataFrame of valid rows. Only valid rows are included."""
        if response is None:
            return pd.DataFrame()
        validated_data = self._validate_pydantic_output_schema(response, text_chunk)
        if validated_data is None:
            return pd.DataFrame()
        instance_dict = validated_data.model_dump()
        row = {}
        for var in self.variables:
            value = instance_dict.get(var.name)
            if value is not None:
                row[var.name] = value
        return pd.DataFrame([row])

    def validate_and_parse_response_to_json(self, response: Any, text_chunk: str) -> dict:
        validated_data = self._validate_pydantic_output_schema(response, text_chunk)
        if validated_data is None:
            return {}
        return validated_data.model_dump(mode='json')

    def _validate_pydantic_output_schema(self, response, text_chunk: str):
        instance_dict = response.model_dump()
        filtered = {}
        for var in self.variables:
            values = instance_dict.get(var.name, [])
            if not isinstance(values, list):
                values = [values]
            valid = []
            for v in values:
                if v is None and var.required:
                    continue
                if var.allowed_values and v is not None and v not in var.allowed_values:
                    continue
                valid.append(v)
            if var.required and not valid:
                return None
            filtered[var.name] = valid
        from pydantic import create_model
        DynamicExtractSchema = self.create_pydantic_schema()
        return DynamicExtractSchema(**filtered)


class NestedSchema(BaseSchema):
    """Nested object extraction (like commodity data)."""
    
    def __init__(self, config: Dict[str, Any]):
        self._container_name = config.get("container_name", "instances")
        self._variables = [ExtractionVariable.from_dict(v) for v in config.get("variables", [])]
        self.prompt_template = config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    
    @property
    def container_name(self) -> str:
        return self._container_name
    
    @property
    def variables(self) -> List[ExtractionVariable]:
        return self._variables
    
    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create nested Pydantic schema with container and item classes."""
        item_fields = {}
        item_annotations = {}
        for var in self.variables:
            if var.data_type == 'string' or var.allowed_values:
                base_type = str
            elif var.data_type == 'number':
                base_type = float
            elif var.data_type == 'integer':
                base_type = int
            elif var.data_type == 'boolean':
                base_type = bool
            elif var.data_type == 'date':
                base_type = str
            else:
                base_type = str

            if var.required:
                field_type = base_type
                field = Field(description=var.description)
            else:
                field_type = Optional[base_type]
                field = Field(default=None, description=var.description)

            item_fields[var.name] = field
            item_annotations[var.name] = field_type

        DynamicItemSchema = type(
            'DynamicItemSchema',
            (BaseModel,),
            {
                '__annotations__': item_annotations,
                **item_fields
            }
        )
        container_fields = {
            self.container_name: Field(default_factory=list, description=f"List of {self.container_name}")
        }
        container_annotations = {
            self.container_name: List[DynamicItemSchema]
        }
        DynamicContainerSchema = type(
            'DynamicContainerSchema',
            (BaseModel,),
            {
                '__annotations__': container_annotations,
                **container_fields
            }
        )
        return DynamicContainerSchema
    
    def _get_field_type(self, data_type: str, required: bool = True):
        """Map data types to Python types."""
        base_type_mapping = {
            'string': str,
            'number': float,
            'integer': int,
            'boolean': bool,
            'date': str,
            'optional_string': Optional[str],
            'optional_number': Optional[float],
            'optional_integer': Optional[int],
            'optional_boolean': Optional[bool],
            'optional_date': Optional[str],
        }
        
        base_type = base_type_mapping.get(data_type, str)
        
        # If field is not required, make it optional
        if not required:
            if base_type == str:
                return Optional[str]
            elif base_type == float:
                return Optional[float]
            elif base_type == int:
                return Optional[int]
            elif base_type == bool:
                return Optional[bool]
            else:
                return Optional[str]  # Default to optional string
        
        return base_type
    
    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:
        """Create extraction prompt with context."""
        variables_text = self.get_variables_text()
        
        # Format the prompt template
        if self.prompt_template:
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                return self.prompt_template.format(text=text, variables=variables_text, context=context_str)
            return self.prompt_template.format(text=text, variables=variables_text)
        else:
            # Default prompt if no template provided
            return f"Extract the following information from the text:\n\n{variables_text}\n\nText to analyze:\n{text}"
    
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        """Validate the response (removing invalid elements) and parse it into a DataFrame of valid rows."""
        if response is None:
            return pd.DataFrame()
        validated_data = self._validate_pydantic_output_schema(response, text_chunk)
        if validated_data is None:
            return pd.DataFrame()
        if not hasattr(validated_data, self.container_name):
            raise SchemaError(
                f"Validated response is not a Pydantic model with container name '{self.container_name}'",
                {"response_type": type(validated_data), "text_chunk": text_chunk[:100]}
            )
        instances = getattr(validated_data, self.container_name)
        if not instances:
            return pd.DataFrame()
        data = []
        for instance in instances:
            instance_dict = instance.model_dump()
            row = {}
            for var in self.variables:
                value = instance_dict.get(var.name)
                if value is not None:
                    row[var.name] = value
            data.append(row)
        return pd.DataFrame(data)

    def validate_and_parse_response_to_json(self, response: Any, text_chunk: str) -> dict:
        validated_data = self._validate_pydantic_output_schema(response, text_chunk)
        if validated_data is None:
            return {}
        return validated_data.model_dump(mode='json')

    def _validate_pydantic_output_schema(self, response, text_chunk: str):
        items = getattr(response, self.container_name, [])
        valid_items = []
        for item in items:
            instance_dict = item.model_dump()
            valid = True
            for var in self.variables:
                value = instance_dict.get(var.name)
                if var.required and (value is None or value == ""):
                    valid = False
                    break
                if var.allowed_values and value is not None and value not in var.allowed_values:
                    valid = False
                    break
            if valid:
                valid_items.append(item)
        if not valid_items:
            return None
        response_dict = response.model_dump()
        response_dict[self.container_name] = [item.model_dump() for item in valid_items]
        DynamicContainerSchema = self.create_pydantic_schema()
        return DynamicContainerSchema(**response_dict)


class MultipleSchema(BaseSchema):
    """Multiple independent schemas in one config."""
    
    def __init__(self, config: Dict[str, Any]):
        self._schemas = {}
        self.schema_registry = SchemaRegistry()
        
        for name, schema_config in config.items():
            if name != "schema_type":
                self._schemas[name] = self.schema_registry.create(schema_config)
    
    @property
    def schemas(self) -> Dict[str, 'BaseSchema']:
        return self._schemas
    
    @property
    def variables(self) -> List[ExtractionVariable]:
        """Get variables from all sub-schemas."""
        all_variables = []
        for schema in self._schemas.values():
            all_variables.extend(schema.variables)
        return all_variables
    
    def create_pydantic_schema(self) -> Type[BaseModel]:
        """Create a combined Pydantic schema for all sub-schemas."""
        from pydantic import BaseModel, Field

        fields = {}
        annotations = {}

        for name, schema in self.schemas.items():
            sub_schema_model = schema.create_pydantic_schema()
            annotations[name] = sub_schema_model
            fields[name] = Field(..., description=f"{name} extraction results")

        # Dynamically create the combined schema class
        CombinedSchema = type(
            "MultipleExtractSchema",
            (BaseModel,),
            {
                "__annotations__": annotations,
                **fields
            }
        )
        return CombinedSchema
    
    def create_prompt(self, text: str, context: Dict[str, Any] | None = None) -> str:
        """Create combined prompt for multiple schemas."""
        prompts = []
        for name, schema in self.schemas.items():
            schema_prompt = schema.create_prompt(text, context)
            prompts.append(f"## {name.upper()} EXTRACTION:\n{schema_prompt}")
        
        return "\n\n".join(prompts)
    
    def validate_and_parse_response_to_exploded_dataframe(self, response: Any, text_chunk: str) -> pd.DataFrame:
        """Validate and parse multiple schema responses into DataFrame."""
        all_data = []
        
        for name, schema in self.schemas.items():
            schema_response = getattr(response, name, None) if hasattr(response, name) else response
            
            schema_df = schema.validate_and_parse_response_to_exploded_dataframe(schema_response, text_chunk)
            if not schema_df.empty:
                # Add schema identifier
                schema_df['schema_type'] = name
                all_data.append(schema_df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


class SchemaRegistry:
    """Registry for different schema types - extensible and maintainable."""
    
    def __init__(self):
        self._schemas = {}
        self._register_default_schemas()
    
    def register(self, name: str, schema_class: Type[BaseSchema]):
        """Register a new schema type."""
        self._schemas[name] = schema_class
    
    def create(self, config: Dict[str, Any]) -> BaseSchema:
        """Create schema instance from config."""
        # default to simple schema
        schema_type = config.get("schema_type", "simple")
        
        if schema_type not in self._schemas:
            raise SchemaError(
                f"Unknown schema type: {schema_type}",
                {
                    "schema_type": schema_type,
                    "available_types": list(self._schemas.keys()),
                    "suggestion": f"Use one of: {', '.join(self._schemas.keys())}"
                }
            )
        
        return self._schemas[schema_type](config)
    
    def _register_default_schemas(self):
        """Register built-in schema types."""
        self.register("simple", SimpleSchema)
        self.register("nested", NestedSchema)
        self.register("multiple", MultipleSchema)
    
    def list_available(self) -> List[str]:
        """List all available schema types."""
        return list(self._schemas.keys()) 