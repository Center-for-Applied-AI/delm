"""
DELM Schema System
==================
Unified schema system for data extraction with extensible registry pattern.
Supports simple, nested, and multiple schema types with progressive complexity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import pandas as pd
from pydantic import BaseModel, Field

from ..models import ExtractionVariable
from ..exceptions import SchemaError
from ..constants import DEFAULT_PROMPT_TEMPLATE


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
    def parse_response(self, response: Any, text_chunk: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse LLM response into DataFrame."""
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
        
        # Add fields dynamically with proper annotations
        for var in self.variables:
            if var.data_type == 'string':
                field_type = List[str]
            elif var.data_type == 'number':
                field_type = List[float]
            elif var.data_type == 'integer':
                field_type = List[int]
            elif var.data_type == 'date':
                field_type = List[str]
            else:
                field_type = List[str]

            # Build description, including allowed values if present
            description = var.description
            if var.allowed_values:
                description += f" (must be one of: {', '.join(var.allowed_values)})"

            if var.required:
                # Required field: must be present in input
                fields[var.name] = Field(..., description=description)  # required
            else:
                # Optional field: defaults to empty list if missing
                fields[var.name] = Field(default_factory=list, description=description)  # optional

            annotations[var.name] = field_type
        
        # Create the schema class
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
    
    def parse_response(self, response: Any, text_chunk: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse simple response into DataFrame."""
        if response is None:
            return pd.DataFrame()
        
        if isinstance(response, dict):
            row = {'text_chunk': text_chunk}
            row.update(response)
            
            if metadata:
                row.update(metadata)
            
            return pd.DataFrame([row])
        
        return pd.DataFrame()


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
        # Create item schema class with proper field definitions
        item_fields = {}
        item_annotations = {}
        
        for var in self.variables:
            field_type = self._get_field_type(var.data_type, var.required)
            
            # Create field with proper defaults
            description = var.description
            if var.allowed_values:
                description += f" (allowed values: {', '.join(var.allowed_values)})"
            
            if var.required:
                field = Field(description=description)
            else:
                field = Field(default=None, description=description)
            
            item_fields[var.name] = field
            item_annotations[var.name] = field_type
        
        # Create the item schema class
        DynamicItemSchema = type(
            'DynamicItemSchema',
            (BaseModel,),
            {
                '__annotations__': item_annotations,
                **item_fields
            }
        )
        
        # Create container schema class
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
    
    def parse_response(self, response: Any, text_chunk: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse nested response into DataFrame with validation."""
        if response is None:
            return pd.DataFrame()
        
        instances = []
        
        # Handle both Pydantic models and dictionaries
        if hasattr(response, self.container_name):
            # It's a Pydantic model
            instances = getattr(response, self.container_name)
        elif isinstance(response, dict) and self.container_name in response:
            # It's a dictionary with the container key
            instances = response[self.container_name]
        else:
            return pd.DataFrame()
        
        if not instances:
            return pd.DataFrame()
        
        data = []
        text_chunk_lower = text_chunk.lower()
        
        for instance in instances:
            row: Dict[str, Any] = {'text_chunk': text_chunk}
            
            # Extract all fields from the instance
            # Handle both Pydantic models and dictionaries
            if hasattr(instance, 'model_dump'):
                # It's a Pydantic model
                instance_dict = instance.model_dump()
            elif hasattr(instance, 'dict'):
                # It's a Pydantic model (older version)
                instance_dict = instance.dict()
            elif isinstance(instance, dict):
                # It's already a dictionary
                instance_dict = instance
            else:
                # Unknown type, skip this instance
                continue
                
            for field_name, field_value in instance_dict.items():
                if field_value is not None:
                    # Find the corresponding variable definition for validation
                    var_def = next((v for v in self.variables if v.name == field_name), None)
                    
                    if var_def and var_def.validate_in_text and isinstance(field_value, str):
                        # Validate that the extracted value appears in the original text
                        if field_value.lower() not in text_chunk_lower:
                            print(f"Warning: Extracted value '{field_value}' for field '{field_name}' not found in text chunk")
                            continue  # Skip this field if validation fails
                    
                    row[field_name] = field_value
            
            # Add metadata if provided
            if metadata:
                row.update(metadata)
            
            data.append(row)
        
        return pd.DataFrame(data)


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
    
    def parse_response(self, response: Any, text_chunk: str, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse multiple schema responses into DataFrame."""
        all_data = []
        
        for name, schema in self.schemas.items():
            # Try to extract response for this schema
            schema_response = getattr(response, name, None) if hasattr(response, name) else response
            
            schema_df = schema.parse_response(schema_response, text_chunk, metadata)
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