"""
DELM Schema Manager
==================
Manages schema loading and validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import SchemaRegistry, BaseSchema


class SchemaManager:
    """Manages schema loading and validation."""
    
    def __init__(self, config):
        self.config = config
        self.schema_registry = SchemaRegistry()
        self.extraction_schema = self._load_schema()
    
    def _load_schema(self) -> Optional[BaseSchema]:
        """Load and validate schema from spec file."""
        if not self.config.spec_path or not self.config.spec_path.exists():
            return None
        
        schema_config = self._load_schema_spec(self.config.spec_path)
        
        # Handle both direct extraction config and nested extraction config
        if 'extraction' in schema_config:
            return self.schema_registry.create(schema_config['extraction'])
        else:
            # Assume the entire config is the extraction schema
            return self.schema_registry.create(schema_config)
    
    def get_extraction_schema(self) -> Optional[BaseSchema]:
        """Get the loaded extraction schema."""
        return self.extraction_schema
    
    @staticmethod
    def _load_schema_spec(path: Path) -> Dict[str, Any]:
        """Load schema specification from YAML or JSON file."""
        import yaml
        import json
        
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text()) or {}
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text())
        raise ValueError("Schema spec must be YAML or JSON") 