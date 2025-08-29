"""
DELM Schema Manager
==================
Manages schema loading and validation.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from delm.schemas import SchemaRegistry, BaseSchema
from delm.config import SchemaConfig

# Module-level logger
log = logging.getLogger(__name__)


class SchemaManager:
    """Manages schema loading and validation."""
    
    def __init__(self, config: SchemaConfig):
        log.debug("Initializing SchemaManager")
        # Ensure spec_path is always a Path object
        if isinstance(config.spec_path, str):
            self.spec_path = Path(config.spec_path)
        else:
            self.spec_path = config.spec_path
        self.prompt_template: str = config.prompt_template
        self.system_prompt: str = config.system_prompt
        log.debug(f"SchemaManager config: spec_path={self.spec_path}, prompt_template_length={len(self.prompt_template)}, system_prompt_length={len(self.system_prompt)}")
        self.schema_registry = SchemaRegistry()
        self.extraction_schema = self._load_schema()
        log.debug("SchemaManager initialized successfully")
    
    def _load_schema(self) -> BaseSchema:
        """Load and validate schema from spec file."""
        log.debug(f"Loading schema from spec file: {self.spec_path}")
        schema_config = self._load_schema_spec(self.spec_path) # type: ignore
        log.debug(f"Schema spec loaded with {len(schema_config)} top-level keys: {list(schema_config.keys())}")
        
        schema = self.schema_registry.create(schema_config)
        
        log.debug(f"Schema loaded successfully: {type(schema).__name__}")
        return schema
    
    def get_extraction_schema(self) -> BaseSchema:
        """Get the loaded extraction schema."""
        log.debug(f"Getting extraction schema: {type(self.extraction_schema).__name__}")
        return self.extraction_schema
    
    @staticmethod
    def _load_schema_spec(path: Path) -> Dict[str, Any]:
        """Load schema specification from YAML file as a dict.
        
        Args:
            path: The path to the schema specification file.

        Returns:
            A dictionary of the schema specification.

        Raises:
            ValueError: If the schema file format is not supported.
        """
        import yaml
        
        log.debug(f"Loading schema specification from: {path}")
        log.debug(f"File suffix: {path.suffix}")

        if path.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError(f"Unsupported schema file format: {path.suffix}")
        
        log.debug("Loading YAML schema specification")
        content = yaml.safe_load(path.read_text()) or {}
        log.debug(f"YAML schema loaded successfully with {len(content)} top-level keys")
        return content