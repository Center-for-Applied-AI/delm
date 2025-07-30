"""
DELM Schema Manager
==================
Manages schema loading and validation.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from delm.schemas import SchemaRegistry, BaseSchema
from delm.exceptions import SchemaError, FileError
from delm.config import SchemaConfig

# Module-level logger
log = logging.getLogger(__name__)


class SchemaManager:
    """Manages schema loading and validation."""
    
    def __init__(self, config: SchemaConfig):
        log.debug("Initializing SchemaManager")
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
        """Load schema specification from YAML or JSON file as a dict."""
        import yaml
        import json
        
        log.debug(f"Loading schema specification from: {path}")
        log.debug(f"File suffix: {path.suffix}")
        
        try:
            if path.suffix.lower() in {".yml", ".yaml"}:
                log.debug("Loading YAML schema specification")
                content = yaml.safe_load(path.read_text()) or {}
                log.debug(f"YAML schema loaded successfully with {len(content)} top-level keys")
                return content
            if path.suffix.lower() == ".json":
                log.debug("Loading JSON schema specification")
                content = json.loads(path.read_text())
                log.debug(f"JSON schema loaded successfully with {len(content)} top-level keys")
                return content
            log.error(f"Unsupported schema file format: {path.suffix}")
            raise SchemaError(
                f"Unsupported schema file format: {path.suffix}",
                {
                    "file_path": str(path),
                    "supported_formats": [".yml", ".yaml", ".json"],
                    "suggestion": "Use YAML (.yml/.yaml) or JSON (.json) format"
                }
            )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            log.error(f"Failed to parse schema specification file: {path}, error: {e}")
            raise SchemaError(
                f"Failed to parse schema specification file: {path}",
                {"file_path": str(path), "parse_error": str(e)}
            ) from e
        except Exception as e:
            log.error(f"Failed to load schema specification: {path}, error: {e}")
            raise SchemaError(f"Failed to load schema specification: {path}", {"file_path": str(path)}) from e 