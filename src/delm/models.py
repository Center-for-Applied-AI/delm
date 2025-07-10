"""
DELM Shared Models
=================
Shared data models to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionVariable:
    """Represents a variable to be extracted from text."""
    
    name: str
    description: str
    data_type: str
    required: bool = False
    allowed_values: Optional[List[str]] = None
    validate_in_text: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionVariable':
        """Create ExtractionVariable from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            data_type=data['data_type'],
            required=data.get('required', False),
            allowed_values=data.get('allowed_values'),
            validate_in_text=data.get('validate_in_text', False)
        ) 