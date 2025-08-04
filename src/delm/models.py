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
        # Handle case where data_type is a list (e.g., [string]) - convert to string format
        data_type = data['data_type']
        if isinstance(data_type, list):
            data_type = f"[{data_type[0]}]"  # Convert [string] to "[string]"
        
        return cls(
            name=data['name'],
            description=data['description'],
            data_type=data_type,
            required=data.get('required', False),
            allowed_values=data.get('allowed_values'),
            validate_in_text=data.get('validate_in_text', False)
        ) 
    
    def is_list(self) -> bool:
        """Return True if the ExtractionVariable describes a list.
        
        Returns:
            True if the ExtractionVariable describes a list, False otherwise.
        """
        return self.data_type.startswith("[") and self.data_type.endswith("]")