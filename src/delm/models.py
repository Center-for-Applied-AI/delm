"""
DELM Shared Models
=================
Shared data models to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionVariable:
    """Represents a variable to be extracted from text.

    Args:
        name: The name of the variable.
        description: The description of the variable.
        data_type: The data type of the variable.
            "string": Text values (default)
            "number": Floating-point numbers
            "integer": Whole numbers
            "boolean": True/False values
            "[string]", "[number]", etc.: Lists of the specified type
        required: Whether the variable is required to return a full schema result.
            True: Will only return the schema container result if the variable is present.
            False: Will return the schema containing this variable even if the variable is missing.
        allowed_values: The allowed values for the variable.
            List of strings (e.g., ["oil", "gas", "copper", "gold", "silver", "steel", "aluminum"]).
            If provided, the variable must be one of these values.
        validate_in_text: Whether to require the exact value of the variable appears in text.
    """

    name: str
    description: str
    data_type: str
    required: bool = False
    allowed_values: Optional[List[str]] = None
    validate_in_text: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionVariable":
        """Create ExtractionVariable from dictionary."""
        # Handle case where data_type is a list (e.g., [string]) - convert to string format
        data_type = data["data_type"]
        if isinstance(data_type, list):
            data_type = f"[{data_type[0]}]"  # Convert [string] to "[string]"

        return cls(
            name=data["name"],
            description=data["description"],
            data_type=data_type,
            required=data.get("required", False),
            allowed_values=data.get("allowed_values"),
            validate_in_text=data.get("validate_in_text", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExtractionVariable to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "required": self.required,
            "allowed_values": self.allowed_values,
            "validate_in_text": self.validate_in_text,
        }

    def is_list(self) -> bool:
        """Return True if the ExtractionVariable describes a list.

        Returns:
            True if the ExtractionVariable describes a list, False otherwise.
        """
        return self.data_type.startswith("[") and self.data_type.endswith("]")
