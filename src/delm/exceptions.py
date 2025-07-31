"""
DELM Exception Hierarchy
========================
Comprehensive exception classes for DELM operations with context support.
"""

from typing import Any, Dict, Optional


class DELMError(Exception):
    """Base exception for DELM operations."""
    
    # def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
    #     super().__init__(message)
    #     self.context = context or {}
    
    # def __str__(self) -> str:
    #     context_str = ""
    #     if self.context:
    #         context_items = [f"{k}={v}" for k, v in self.context.items()]
    #         context_str = f" (Context: {', '.join(context_items)})"
    #     return f"{super().__str__()}{context_str}"


class ExperimentManagementError(DELMError):
    """Raised when experiment management operations fail."""
    pass

class InstructorError(DELMError):
    """Raised when Instructor API calls fail."""
    pass

class ProcessingError(DELMError):
    """Raised when LLM processing fails."""
    pass

# class ConfigurationError(DELMError):
#     """Raised when configuration operations fail."""
#     pass


# class DataError(DELMError):
#     """Raised when data operations fail."""
#     pass




# class SchemaError(DELMError):
#     """Raised when schema operations fail."""
#     pass


# class ValidationError(DELMError):
#     """Raised when data validation fails."""
#     pass


# class FileError(DELMError):
#     """Raised when file operations fail."""
#     pass


# class APIError(DELMError):
#     """Raised when API calls fail."""
#     pass


# class DependencyError(DELMError):
#     """Raised when required dependencies are missing."""
#     pass


# class ExperimentError(DELMError):
#     """Raised when experiment operations fail."""
#     pass 