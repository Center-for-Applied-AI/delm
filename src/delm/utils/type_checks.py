"""Lightweight runtime type checks for optional dependencies."""

def is_pydantic_model(obj) -> bool:
    """Return True if obj is a Pydantic model instance (v1 or v2)."""
    try:
        from pydantic import BaseModel
        return isinstance(obj, BaseModel)
    except ImportError:
        return False