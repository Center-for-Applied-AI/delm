from delm.constants import SYSTEM_REGEX_EXTRACTED_KEY

def is_pydantic_model(obj) -> bool:
    """Return True if obj is a Pydantic model instance (v1 or v2)."""
    try:
        from pydantic import BaseModel
        return isinstance(obj, BaseModel)
    except ImportError:
        return False


def is_regex_fallback_response(obj) -> bool:
    """Return True if obj is a dict with the expected regex fallback structure (e.g., has SYSTEM_REGEX_EXTRACTED_KEY)."""
    return isinstance(obj, dict) and SYSTEM_REGEX_EXTRACTED_KEY in obj 