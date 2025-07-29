"""
DELM Logging Utility
===================
Centralized logging configuration for the DELM package.

This module provides a single point of configuration for all DELM logging,
following Python best practices for libraries.
"""

from __future__ import annotations
import logging
import logging.config
from pathlib import Path
from delm.constants import (
    DEFAULT_CONSOLE_LOG_LEVEL, 
    DEFAULT_FILE_LOG_LEVEL,
    DEFAULT_LOG_DIR,
)

# Global flag to track if logging has been configured
_configured = False


def configure(
    *,
    console_level: str = DEFAULT_CONSOLE_LOG_LEVEL,
    file_dir: str | Path | None = DEFAULT_LOG_DIR,
    file_name: str | None = None, # if None, no file handler is will be added
    file_level: str = DEFAULT_FILE_LOG_LEVEL,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    disable_existing: bool = False,
    force: bool = False,
) -> None:
    """
    Configure logging for the *delm* package and its children.
    
    This function can only be called once unless force=True is specified.
    Subsequent calls without force=True will be ignored.

    Parameters
    ----------
    console_level : str
        Level for stderr (default INFO).
    file : str | Path | None
        Path to a log file. ``None`` = no file handler.
    file_level : str
        Level for the file handler (default DEBUG).
    fmt : str
        Log‑record format.
    disable_existing : bool
        If True, wipe out any handlers the application has already set up.
    force : bool
        If True, force re-configuration even if already configured (default False).
    """
    global _configured
    
    if _configured and not force:
        # Use a temporary logger to avoid circular dependency
        temp_logger = logging.getLogger("delm.logging")
        temp_logger.debug("Logging already configured, ignoring configuration request")
        return
    
    handlers: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": "basic",
            "stream": "ext://sys.stderr",
        }
    }

    if file_name:
        file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": file_level,
            "formatter": "basic",
            "filename": str(file_dir / file_name),
            "maxBytes": 5 * 1024 * 1024,  # 5 MB per slice
            "backupCount": 3,
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": disable_existing,
            "formatters": {"basic": {"format": fmt}},
            "handlers": handlers,
            "loggers": {
                "delm": {
                    "handlers": list(handlers),
                    "level": "DEBUG",        # Capture everything; handlers filter.
                    "propagate": False,
                }
            },
        }
    )
    
    _configured = True
    logger = logging.getLogger("delm.logging")
    if file_name:
        full_path = file_dir / file_name
        logger.info("Logging configured successfully - console_level: %s, file: %s", console_level, full_path)
    else:
        logger.info("Logging configured successfully - console_level: %s, file: None", console_level)


def is_configured() -> bool:
    """Check if logging has been configured."""
    return _configured


def reset() -> None:
    """Reset the configuration state (for testing purposes)."""
    global _configured
    _configured = False 