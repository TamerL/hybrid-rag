"""Centralized logging setup."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure root or named logger with rich formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger
