# src/logging_utils.py
"""Logging configuration shared by the CLI and the benchmark orchestrator."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class ColoredFormatter(logging.Formatter):
    """A logging formatter that adds color to log messages based on their level and highlights specific words."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m" # Bold red
    }
    
    RESET = "\033[0m"
    HIGHLIGHT = "\033[1;35m"  # Bold magenta

    HIGHLIGHT_WORDS = frozenset([
        "random",
        "dice",
        "train",
        "valid",
        "test",
        "Completed Stage",
    ])


    def __init__(self, fmt=None, datefmt=None, highlight_words=None):
        super().__init__(fmt, datefmt)
        self.highlight_words = self.HIGHLIGHT_WORDS if highlight_words is None else highlight_words

    def format(self, record):
        # First, create the normal formatted message
        message = super().format(record)

        # Highlight specific words
        for word in self.highlight_words:
            message = re.sub(
                re.escape(word),
                f"{self.HIGHLIGHT}{word}{self.RESET}",
                message,
                flags=re.IGNORECASE
            )

        # Color the complete message depending on log level
        color = self.COLORS.get(record.levelno, self.RESET)

        return f"{color}{message}{self.RESET}"


# logger = logging.getLogger("my_application")
# logger.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()



def configure_logging(log_file: Path | None = None, *, level: int = logging.INFO):
    """Attach a stream handler and, when given, a per-run file handler."""
    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file is None:
        return None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    return handler
