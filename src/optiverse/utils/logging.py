# src/optiverse/utils/logging.py
from __future__ import annotations

import logging
import sys

_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a Console (stdout) handler once.
    Safe to call multiple times (won't add duplicate handlers).
    """
    root = logging.getLogger()
    # Only add our handler if none exist
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_DEF_FMT))
        root.addHandler(handler)
    root.setLevel(level)