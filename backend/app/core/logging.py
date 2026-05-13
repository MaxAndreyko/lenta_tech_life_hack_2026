from __future__ import annotations

import logging
from datetime import datetime

from backend.app.core.settings import settings

_LOG_FILE_PATH = settings.storage.logs_dir / f"app_{datetime.now():%Y%m%d_%H%M%S}.log"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console and file logging once for scripts and services."""

    settings.storage.logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not _has_handler(root_logger, logging.StreamHandler):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if not _has_file_handler(root_logger, _LOG_FILE_PATH):
        file_handler = logging.FileHandler(filename=_LOG_FILE_PATH, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _has_handler(logger: logging.Logger, handler_type: type[logging.Handler]) -> bool:
    """Return whether logger already has a handler of the requested type."""

    return any(isinstance(handler, handler_type) for handler in logger.handlers)


def _has_file_handler(logger: logging.Logger, log_path: object) -> bool:
    """Return whether logger already writes to the requested log file."""

    return any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == str(log_path)
        for handler in logger.handlers
    )
