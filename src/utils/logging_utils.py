"""Logging helpers for repeatable experiment runs."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_dir


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def make_run_name(prefix: str = "run") -> str:
    """Create a readable timestamped run name."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = prefix.strip().replace(" ", "_") or "run"
    return f"{safe_prefix}_{timestamp}"


def setup_logging(
    *,
    log_dir: str | Path,
    run_name: str,
    level: str | int = "INFO",
    file_enabled: bool = True,
) -> logging.Logger:
    """Configure root logging with console and optional file handlers."""

    numeric_level = _coerce_log_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if file_enabled:
        directory = ensure_dir(log_dir)
        file_handler = logging.FileHandler(directory / f"{run_name}.log", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger("acgd_nci_forecasting")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a project logger."""

    logger_name = "acgd_nci_forecasting" if name is None else f"acgd_nci_forecasting.{name}"
    return logging.getLogger(logger_name)


def _coerce_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if normalized not in logging._nameToLevel:
        raise ValueError(f"Unknown logging level: {level}")
    return int(logging._nameToLevel[normalized])

