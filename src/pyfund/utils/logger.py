# src/pyfundlib/utils/logger.py
from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


# -----------------------------
# Core Logger
# -----------------------------
logger = logging.getLogger("pyfundlib")
logger.propagate = False  # Prevent double logging if user has their own handlers

# Avoid duplicate handlers
if not logger.handlers:
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


# -----------------------------
# Logger getter
# -----------------------------
def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a child logger.

    Usage:
        logger = get_logger(__name__)       # e.g., pyfundlib.data.fetcher
        logger = get_logger("my_strategy")  # custom name
    """
    if name is None:
        return logger
    return logging.getLogger(f"pyfundlib.{name}" if not name.startswith("pyfundlib") else name)


# -----------------------------
# Rotating file handler
# -----------------------------
def add_file_handler(
    log_dir: str = "logs",
    filename: str = "pyfundlib.log",
    level: int = logging.DEBUG,
    max_mb: int = 10,
    backup_count: int = 5,
) -> None:
    """
    Add a rotating file handler.

    Parameters
    ----------
    log_dir : str
        Directory for log files.
    filename : str
        Log file name.
    level : int
        Logging level (DEBUG, INFO, etc.).
    max_mb : int
        Maximum file size in MB before rotation.
    backup_count : int
        Number of backup files to keep.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    full_path = log_path / filename

    file_handler = RotatingFileHandler(
        full_path,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"File logging enabled → {full_path}")


# -----------------------------
# Environment-level override
# -----------------------------
env_level = os.getenv("PYFUNDLIB_LOG_LEVEL")
if env_level in ("DEBUG", "INFO", "WARNING", "ERROR"):
    logger.setLevel(env_level)
    logger.info(f"Log level set to {env_level} via environment variable")


# -----------------------------
# Quick test when run as main
# -----------------------------
if __name__ == "__main__":
    add_file_handler()
    test_logger = get_logger("test")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
