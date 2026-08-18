"""
Centralized logging setup — writes to both console and a timestamped
file under logs/, so pipeline/benchmark runs can be inspected after the
fact (e.g. grep "Dropping clause" to audit rubric-violation rate per
model run — see clause_extractor.py's risk_level validation).

Usage (call ONCE, at the top of an entry-point script only — never
inside a module that might be imported by tests, since this reconfigures
the ROOT logger and would affect unrelated test runs):
    from config.logging_config import setup_logging
    setup_logging("run_pipeline")
"""
import logging
import os
from datetime import datetime

_LOG_DIR = "logs"


def setup_logging(name_prefix: str, level: int = logging.INFO) -> str:
    """
    Configures root logger with a console handler + a file handler.
    File name: logs/{name_prefix}_{YYYYMMDD_HHMMSS}.log — one file per run.

    Returns the log file path (useful if the caller wants to print it).
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(_LOG_DIR, f"{name_prefix}_{timestamp}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # avoid duplicate handlers if called twice accidentally
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.info("Logging initialized. File: %s", log_path)
    return log_path
