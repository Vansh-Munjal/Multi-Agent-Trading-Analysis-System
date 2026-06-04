"""
logger.py
---------
Centralised logging configuration for the Multi-Agent Trading Analysis System.

Usage in any module:
    import logging
    logger = logging.getLogger(__name__)

Call setup_logging() once at application startup (in main.py before imports).

Log level can be controlled via the LOG_LEVEL environment variable:
    LOG_LEVEL=DEBUG streamlit run main.py   # see everything
    LOG_LEVEL=INFO  streamlit run main.py   # normal operation (default)
    LOG_LEVEL=WARNING streamlit run main.py # only warnings and errors
"""

from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    """
    Configure root logger with console + file handlers.

    - Console (StreamHandler): logs appear in the terminal where you ran
      `streamlit run main.py`.
    - File (FileHandler): logs are appended to `trading_analysis.log` in
      the project root. Open it with `cat trading_analysis.log` or follow
      live with `tail -f trading_analysis.log`.
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),                                        # → terminal
        logging.FileHandler("trading_analysis.log", mode="a"),         # → file
    ]

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )

    # Silence overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised | level=%s | output=terminal+trading_analysis.log",
        log_level_str,
    )
