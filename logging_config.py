"""Centralised logging configuration."""
import logging
import sys

import config


def setup_logging() -> None:
    """Configure root logger to write to stdout with a sensible format.

    Should be called once at application startup.
    """
    level = getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers in reload scenarios
    root.handlers = [handler]

    # Quiet down a couple of noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
