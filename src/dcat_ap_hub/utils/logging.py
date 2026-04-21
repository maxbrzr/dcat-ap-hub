"""Shared package logger configuration."""

import logging

# All modules in the package use this logger name so applications can
# override behavior once (handlers/levels/format) in a single place.
logger = logging.getLogger("dcat-ap-hub")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    # Attach a default stream handler only once to avoid duplicate logs.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
