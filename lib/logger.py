"""Logging utility."""

import logging
import sys

def setup_logger(name="ailm-sec", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(console)
    return logger

log = setup_logger()
