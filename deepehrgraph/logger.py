""" Logger module. """
import logging
import sys


def get_logger(name=__name__, level=logging.DEBUG) -> logging.Logger:
    """
    Create a logger with the specified name and level.
    Logs will be outputted to stdout.

    :param str name: name of the logger
    :param int level: logging level
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
