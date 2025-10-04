"""
Custom logger configuration with colored output for iceaggr.

Original implementation by Midori Kato (https://github.com/pomidori).
Adapted for the iceaggr project.
"""

import logging
import sys


class CustomFormatter(logging.Formatter):
    """Custom formatter with color-coded output for different log levels."""

    # Define color codes
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # Define log format
    FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

    def __init__(self):
        """Initialize the formatter with the default format string."""
        super().__init__(self.FORMAT)

    def format(self, record):
        """
        Format the log record with appropriate colors based on log level.

        Args:
            record: LogRecord instance to format

        Returns:
            Formatted log string with color codes
        """
        # Add colors to the levelname based on log level
        if record.levelno == logging.ERROR:
            record.levelname = f"{self.RED}{record.levelname}{self.RESET}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{self.YELLOW}{record.levelname}{self.RESET}"
        elif record.levelno == logging.INFO:
            record.levelname = f"{self.GREEN}{record.levelname}{self.RESET}"
        elif record.levelno == logging.DEBUG:
            record.levelname = f"{self.BLUE}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name, level=logging.INFO):
    """
    Create and configure a logger with color-coded output.

    This function creates a logger instance with a custom formatter that adds
    color-coding to log messages based on their severity level. The logger is
    configured to output to stderr (standard for logging) and will not add
    duplicate handlers if called multiple times with the same name.

    Args:
        name: Name of the logger (typically __name__ of the calling module)
        level: Logging level (default: logging.INFO). Can be any of:
               - logging.DEBUG
               - logging.INFO
               - logging.WARNING
               - logging.ERROR
               - logging.CRITICAL

    Returns:
        Configured logging.Logger instance

    Example:
        >>> from iceaggr.utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        >>> logger.debug("This is a debug message")  # Won't show with default level
        >>> logger = get_logger(__name__, level=logging.DEBUG)  # Enable debug
        >>> logger.debug("Now debug messages will show")
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Avoid adding multiple handlers
        logger.setLevel(level)
        # Use stderr as it's the standard stream for logging
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    else:
        logger.setLevel(level)  # Update the level if the logger already exists

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger
