"""
Demo script to show the logger functionality with colored output.

Run this script to see the logger in action:
    uv run python scripts/demo_logger.py
"""

import logging
import sys
from pathlib import Path

# Add src to path for standalone script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceaggr.utils import get_logger

# Create a logger with default INFO level
logger = get_logger(__name__)

print("\n" + "=" * 70)
print("LOGGER DEMO - Color-coded output")
print("=" * 70 + "\n")

print("1. Default INFO level logger:\n")
logger.info("This is an INFO message (GREEN)")
logger.warning("This is a WARNING message (YELLOW)")
logger.error("This is an ERROR message (RED)")
logger.debug("This DEBUG message won't show (level too low)")

print("\n2. Changing to DEBUG level:\n")
logger_debug = get_logger(__name__, level=logging.DEBUG)
logger_debug.debug("Now DEBUG messages show (BLUE)")
logger_debug.info("INFO still works (GREEN)")
logger_debug.warning("WARNING still works (YELLOW)")
logger_debug.error("ERROR still works (RED)")

print("\n3. Different modules can have different loggers:\n")
module1_logger = get_logger("module1", level=logging.INFO)
module2_logger = get_logger("module2", level=logging.DEBUG)

module1_logger.info("Module 1: INFO level")
module1_logger.debug("Module 1: DEBUG (won't show)")

module2_logger.info("Module 2: INFO level")
module2_logger.debug("Module 2: DEBUG shows!")

print("\n4. Practical example - Loading data:\n")
data_logger = get_logger("iceaggr.data.loader", level=logging.INFO)
data_logger.info("Starting data load...")
data_logger.info("Loading batch 001...")
data_logger.warning("Batch 002 has unusual pulse count")
data_logger.info("Loaded 10,000 events successfully")

print("\n" + "=" * 70)
print("Demo complete! Check the colored output above.")
print("=" * 70 + "\n")
