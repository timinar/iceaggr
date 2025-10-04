# Logger Utility

Color-coded logging utility for iceaggr. Original implementation by [Midori Kato](https://github.com/pomidori).

## Usage

```python
from iceaggr.utils import get_logger

logger = get_logger(__name__)
logger.info("Loading data...")
logger.warning("Cache miss")
logger.error("Failed to load batch")
```

## Features

- **Color-coded output**: DEBUG (blue), INFO (green), WARNING (yellow), ERROR (red)
- **Standard format**: `timestamp | level | message`
- **No duplicate handlers**: Safe to call `get_logger()` multiple times
- **Per-module control**: Each module can have its own log level

## Log Levels

```python
import logging
from iceaggr.utils import get_logger

# Default: INFO level
logger = get_logger(__name__)

# Set to DEBUG to see all messages
logger = get_logger(__name__, level=logging.DEBUG)

# Set to WARNING to only see warnings and errors
logger = get_logger(__name__, level=logging.WARNING)
```

## Demo

Run the demo to see colored output:
```bash
uv run python scripts/demo_logger.py
```
