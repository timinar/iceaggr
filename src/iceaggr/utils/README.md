# Utilities

Color-coded logging for iceaggr. Original implementation by [Midori Kato](https://github.com/pomidori).

## Usage

```python
from iceaggr.utils import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.debug("Batch shape: (32, 128)")  # Won't show by default
```

Set level: `get_logger(__name__, level=logging.DEBUG)`

Demo: `uv run python scripts/demo_logger.py`
