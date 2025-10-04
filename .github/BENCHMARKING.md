# Benchmarking & Performance Testing Guide

This document clarifies where to put performance-related code in the iceaggr project.

## Two Types of Performance Testing

### 1. Performance Regression Tests → `tests/benchmarks/`

**Purpose**: Ensure performance doesn't degrade as the codebase evolves

**Characteristics**:
- Run via pytest: `uv run pytest tests/benchmarks/`
- Fast execution (<5 minutes total)
- Assert specific performance requirements
- Part of CI/CD pipeline (future)
- Named: `test_*.py` or `benchmark_*.py`

**Example**:
```python
# tests/benchmarks/test_dataloader_performance.py
import pytest
from iceaggr.data import get_dataloader

def test_dataloader_throughput():
    """Ensure dataloader maintains >100 events/sec."""
    loader = get_dataloader(batch_size=32, num_workers=4)

    # Measure throughput
    start = time.time()
    n_events = 0
    for batch in loader:
        n_events += batch['batch_size']
        if n_events >= 1000:
            break

    elapsed = time.time() - start
    throughput = n_events / elapsed

    assert throughput > 100, f"Throughput {throughput:.1f} events/sec is too slow"

def test_model_forward_latency():
    """Ensure T1 forward pass <50ms per batch."""
    model = DOMTransformer(d_model=128, n_heads=8).cuda()
    batch = create_mock_batch(n_events=32)

    # Warmup
    _ = model(batch)

    # Measure
    start = time.time()
    _ = model(batch)
    latency = (time.time() - start) * 1000  # ms

    assert latency < 50, f"Forward pass {latency:.1f}ms is too slow"
```

### 2. Exploratory Benchmarks → `scripts/`

**Purpose**: Understand performance characteristics, find bottlenecks, generate reports

**Characteristics**:
- Standalone scripts: `uv run python scripts/benchmark_*.py`
- Generate detailed reports, plots, tables
- Not part of automated testing
- Can run for extended periods (hours)
- Named: `benchmark_*.py`, `analyze_*.py`, `profile_*.py`

**Example**:
```python
# scripts/benchmark_dom_packing.py
"""
Benchmark DOM packing approach with real extreme events.

Generates:
- Memory usage report
- Performance metrics across event sizes
- Comparison with baseline approach
"""

import torch
from iceaggr.models import DOMTransformer
from iceaggr.utils import get_logger

logger = get_logger(__name__)

def benchmark_extreme_events():
    # Load real data
    dataset = load_dataset()

    # Find extreme events
    extreme_events = find_top_k_events(dataset, k=20)

    # Benchmark each
    results = []
    for event in extreme_events:
        metrics = benchmark_single_event(event)
        results.append(metrics)
        logger.info(f"Event {event.id}: {metrics}")

    # Generate report
    generate_report(results)
    plot_memory_scaling(results)

if __name__ == "__main__":
    benchmark_extreme_events()
```

## Decision Matrix

| Use Case | Location | Run Via | Duration | Output |
|----------|----------|---------|----------|--------|
| CI/CD performance gate | `tests/benchmarks/` | `pytest` | <5 min | Pass/Fail |
| Prevent regression in dataloader | `tests/benchmarks/` | `pytest` | <1 min | Pass/Fail |
| Ensure model stays fast | `tests/benchmarks/` | `pytest` | <2 min | Pass/Fail |
| Find memory bottleneck | `scripts/` | Direct | 10-60 min | Report |
| Generate scaling plots | `scripts/` | Direct | 30-120 min | Plots/Tables |
| Profile attention layers | `scripts/` | Direct | Variable | Profiling data |
| Compare architectures | `scripts/` | Direct | Hours | Comparative report |

## Examples in This Project

### Regression Tests (`tests/benchmarks/`)
- `test_dataset_performance.py` - Ensures dataloader maintains throughput
- `benchmark_t1_forward.py` - Checks T1 forward pass latency

### Exploratory Benchmarks (`scripts/`)
- `benchmark_dom_packing.py` - Tests memory usage on extreme events
- `benchmark_io.py` - I/O performance deep dive
- `analyze_packing.py` - Analyzes packing efficiency

## Best Practices

### For Regression Tests (`tests/benchmarks/`)

✅ **Do**:
- Assert clear performance requirements
- Use representative but small datasets
- Keep execution time <5 minutes
- Mark tests with `@pytest.mark.slow` if needed
- Document why the threshold was chosen

❌ **Don't**:
- Generate plots or reports
- Run on full dataset
- Print verbose output
- Take more than a few minutes

### For Exploratory Benchmarks (`scripts/`)

✅ **Do**:
- Generate detailed reports
- Use full/realistic datasets
- Log progress with the project logger
- Save results to `results/` directory
- Document findings in `personal_notes/` or `notes/`

❌ **Don't**:
- Use `assert` statements (not a test)
- Run via pytest
- Make them part of CI/CD

## Running Benchmarks

```bash
# Regression tests (quick checks)
uv run pytest tests/benchmarks/ -v

# Specific regression test
uv run pytest tests/benchmarks/test_dataloader_performance.py

# Exploratory benchmark (detailed analysis)
uv run python scripts/benchmark_dom_packing.py

# With logging level adjustment
LOGLEVEL=DEBUG uv run python scripts/benchmark_dom_packing.py
```

## Adding New Benchmarks

### Adding a Regression Test

1. Create file in `tests/benchmarks/test_*.py`
2. Write test with clear assertion
3. Run: `uv run pytest tests/benchmarks/test_yourtest.py`
4. Commit to repository

### Adding an Exploratory Benchmark

1. Create file in `scripts/benchmark_*.py`
2. Use project logger for output
3. Save results to file or personal_notes
4. Document findings
5. Commit script (not necessarily results)

## Questions?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines or ask in team discussions.
