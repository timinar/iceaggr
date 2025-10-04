# Testing Guide for iceaggr

## Quick Start

```bash
# Run all tests
uv run pytest

# Run only fast unit tests
uv run pytest tests/unit/

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=src/iceaggr

# Run specific test file
uv run pytest tests/unit/test_dataset.py

# Run specific test
uv run pytest tests/unit/test_dataset.py::TestIceCubeDataset::test_dataset_length

# Run tests matching a pattern
uv run pytest -k "sensor_geometry"
```

## Test Organization

```
tests/
├── unit/                    # Fast tests, no heavy I/O
│   ├── test_data_integrity.py   # Data file validation
│   └── test_dataset.py          # Dataset functionality
├── integration/             # Tests requiring multiple components
└── benchmarks/              # Performance measurements
    └── test_dataset_performance.py
```

## Test Types

### Unit Tests (`tests/unit/`)
- **Fast** (< 1 second per test ideally)
- Test single functions/classes in isolation
- Use small test datasets (`max_events=10-100`)
- Mock external dependencies when possible

Example:
```python
def test_dataset_length():
    """Test that dataset returns correct length."""
    dataset = IceCubeDataset(max_events=50)
    assert len(dataset) == 50
```

### Integration Tests (`tests/integration/`)
- Test components working together
- May be slower (load real data)
- Test end-to-end workflows

Example:
```python
def test_full_training_pipeline():
    """Test complete training loop with real data."""
    # Creates dataloader, model, trains for 1 epoch
    pass
```

### Benchmarks (`tests/benchmarks/`)
- Measure performance (throughput, memory usage)
- Not part of regular test suite
- Run manually to validate optimization

## Writing Good Tests

### 1. Test One Thing
❌ Bad:
```python
def test_everything():
    dataset = IceCubeDataset()
    assert len(dataset) > 0
    assert dataset[0] is not None
    assert dataset[0]["pulse_features"].shape[1] == 4
```

✅ Good:
```python
def test_dataset_length():
    dataset = IceCubeDataset(max_events=10)
    assert len(dataset) == 10

def test_pulse_features_shape():
    dataset = IceCubeDataset(max_events=10)
    event = dataset[0]
    n_pulses = event["n_pulses"].item()
    assert event["pulse_features"].shape == (n_pulses, 4)
```

### 2. Use Descriptive Names
- `test_<function>_<scenario>_<expected_behavior>`
- `test_dataset_empty_returns_error`
- `test_collate_preserves_total_pulses`

### 3. Add Docstrings
```python
def test_sensor_geometry_first_rows():
    """Test first 10 rows of sensor_geometry.csv match expected values.

    This validates that the geometry file hasn't been corrupted
    and matches the official IceCube detector configuration.
    """
    # test code...
```

### 4. Arrange-Act-Assert Pattern
```python
def test_collate_flattens_pulses():
    # Arrange: Set up test data
    dataset = IceCubeDataset(max_events=10)
    batch = [dataset[i] for i in range(3)]

    # Act: Execute function
    result = collate_variable_length(batch)

    # Assert: Verify behavior
    total_pulses = sum(e["n_pulses"].item() for e in batch)
    assert result["pulse_features"].shape[0] == total_pulses
```

### 5. Test Edge Cases
```python
def test_dataset_single_event():
    """Test dataset with only one event."""
    dataset = IceCubeDataset(max_events=1)
    assert len(dataset) == 1

def test_dataset_very_large_event():
    """Test handling of events with many pulses."""
    # Find event with >10K pulses and test it loads correctly
    pass

def test_collate_empty_batch():
    """Test collate with empty batch."""
    with pytest.raises(ValueError):
        collate_variable_length([])
```

## Common Patterns

### Using Fixtures
```python
# In conftest.py or test file
@pytest.fixture
def small_dataset():
    """Reusable small dataset for tests."""
    return IceCubeDataset(max_events=50)

# In test
def test_something(small_dataset):
    assert len(small_dataset) == 50
```

### Parametrized Tests
```python
@pytest.mark.parametrize("max_events", [1, 10, 100, 1000])
def test_dataset_various_sizes(max_events):
    """Test dataset works with various sizes."""
    dataset = IceCubeDataset(max_events=max_events)
    assert len(dataset) == max_events
```

### Testing Exceptions
```python
def test_invalid_split_raises_error():
    """Test that invalid split raises AssertionError."""
    with pytest.raises(AssertionError, match="split must be"):
        IceCubeDataset(split="invalid")
```

### Testing Approximate Values
```python
def test_azimuth_in_range():
    """Test azimuth is in [0, 2π]."""
    dataset = IceCubeDataset(max_events=10)
    azimuth = dataset[0]["target"][0]

    assert 0 <= azimuth <= 2 * np.pi + 0.01  # Small tolerance
```

## Coverage

Check what code is tested:
```bash
# Generate coverage report
uv run pytest --cov=src/iceaggr --cov-report=html

# View report in browser
open htmlcov/index.html
```

**Aim for**:
- Critical code (data loading, model forward pass): >90%
- Utility functions: >80%
- Overall: >80%

## Continuous Integration (Future)

When we add CI, tests will run automatically on:
- Every push to GitHub
- Every pull request
- Before merging to main

## Debugging Failed Tests

```bash
# Run with more verbose output
uv run pytest -vv

# Show print statements
uv run pytest -s

# Stop at first failure
uv run pytest -x

# Drop into debugger on failure
uv run pytest --pdb

# Run last failed tests only
uv run pytest --lf
```

## Performance Testing

For performance-critical code:
```python
import time

def test_dataloader_throughput():
    """Measure dataloader throughput."""
    dataloader = get_dataloader(batch_size=32, max_events=1000)

    start = time.time()
    for batch in dataloader:
        pass
    elapsed = time.time() - start

    events_per_sec = 1000 / elapsed
    assert events_per_sec > 1000  # Should be >1000 events/sec
```

## Further Reading

- **Pytest docs**: https://docs.pytest.org/
- **Testing best practices**: See `notes/testing_guide.md`
- **Example tests**: Look at existing tests in `tests/unit/`

## Questions?

Check `notes/testing_guide.md` for detailed philosophy and practices.
