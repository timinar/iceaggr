"""Pytest configuration and shared fixtures."""

import pytest
import yaml
from pathlib import Path


@pytest.fixture(scope="session")
def data_config():
    """Load data config for tests."""
    config_path = Path(__file__).parent.parent / "src/iceaggr/data/data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def data_root(data_config):
    """Get data root directory."""
    return Path(data_config["data"]["root"])


@pytest.fixture(scope="session")
def train_dir(data_config):
    """Get training data directory."""
    return Path(data_config["data"]["train"])


@pytest.fixture(scope="session")
def test_dir(data_config):
    """Get test data directory."""
    return Path(data_config["data"]["test"])
