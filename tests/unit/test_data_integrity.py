"""
Unit tests for IceCube data integrity.

Tests that required data files exist and have expected structure.
"""

import pytest
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


class TestSensorGeometry:
    """Test sensor_geometry.csv integrity."""

    def test_sensor_geometry_exists(self, data_root):
        """Test that sensor_geometry.csv exists."""
        geometry_file = data_root / "sensor_geometry.csv"
        assert geometry_file.exists(), f"sensor_geometry.csv not found at {geometry_file}"

    def test_sensor_geometry_structure(self, data_root):
        """Test sensor_geometry.csv has correct columns and structure."""
        geometry_file = data_root / "sensor_geometry.csv"
        df = pd.read_csv(geometry_file)

        # Check columns
        expected_columns = ['sensor_id', 'x', 'y', 'z']
        assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"

        # Check number of sensors (IceCube has 5160 DOMs)
        assert len(df) == 5160, f"Expected 5160 sensors, got {len(df)}"

        # Check sensor IDs are sequential from 0 to 5159
        assert df['sensor_id'].min() == 0, "Minimum sensor_id should be 0"
        assert df['sensor_id'].max() == 5159, "Maximum sensor_id should be 5159"
        assert df['sensor_id'].is_monotonic_increasing, "sensor_id should be monotonically increasing"

    def test_sensor_geometry_first_rows(self, data_root):
        """Test first 10 rows match expected values."""
        geometry_file = data_root / "sensor_geometry.csv"
        df = pd.read_csv(geometry_file)

        # First sensor (after header)
        expected_first = {
            'sensor_id': 0,
            'x': -256.14,
            'y': -521.08,
            'z': 496.03
        }
        first_row = df.iloc[0].to_dict()
        assert first_row == expected_first, f"First row mismatch: {first_row}"

        # Check first 10 sensors have same x,y coordinates (vertical string)
        first_10 = df.iloc[:10]
        assert (first_10['x'] == -256.14).all(), "First 10 sensors should have x=-256.14"
        assert (first_10['y'] == -521.08).all(), "First 10 sensors should have y=-521.08"

        # Check z decreases (going down)
        assert first_10['z'].is_monotonic_decreasing, "First 10 sensors should have decreasing z"

    def test_sensor_geometry_last_rows(self, data_root):
        """Test last rows match expected values."""
        geometry_file = data_root / "sensor_geometry.csv"
        df = pd.read_csv(geometry_file)

        # Last 4 sensors
        expected_last_4 = [
            {'sensor_id': 5156, 'x': -10.97, 'y': 6.72, 'z': -479.39},
            {'sensor_id': 5157, 'x': -10.97, 'y': 6.72, 'z': -486.4},
            {'sensor_id': 5158, 'x': -10.97, 'y': 6.72, 'z': -493.41},
            {'sensor_id': 5159, 'x': -10.97, 'y': 6.72, 'z': -500.73},
        ]

        for i, expected in enumerate(expected_last_4):
            row = df.iloc[5156 + i].to_dict()
            assert row == expected, f"Row {5156 + i} mismatch: {row}"

    def test_sensor_geometry_coordinate_ranges(self, data_root):
        """Test that coordinates are in reasonable ranges for IceCube."""
        geometry_file = data_root / "sensor_geometry.csv"
        df = pd.read_csv(geometry_file)

        # IceCube detector is roughly within [-600, 600] meters in x,y
        # and [-600, 600] meters in z
        assert df['x'].min() >= -700, "x coordinates should be >= -700m"
        assert df['x'].max() <= 700, "x coordinates should be <= 700m"
        assert df['y'].min() >= -700, "y coordinates should be >= -700m"
        assert df['y'].max() <= 700, "y coordinates should be <= 700m"
        assert df['z'].min() >= -600, "z coordinates should be >= -600m"
        assert df['z'].max() <= 600, "z coordinates should be <= 600m"


class TestMetadata:
    """Test metadata files integrity."""

    def test_train_metadata_exists(self, data_root):
        """Test that train_meta.parquet exists."""
        meta_file = data_root / "train_meta.parquet"
        assert meta_file.exists(), f"train_meta.parquet not found at {meta_file}"

    def test_train_metadata_structure(self, data_root):
        """Test train metadata has correct columns."""
        meta_file = data_root / "train_meta.parquet"
        meta = pq.read_table(meta_file)

        expected_columns = [
            'batch_id',
            'event_id',
            'first_pulse_index',
            'last_pulse_index',
            'azimuth',
            'zenith'
        ]

        assert meta.column_names == expected_columns, \
            f"Expected columns {expected_columns}, got {meta.column_names}"

    def test_train_metadata_consistency(self, data_root):
        """Test train metadata is internally consistent."""
        meta_file = data_root / "train_meta.parquet"
        meta = pq.read_table(meta_file)

        # first_pulse_index should be <= last_pulse_index
        first = meta.column('first_pulse_index').to_numpy()
        last = meta.column('last_pulse_index').to_numpy()
        assert (first <= last).all(), "first_pulse_index should be <= last_pulse_index"

        # Pulse indices should be non-negative
        assert (first >= 0).all(), "first_pulse_index should be >= 0"
        assert (last >= 0).all(), "last_pulse_index should be >= 0"

        # Azimuth should be in [0, 2π]
        azimuth = meta.column('azimuth').to_numpy()
        assert (azimuth >= 0).all(), "azimuth should be >= 0"
        assert (azimuth <= 2 * 3.14159 + 0.01).all(), "azimuth should be <= 2π"

        # Zenith should be in [0, π]
        zenith = meta.column('zenith').to_numpy()
        assert (zenith >= 0).all(), "zenith should be >= 0"
        assert (zenith <= 3.14159 + 0.01).all(), "zenith should be <= π"

    def test_test_metadata_exists(self, data_root):
        """Test that test_meta.parquet exists."""
        meta_file = data_root / "test_meta.parquet"
        assert meta_file.exists(), f"test_meta.parquet not found at {meta_file}"


class TestBatchFiles:
    """Test batch parquet files integrity."""

    def test_batch_files_exist(self, train_dir):
        """Test that batch files exist."""
        # Check first few batches exist
        for batch_id in [1, 2, 3]:
            batch_file = train_dir / f"batch_{batch_id}.parquet"
            assert batch_file.exists(), f"batch_{batch_id}.parquet not found"

    def test_batch_file_structure(self, train_dir):
        """Test batch files have correct columns."""
        batch_file = train_dir / "batch_1.parquet"
        batch = pq.read_table(batch_file)

        expected_columns = [
            'sensor_id',
            'time',
            'charge',
            'auxiliary',
            'event_id'
        ]

        assert batch.column_names == expected_columns, \
            f"Expected columns {expected_columns}, got {batch.column_names}"

    def test_batch_file_data_types(self, train_dir):
        """Test batch files have correct data types."""
        batch_file = train_dir / "batch_1.parquet"
        batch = pq.read_table(batch_file)

        # Check sensor_id is in valid range [0, 5159]
        sensor_ids = batch.column('sensor_id').to_numpy()
        assert sensor_ids.min() >= 0, "sensor_id should be >= 0"
        assert sensor_ids.max() <= 5159, "sensor_id should be <= 5159"

        # Check charge is positive
        charge = batch.column('charge').to_numpy()
        assert (charge > 0).all(), "charge should be positive"

        # Check auxiliary is boolean (0 or 1)
        auxiliary = batch.column('auxiliary').to_numpy()
        assert set(auxiliary).issubset({0, 1, True, False}), \
            "auxiliary should only contain 0/1 or True/False"

    def test_batch_count(self, data_root):
        """Test expected number of batch files."""
        meta_file = data_root / "train_meta.parquet"
        meta = pq.read_table(meta_file)

        # Get unique batch IDs from metadata
        batch_ids = meta.column('batch_id').to_numpy()
        unique_batches = set(batch_ids)

        # Should have 660 batches according to benchmark
        assert len(unique_batches) >= 650, f"Expected at least 650 batches, found {len(unique_batches)}"
