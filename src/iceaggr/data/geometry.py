"""
Geometry loader for IceCube sensor positions.

This module provides efficient lookup of DOM (Digital Optical Module) positions
by sensor_id. The geometry is loaded once and cached as a tensor for fast indexing.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union


class GeometryLoader:
    """
    Load and cache sensor geometry for fast position lookup.

    The geometry file should be a CSV with columns: sensor_id, x, y, z
    where sensor_id is the index (0 to N-1) and x, y, z are coordinates.

    Args:
        geometry_path: Path to geometry CSV file
        device: Device to store the geometry tensor on (default: cpu)

    Example:
        >>> geometry = GeometryLoader("/path/to/sensor_geometry_normalized.csv")
        >>> sensor_ids = torch.tensor([0, 1, 5, 100])
        >>> positions = geometry[sensor_ids]  # (4, 3) tensor of x,y,z
    """

    def __init__(
        self,
        geometry_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        self.geometry_path = Path(geometry_path)
        self.device = device or torch.device("cpu")

        # Load geometry
        self._positions = self._load_geometry()
        self.n_sensors = self._positions.shape[0]

    def _load_geometry(self) -> torch.Tensor:
        """Load geometry from CSV file."""
        # Use numpy for fast CSV loading
        data = np.loadtxt(
            self.geometry_path,
            delimiter=",",
            skiprows=1,  # Skip header
            dtype=np.float32,
        )

        # Extract columns: sensor_id, x, y, z
        sensor_ids = data[:, 0].astype(np.int64)
        positions = data[:, 1:4]  # x, y, z

        # Verify sensor_ids are contiguous from 0
        max_id = sensor_ids.max()
        n_sensors = max_id + 1

        # Create position tensor indexed by sensor_id
        pos_tensor = torch.zeros(n_sensors, 3, dtype=torch.float32, device=self.device)
        pos_tensor[sensor_ids] = torch.from_numpy(positions).to(self.device)

        return pos_tensor

    def __getitem__(self, sensor_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up positions for given sensor IDs.

        Args:
            sensor_ids: Tensor of sensor IDs (any shape)

        Returns:
            Positions tensor with shape (*sensor_ids.shape, 3)
        """
        # Handle device mismatch
        if sensor_ids.device != self.device:
            sensor_ids = sensor_ids.to(self.device)

        return self._positions[sensor_ids.long()]

    def to(self, device: torch.device) -> "GeometryLoader":
        """Move geometry to specified device."""
        self.device = device
        self._positions = self._positions.to(device)
        return self

    @property
    def positions(self) -> torch.Tensor:
        """Return the full positions tensor (n_sensors, 3)."""
        return self._positions

    def __repr__(self) -> str:
        return f"GeometryLoader(n_sensors={self.n_sensors}, device={self.device})"
