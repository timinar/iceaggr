"""
Integration tests for the full E2E training pipeline.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from iceaggr.data import collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss, E2ETrainer, TrainingConfig


def create_synthetic_event(
    event_id: int, num_doms: int = 10, pulses_per_dom: int = 5
):
    """
    Create a synthetic IceCube event for testing.

    Returns:
        Dictionary matching IceCubeDataset output format
    """
    total_pulses = num_doms * pulses_per_dom

    # Create pulse features (time, charge, auxiliary, dom_index)
    pulses = torch.rand(total_pulses, 3)  # time, charge, auxiliary
    dom_indices = torch.repeat_interleave(
        torch.arange(num_doms), pulses_per_dom
    )

    # Create random target (azimuth, zenith)
    target = torch.tensor([
        torch.rand(1).item() * 2 * 3.14159,  # azimuth [0, 2π]
        torch.rand(1).item() * 3.14159,      # zenith [0, π]
    ])

    return {
        "event_id": f"event_{event_id}",
        "pulses": pulses,
        "dom_indices": dom_indices,
        "target": target,
    }


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_events: int = 10):
        self.events = [
            create_synthetic_event(i, num_doms=10, pulses_per_dom=5)
            for i in range(num_events)
        ]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx]


class TestE2ETrainingPipeline:
    """Integration tests for full training pipeline."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create synthetic dataset."""
        return SyntheticDataset(num_events=20)

    @pytest.fixture
    def train_loader(self, synthetic_dataset):
        """Create training dataloader."""
        return DataLoader(
            synthetic_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=128),
        )

    @pytest.fixture
    def model(self, tmp_path):
        """Create hierarchical transformer model."""
        # Create dummy geometry file
        geometry_path = tmp_path / "sensor_geometry.csv"
        with open(geometry_path, "w") as f:
            f.write("sensor_id,x,y,z\n")
            for i in range(100):  # Small geometry for testing
                f.write(f"{i},{i*10},{i*5},{i*2}\n")

        return HierarchicalTransformer(
            d_model=64,
            t1_n_heads=4,
            t1_n_layers=2,
            t1_max_seq_len=128,
            t1_max_batch_size=32,
            t2_n_heads=4,
            t2_n_layers=2,
            t2_max_doms=100,
            dropout=0.1,
            sensor_geometry_path=str(geometry_path),
        )

    @pytest.fixture
    def config(self, tmp_path):
        """Create training config."""
        return TrainingConfig(
            num_epochs=3,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_every_n_steps=1,
            device="cpu",
        )

    def test_single_forward_pass(self, model, train_loader):
        """Test single forward pass through model."""
        batch = next(iter(train_loader))
        predictions = model(batch)

        # Check output shape
        batch_size = len(batch["metadata"]["event_ids"])
        assert predictions.shape == (batch_size, 2)

        # Check predictions are finite
        assert torch.all(torch.isfinite(predictions))

    def test_loss_computation(self, model, train_loader):
        """Test loss computation on batch."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)

        batch = next(iter(train_loader))
        predictions = model(batch)
        targets = batch["metadata"]["targets"]

        loss = loss_fn(predictions, targets)

        # Check loss is finite and positive
        assert torch.isfinite(loss)
        assert loss >= 0.0

    def test_backward_pass(self, model, train_loader):
        """Test backward pass and gradient flow."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = next(iter(train_loader))

        # Forward pass
        predictions = model(batch)
        targets = batch["metadata"]["targets"]
        loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.all(
                    torch.isfinite(param.grad)
                ), f"Non-finite gradient for {name}"

    def test_parameter_updates(self, model, train_loader):
        """Test that parameters update during training."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Get initial parameters
        initial_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Training step
        batch = next(iter(train_loader))
        optimizer.zero_grad()
        predictions = model(batch)
        targets = batch["metadata"]["targets"]
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        # Check parameters changed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(
                    param, initial_params[name], atol=1e-6
                ), f"Parameter {name} did not update"

    def test_full_training_epoch(self, model, train_loader, config):
        """Test training for full epoch."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        trainer = E2ETrainer(model, loss_fn, optimizer, config)
        metrics = trainer.train_epoch(train_loader)

        assert "loss" in metrics
        assert "num_batches" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_overfitting_single_batch(self, model, train_loader):
        """Test that model can overfit a single batch."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Get single batch
        batch = next(iter(train_loader))
        targets = batch["metadata"]["targets"]

        # Train for many iterations
        initial_loss = None
        final_loss = None

        for i in range(100):
            optimizer.zero_grad()
            predictions = model(batch)
            loss = loss_fn(predictions, targets)

            if i == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if i == 99:
                final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, (
            f"Model did not overfit: initial loss {initial_loss:.4f}, "
            f"final loss {final_loss:.4f}"
        )

    def test_validation_loop(self, model, train_loader, config):
        """Test validation loop."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        trainer = E2ETrainer(model, loss_fn, optimizer, config)
        metrics = trainer.validate(train_loader)

        assert "val_loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["val_loss"]))

    def test_complete_training_run(self, model, train_loader, config, tmp_path):
        """Test complete training run with checkpointing."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        # Split data into train/val
        dataset = train_loader.dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=4,
            collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=128),
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=4,
            collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=128),
        )

        # Train
        history = trainer.fit(train_dl, val_dl)

        # Check history
        assert len(history["train_loss"]) == config.num_epochs
        assert len(history["val_loss"]) > 0

        # Check checkpoint was saved
        checkpoint_path = Path(config.checkpoint_dir) / "checkpoint_epoch_0.pt"
        assert checkpoint_path.exists()

        # Load checkpoint and verify
        checkpoint = torch.load(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 0
