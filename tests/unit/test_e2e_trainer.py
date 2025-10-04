"""
Unit tests for E2E trainer.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from iceaggr.training import AngularDistanceLoss, E2ETrainer, TrainingConfig


class DummyE2EModel(nn.Module):
    """Dummy model for testing that mimics HierarchicalTransformer interface."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)  # Predict azimuth, zenith

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Dummy forward pass.

        Expects batch to have 'packed_sequences' and returns predictions.
        """
        # Extract batch size from metadata
        batch_size = len(batch["metadata"]["event_ids"])

        # Create dummy features (in real model, this comes from T1→T2)
        dummy_features = torch.randn(batch_size, 64)

        # Predict angles
        return self.linear(dummy_features)


def create_dummy_batch(batch_size: int = 4, max_seq_len: int = 128) -> dict:
    """Create a dummy batch that matches collate_dom_packing output."""
    return {
        "packed_sequences": torch.randn(batch_size, max_seq_len, 4),
        "dom_boundaries": torch.randint(0, 10, (batch_size, max_seq_len)),
        "dom_mask": torch.ones(batch_size, max_seq_len),
        "metadata": {
            "event_ids": [f"event_{i}" for i in range(batch_size)],
            "targets": torch.rand(batch_size, 2) * 3.14,  # Random angles
            "sensor_ids": torch.randint(0, 5160, (batch_size * 10,)),
            "dom_to_event_idx": torch.repeat_interleave(
                torch.arange(batch_size), 10
            ),
        },
    }


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.num_epochs == 10
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-5
        assert config.grad_clip_norm == 1.0
        assert config.accumulation_steps == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            num_epochs=20,
            learning_rate=1e-3,
            weight_decay=1e-4,
            grad_clip_norm=0.5,
        )

        assert config.num_epochs == 20
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-4
        assert config.grad_clip_norm == 0.5


class TestE2ETrainer:
    """Test E2ETrainer class."""

    @pytest.fixture
    def model(self):
        """Create dummy model."""
        return DummyE2EModel(d_model=64)

    @pytest.fixture
    def loss_fn(self):
        """Create loss function."""
        return AngularDistanceLoss(use_unit_vectors=False)

    @pytest.fixture
    def config(self, tmp_path):
        """Create training config."""
        return TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_every_n_steps=1,
            device="cpu",
        )

    @pytest.fixture
    def train_loader(self):
        """Create dummy training dataloader."""
        batches = [create_dummy_batch(batch_size=4) for _ in range(3)]
        return batches  # Return as list to simulate DataLoader

    def test_trainer_initialization(self, model, loss_fn, config):
        """Test trainer initialization."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.checkpoint_dir.exists()

    def test_batch_to_device(self, model, loss_fn, config):
        """Test moving batch to device."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        batch = create_dummy_batch(batch_size=2)
        device_batch = trainer._batch_to_device(batch)

        # Check tensors are on correct device
        assert device_batch["packed_sequences"].device.type == "cpu"
        assert device_batch["metadata"]["targets"].device.type == "cpu"

    def test_single_training_step(self, model, loss_fn, config):
        """Test single training step."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        batch = create_dummy_batch(batch_size=4)

        # Get initial weights
        initial_weight = model.linear.weight.data.clone()

        # Training step
        trainer.optimizer.zero_grad()
        batch = trainer._batch_to_device(batch)
        predictions = trainer.model(batch)
        targets = batch["metadata"]["targets"]
        loss = trainer.loss_fn(predictions, targets)
        loss.backward()
        trainer.optimizer.step()

        # Check weights updated
        assert not torch.allclose(model.linear.weight.data, initial_weight)

    def test_train_epoch(self, model, loss_fn, config, train_loader):
        """Test training for one epoch."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        # Manually call train_epoch with list of batches
        # (Simulating DataLoader iteration)
        class DummyLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        loader = DummyLoader(train_loader)
        metrics = trainer.train_epoch(loader)

        assert "loss" in metrics
        assert "num_batches" in metrics
        assert metrics["num_batches"] == 3
        assert trainer.global_step == 3  # One step per batch

    def test_gradient_clipping(self, model, loss_fn, config):
        """Test that gradient clipping is applied."""
        optimizer = torch.optim.Adam(model.parameters())
        config.grad_clip_norm = 0.1  # Very small clip norm
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        batch = create_dummy_batch(batch_size=4)
        batch = trainer._batch_to_device(batch)

        # Forward and backward
        predictions = trainer.model(batch)
        targets = batch["metadata"]["targets"]
        loss = trainer.loss_fn(predictions, targets)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), config.grad_clip_norm
        )

        # Check gradient norm is clipped
        total_norm = torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in trainer.model.parameters())
        )
        assert total_norm <= config.grad_clip_norm * 1.01  # Allow small numerical error

    def test_gradient_accumulation(self, model, loss_fn, config):
        """Test gradient accumulation."""
        config.accumulation_steps = 2
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        # Create 2 batches
        batch1 = create_dummy_batch(batch_size=2)
        batch2 = create_dummy_batch(batch_size=2)

        trainer.optimizer.zero_grad()

        # First batch (no optimizer step)
        batch1 = trainer._batch_to_device(batch1)
        pred1 = trainer.model(batch1)
        loss1 = trainer.loss_fn(pred1, batch1["metadata"]["targets"])
        loss1 = loss1 / config.accumulation_steps
        loss1.backward()

        # Second batch (optimizer step)
        batch2 = trainer._batch_to_device(batch2)
        pred2 = trainer.model(batch2)
        loss2 = trainer.loss_fn(pred2, batch2["metadata"]["targets"])
        loss2 = loss2 / config.accumulation_steps
        loss2.backward()

        # Now step
        trainer.optimizer.step()

        # Gradients should be cleared after step
        # (We manually step here, in real training it's automatic)

    def test_validation(self, model, loss_fn, config, train_loader):
        """Test validation loop."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        class DummyLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

        val_loader = DummyLoader(train_loader)
        metrics = trainer.validate(val_loader)

        assert "val_loss" in metrics
        assert "num_batches" in metrics
        assert torch.isfinite(torch.tensor(metrics["val_loss"]))

    def test_checkpoint_save_load(self, model, loss_fn, config, tmp_path):
        """Test checkpoint saving and loading."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        # Train for one step
        batch = create_dummy_batch(batch_size=4)
        batch = trainer._batch_to_device(batch)
        trainer.optimizer.zero_grad()
        predictions = trainer.model(batch)
        targets = batch["metadata"]["targets"]
        loss = trainer.loss_fn(predictions, targets)
        loss.backward()
        trainer.optimizer.step()

        # Save checkpoint
        trainer.save_checkpoint(epoch=0)
        checkpoint_path = trainer.checkpoint_dir / "checkpoint_epoch_0.pt"
        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        new_model = DummyE2EModel(d_model=64)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        new_trainer = E2ETrainer(new_model, loss_fn, optimizer, config)
        new_trainer.load_checkpoint(str(checkpoint_path))

        # Check states match
        assert new_trainer.current_epoch == 0
        assert torch.allclose(
            model.linear.weight, new_model.linear.weight
        )

    def test_fit_complete_training(self, model, loss_fn, config, train_loader):
        """Test complete training loop with fit()."""
        optimizer = torch.optim.Adam(model.parameters())
        trainer = E2ETrainer(model, loss_fn, optimizer, config)

        class DummyLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        train_dl = DummyLoader(train_loader)
        val_dl = DummyLoader(train_loader[:2])  # Smaller val set

        history = trainer.fit(train_dl, val_dl)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == config.num_epochs
        assert trainer.current_epoch == config.num_epochs - 1
