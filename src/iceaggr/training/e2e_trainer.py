"""
End-to-end training loop for hierarchical transformer (T1 + T2).

Much simpler than T1-only trainer since the HierarchicalTransformer
handles the T1→T2 pipeline internally.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iceaggr.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    # Training hyperparameters
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = 1.0

    # Gradient accumulation
    accumulation_steps: int = 1  # Number of mini-batches to accumulate

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1

    # Logging
    log_every_n_steps: int = 10
    validate_every_n_epochs: int = 1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Wandb (optional)
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None


class E2ETrainer:
    """
    End-to-end trainer for HierarchicalTransformer (T1 + T2).

    Simpler than T1-only trainer since model handles full pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb (optional)
        self.wandb_run = None
        if config.use_wandb:
            import wandb

            self.wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                tags=config.wandb_tags,
                config=vars(config),
            )

        logger.info(f"Initialized E2E trainer on device: {self.device}")
        logger.info(
            f"Model has {sum(p.numel() for p in model.parameters()):,} parameters"
        )

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with collate_dom_packing

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        start_time = time.time()
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move batch to device (batch is already collated with dom packing)
            batch = self._batch_to_device(batch)

            # Forward pass: T1 → T2 → predictions
            predictions = self.model(batch)  # (batch_size, 2)

            # Get targets from metadata
            targets = batch["metadata"]["targets"]  # (batch_size, 2)

            # Compute loss
            loss = self.loss_fn(predictions, targets)

            # Scale loss by accumulation steps
            loss = loss / self.config.accumulation_steps

            # Backward pass
            loss.backward()

            # Optimizer step every accumulation_steps
            if (step + 1) % self.config.accumulation_steps == 0 or (step + 1) == len(
                train_loader
            ):
                # Gradient clipping
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    batch_size = len(batch["metadata"]["event_ids"])
                    logger.info(
                        f"Epoch {self.current_epoch} | Step {self.global_step} | "
                        f"Loss: {loss.item() * self.config.accumulation_steps:.4f} | "
                        f"Batch size: {batch_size}"
                    )

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "train/loss": loss.item()
                                * self.config.accumulation_steps,
                                "train/step": self.global_step,
                                "train/epoch": self.current_epoch,
                            }
                        )

            epoch_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1

        # Epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        metrics = {
            "loss": avg_loss,
            "num_batches": num_batches,
            "elapsed_sec": elapsed,
        }

        logger.info(
            f"Epoch {self.current_epoch} completed | "
            f"Avg Loss: {avg_loss:.4f} | Batches: {num_batches} | "
            f"Time: {elapsed:.1f}s"
        )

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Validation loop.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = self._batch_to_device(batch)

            # Forward pass
            predictions = self.model(batch)
            targets = batch["metadata"]["targets"]

            # Compute loss
            loss = self.loss_fn(predictions, targets)

            total_loss += loss.item()
            num_batches += 1

        metrics = {
            "val_loss": total_loss / max(num_batches, 1),
            "num_batches": num_batches,
        }

        logger.info(f"Validation | Loss: {metrics['val_loss']:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "val/loss": metrics["val_loss"],
                    "val/epoch": self.current_epoch,
                }
            )

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader

        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])

            # Validate
            if (
                val_loader is not None
                and epoch % self.config.validate_every_n_epochs == 0
            ):
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])

            # Checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

        logger.info("Training completed!")
        return history

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": vars(self.config),
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        device_batch = {}

        for key, value in batch.items():
            if key == "metadata":
                # Metadata contains nested structure
                device_batch[key] = {}
                for meta_key, meta_value in value.items():
                    if isinstance(meta_value, torch.Tensor):
                        device_batch[key][meta_key] = meta_value.to(self.device)
                    else:
                        device_batch[key][meta_key] = meta_value
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value

        return device_batch
