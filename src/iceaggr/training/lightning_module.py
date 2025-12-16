"""
PyTorch Lightning module for training the hierarchical model.

This module wraps HierarchicalDOMModel for training with PyTorch Lightning,
providing training/validation steps, optimizer configuration, and logging.
"""

import torch
import pytorch_lightning as pl
from typing import Any, Dict, Optional

from iceaggr.models import HierarchicalDOMModel, angular_distance_loss


class HierarchicalModelModule(pl.LightningModule):
    """
    PyTorch Lightning module for the hierarchical DOM model.

    Handles training loop, validation, optimizer setup, and logging.

    Args:
        config: Configuration dictionary containing:
            Model config:
                - embed_dim, max_doms, pulse_hidden_dims, etc.
            Training config:
                - lr: Learning rate (default: 1e-4)
                - max_lr: Max LR for OneCycleLR (default: 1e-3)
                - weight_decay: AdamW weight decay (default: 0.01)
                - warmup_steps: LR warmup steps (default: 1000)

    Example:
        >>> config = {"embed_dim": 64, "max_doms": 128, "lr": 1e-4}
        >>> module = HierarchicalModelModule(config)
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(module, train_dataloader, val_dataloader)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}

        self.save_hyperparameters(config)
        self.config = config

        # Create model
        self.model = HierarchicalDOMModel(config)

        # Training hyperparameters (ensure float conversion for YAML values)
        self.lr = float(config.get('lr', 1e-4))
        self.max_lr = float(config.get('max_lr', 1e-3))
        self.weight_decay = float(config.get('weight_decay', 0.01))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(batch)

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        # Forward pass
        y_pred = self(batch)

        # Compute loss
        loss = angular_distance_loss(y_pred, batch['targets'])

        # Log metrics
        self.log(f'{stage}/loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/angular_error_deg', torch.rad2deg(loss), sync_dist=True)

        return loss

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        loss = self._shared_step(batch, 'train')

        # Log learning rate
        if self.trainer.global_step % 100 == 0:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/lr', lr)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, 'val')

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        pass

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level validation metrics."""
        pass
