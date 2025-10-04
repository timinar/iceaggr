"""
Unit tests for transformer models.

Tests:
- DOMTransformer (T1): Forward pass, gradient flow, aggregation
- Model initialization and parameter counts
"""

import pytest
import torch
import torch.nn as nn

from iceaggr.models import DOMTransformer


class TestDOMTransformer:
    """Tests for T1 (DOM-level transformer)."""

    @pytest.fixture
    def model(self):
        """Create a small T1 model for testing."""
        return DOMTransformer(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,  # No dropout for deterministic tests
        )

    @pytest.fixture
    def small_batch(self):
        """Create a small synthetic batch with DOM grouping structure."""
        # Event 0: DOM 0 (3 pulses), DOM 1 (2 pulses)
        # Event 1: DOM 0 (1 pulse), DOM 2 (4 pulses)
        pulse_features = torch.randn(10, 4)  # 10 total pulses
        pulse_to_dom_idx = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
        dom_pulse_counts = torch.tensor([3, 2, 1, 4])
        dom_to_event_idx = torch.tensor([0, 0, 1, 1])
        dom_ids = torch.tensor([42, 99, 42, 17])

        return {
            'pulse_features': pulse_features,
            'pulse_to_dom_idx': pulse_to_dom_idx,
            'dom_pulse_counts': dom_pulse_counts,
            'dom_to_event_idx': dom_to_event_idx,
            'dom_ids': dom_ids,
            'event_dom_counts': torch.tensor([2, 2]),
            'total_doms': 4,
            'batch_size': 2,
        }

    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = DOMTransformer(
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
        )

        assert model.d_model == 128
        assert model.n_heads == 8
        assert model.n_layers == 4
        assert model.d_ff == 512

        # Check model has expected components
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert len(model.layers) == 4

    def test_parameter_count(self, model):
        """Test model has reasonable parameter count."""
        total_params = sum(p.numel() for p in model.parameters())

        # d_model=64, n_heads=4, n_layers=2, d_ff=256
        # Rough estimate: input_proj (4->64) + 2 layers of (attn + ff)
        # Should be ~100k params
        assert 50_000 < total_params < 500_000

    def test_forward_pass_shape(self, model, small_batch):
        """Test forward pass produces correct output shape."""
        with torch.no_grad():
            dom_embeddings = model(small_batch)

        # Should produce one embedding per DOM
        assert dom_embeddings.shape == (4, 64)  # (total_doms, d_model)
        assert dom_embeddings.dtype == torch.float32

    def test_forward_pass_no_nans(self, model, small_batch):
        """Test forward pass doesn't produce NaNs or Infs."""
        with torch.no_grad():
            dom_embeddings = model(small_batch)

        assert not torch.isnan(dom_embeddings).any()
        assert not torch.isinf(dom_embeddings).any()

    def test_gradient_flow(self, model, small_batch):
        """Test gradients flow through all parameters."""
        model.train()

        # Forward pass
        dom_embeddings = model(small_batch)

        # Dummy loss
        loss = dom_embeddings.sum()

        # Backward pass
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"

    def test_different_batch_sizes(self, model):
        """Test model handles different batch sizes."""
        batch_configs = [
            (5, 2),    # 5 pulses, 2 DOMs
            (100, 50), # 100 pulses, 50 DOMs
            (1, 1),    # Edge case: 1 pulse, 1 DOM
        ]

        for n_pulses, n_doms in batch_configs:
            batch = {
                'pulse_features': torch.randn(n_pulses, 4),
                'pulse_to_dom_idx': torch.randint(0, n_doms, (n_pulses,)),
                'dom_pulse_counts': torch.ones(n_doms, dtype=torch.long),
                'total_doms': n_doms,
                'batch_size': 1,
            }

            with torch.no_grad():
                dom_embeddings = model(batch)

            assert dom_embeddings.shape == (n_doms, 64)

    def test_dom_isolation(self, model):
        """Test that DOMs don't leak information across each other."""
        # Create batch where DOM 0 has feature [1,0,0,0] and DOM 1 has [0,1,0,0]
        batch = {
            'pulse_features': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],  # DOM 0, pulse 0
                [1.0, 0.0, 0.0, 0.0],  # DOM 0, pulse 1
                [0.0, 1.0, 0.0, 0.0],  # DOM 1, pulse 0
            ]),
            'pulse_to_dom_idx': torch.tensor([0, 0, 1]),
            'dom_pulse_counts': torch.tensor([2, 1]),
            'total_doms': 2,
            'batch_size': 1,
        }

        with torch.no_grad():
            dom_embeddings = model(batch)

        # DOM embeddings should be different (model should distinguish inputs)
        assert not torch.allclose(dom_embeddings[0], dom_embeddings[1], atol=1e-5)

    def test_single_pulse_dom(self, model):
        """Test model handles DOMs with only 1 pulse correctly."""
        batch = {
            'pulse_features': torch.randn(1, 4),
            'pulse_to_dom_idx': torch.tensor([0]),
            'dom_pulse_counts': torch.tensor([1]),
            'total_doms': 1,
            'batch_size': 1,
        }

        with torch.no_grad():
            dom_embeddings = model(batch)

        assert dom_embeddings.shape == (1, 64)
        assert not torch.isnan(dom_embeddings).any()

    def test_aggregation_correctness(self, model):
        """Test mean pooling aggregation works correctly."""
        # Set model to eval and use simple batch
        model.eval()

        # Create batch where all pulses in DOM have same embedding after encoding
        batch = {
            'pulse_features': torch.ones(5, 4),  # Same input for all pulses
            'pulse_to_dom_idx': torch.tensor([0, 0, 0, 1, 1]),  # 2 DOMs
            'dom_pulse_counts': torch.tensor([3, 2]),
            'total_doms': 2,
            'batch_size': 1,
        }

        with torch.no_grad():
            dom_embeddings = model(batch)

        # Both DOMs should have same embedding (same inputs)
        # Allow some tolerance for numerical precision
        assert torch.allclose(dom_embeddings[0], dom_embeddings[1], atol=1e-4)

    def test_cuda_compatibility(self, model, small_batch):
        """Test model works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device('cuda')
        model = model.to(device)

        # Move batch to CUDA
        batch_cuda = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in small_batch.items()
        }

        with torch.no_grad():
            dom_embeddings = model(batch_cuda)

        assert dom_embeddings.device.type == 'cuda'
        assert dom_embeddings.shape == (4, 64)

    def test_deterministic_forward(self, small_batch):
        """Test forward pass is deterministic in eval mode."""
        torch.manual_seed(42)
        model1 = DOMTransformer(d_model=64, n_heads=4, n_layers=2, dropout=0.0)
        model1.eval()

        torch.manual_seed(42)
        model2 = DOMTransformer(d_model=64, n_heads=4, n_layers=2, dropout=0.0)
        model2.eval()

        with torch.no_grad():
            out1 = model1(small_batch)
            out2 = model2(small_batch)

        # Same initialization should give same output
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_batch_with_varying_dom_sizes(self, model):
        """Test batch where DOMs have very different numbers of pulses."""
        # DOM 0: 1 pulse, DOM 1: 50 pulses, DOM 2: 5 pulses
        batch = {
            'pulse_features': torch.randn(56, 4),
            'pulse_to_dom_idx': torch.cat([
                torch.zeros(1, dtype=torch.long),
                torch.ones(50, dtype=torch.long),
                torch.full((5,), 2, dtype=torch.long),
            ]),
            'dom_pulse_counts': torch.tensor([1, 50, 5]),
            'total_doms': 3,
            'batch_size': 1,
        }

        with torch.no_grad():
            dom_embeddings = model(batch)

        assert dom_embeddings.shape == (3, 64)
        assert not torch.isnan(dom_embeddings).any()


class TestDOMTransformerEdgeCases:
    """Edge case tests for DOMTransformer."""

    def test_empty_batch_fails_gracefully(self):
        """Test that empty batch raises appropriate error."""
        model = DOMTransformer(d_model=32, n_heads=4, n_layers=1)

        batch = {
            'pulse_features': torch.randn(0, 4),  # No pulses
            'pulse_to_dom_idx': torch.tensor([], dtype=torch.long),
            'dom_pulse_counts': torch.tensor([], dtype=torch.long),
            'total_doms': 0,
            'batch_size': 0,
        }

        # Should either handle gracefully or raise clear error
        # (Implementation dependent - for now just check it doesn't crash silently)
        try:
            with torch.no_grad():
                _ = model(batch)
        except Exception as e:
            # If it raises an error, it should be a meaningful one
            assert isinstance(e, (ValueError, RuntimeError, IndexError))

    def test_large_batch(self):
        """Test model can handle large batches (memory test)."""
        model = DOMTransformer(d_model=64, n_heads=4, n_layers=2)
        model.eval()

        # Simulate large batch: 1000 DOMs, 2000 pulses
        batch = {
            'pulse_features': torch.randn(2000, 4),
            'pulse_to_dom_idx': torch.randint(0, 1000, (2000,)),
            'dom_pulse_counts': torch.ones(1000, dtype=torch.long),
            'total_doms': 1000,
            'batch_size': 16,
        }

        with torch.no_grad():
            dom_embeddings = model(batch)

        assert dom_embeddings.shape == (1000, 64)
        assert not torch.isnan(dom_embeddings).any()
