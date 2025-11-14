"""Tests for utility functions."""
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.normalization import normalize_per_sample


class TestNormalization:
    """Test normalization utilities."""
    
    def test_normalize_per_sample_shape(self):
        """Test that normalize_per_sample preserves input shape."""
        batch_size = 4
        n_mels = 128
        time_frames = 250
        
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        x_norm = normalize_per_sample(x)
        
        assert x_norm.shape == x.shape
    
    def test_normalize_per_sample_statistics(self):
        """Test that normalize_per_sample produces mean≈0 and std≈1 per sample."""
        batch_size = 2
        n_mels = 64
        time_frames = 100
        
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        x_norm = normalize_per_sample(x)
        
        # Check each sample in the batch
        for i in range(batch_size):
            sample = x_norm[i, 0, :, :]
            mean = sample.mean().item()
            std = sample.std().item()
            
            # Mean should be close to 0
            assert abs(mean) < 1e-5, f"Sample {i} mean should be ≈0, got {mean}"
            # Std should be close to 1 (relaxed tolerance for floating-point precision)
            assert abs(std - 1.0) < 1e-4, f"Sample {i} std should be ≈1, got {std}"
    
    def test_normalize_per_sample_different_inputs(self):
        """Test that different inputs produce different normalized outputs."""
        x1 = torch.randn(2, 1, 64, 100)
        x2 = torch.randn(2, 1, 64, 100) * 2.0 + 5.0  # Different scale and offset
        
        x1_norm = normalize_per_sample(x1)
        x2_norm = normalize_per_sample(x2)
        
        # After normalization, both should have similar statistics
        assert abs(x1_norm.mean().item()) < 1e-5
        assert abs(x2_norm.mean().item()) < 1e-5
        # Relaxed tolerance for floating-point precision (normalization uses std + 1e-6)
        assert abs(x1_norm.std().item() - 1.0) < 1e-4, f"x1_norm std should be ≈1, got {x1_norm.std().item()}"
        assert abs(x2_norm.std().item() - 1.0) < 1e-4, f"x2_norm std should be ≈1, got {x2_norm.std().item()}"
    
    def test_normalize_per_sample_constant_input(self):
        """Test that constant input is handled correctly."""
        # Constant input (all zeros or all same value)
        x = torch.ones(2, 1, 64, 100)
        x_norm = normalize_per_sample(x)
        
        # Should handle division by zero gracefully (std + 1e-6 prevents division by zero)
        assert torch.isfinite(x_norm).all()


class TestTensorUtils:
    """Test tensor utility functions."""
    
    def test_tensor_shapes(self):
        """Test that we can create tensors with expected shapes."""
        # Test spectrogram shape
        batch_size = 2
        n_mels = 64
        time_frames = 250
        
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        assert x.shape == (batch_size, 1, n_mels, time_frames)
    
    def test_device_placement(self):
        """Test tensor device placement."""
        x = torch.randn(2, 3)
        assert x.device.type == 'cpu'
        
        if torch.cuda.is_available():
            x_cuda = x.to('cuda')
            assert x_cuda.device.type == 'cuda'
        
        if torch.backends.mps.is_available():
            x_mps = x.to('mps')
            assert x_mps.device.type == 'mps'
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        x = torch.randn(2, 3, requires_grad=True)
        y = x.sum()
        y.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

