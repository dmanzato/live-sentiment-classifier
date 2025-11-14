"""Normalization utilities for live-sentiment-classifier."""
import torch


def normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample's spectrogram: (x - mean) / (std + eps)
    
    Performs per-sample normalization across the spatial dimensions (height and width)
    of each spectrogram in the batch. This matches the training pipeline and ensures
    consistent preprocessing between training and inference.
    
    Args:
        x: Input tensor of shape [B, 1, H, W] (batch, channels, mel_bins, time_frames)
           representing log-mel spectrograms.
    
    Returns:
        Normalized tensor of the same shape as input.
    
    Example:
        >>> import torch
        >>> from utils.normalization import normalize_per_sample
        >>> x = torch.randn(4, 1, 128, 250)  # batch of 4 spectrograms
        >>> x_norm = normalize_per_sample(x)
        >>> # Each sample now has mean≈0 and std≈1 across its spatial dimensions
    """
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-6)

