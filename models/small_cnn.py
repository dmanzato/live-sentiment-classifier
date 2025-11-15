"""Small CNN model for audio classification."""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    A lightweight CNN for spectrogram-based audio classification.
    
    Architecture:
        - 3 convolutional layers with BatchNorm, ReLU, and MaxPool
        - Dropout for regularization
        - Adaptive average pooling
        - Linear classifier
    
    Input shape: [B, 1, n_mels, time_frames]
    Output shape: [B, n_classes]
    
    Args:
        n_classes: Number of output classes.
        dropout: Dropout probability (default: 0.3).
        use_bn: Whether to use batch normalization (default: True).
    
    Example:
        >>> model = SmallCNN(n_classes=3)
        >>> x = torch.randn(4, 1, 64, 250)  # [B, C, H, W]
        >>> logits = model(x)  # [4, 3]
    """
    def __init__(self, n_classes: int = 3, dropout: float = 0.0, use_bn: bool = False):
        super().__init__()
        
        # Build features - match original structure when dropout=0.0 and use_bn=False
        if not use_bn and dropout <= 0.0:
            # Original structure - exact match
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # -> [B,128,1,1]
            )
            self.classifier = nn.Linear(128, n_classes)
        else:
            # Enhanced structure with batch norm and/or dropout
            layers = []
            # First conv block
            layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(32))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            
            # Second conv block
            layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            
            # Third conv block
            layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(128))
            layers.append(nn.ReLU())
            
            # Adaptive pooling
            layers.append(nn.AdaptiveAvgPool2d(1))  # -> [B,128,1,1]
            
            self.features = nn.Sequential(*layers)
            
            # Classifier - only add dropout if dropout > 0
            if dropout > 0:
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(128, n_classes)
                )
            else:
                self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [B, 1, n_mels, time_frames].
        
        Returns:
            Class logits tensor of shape [B, n_classes].
        """
        x = self.features(x)              # [B,128,1,1]
        x = x.squeeze(-1).squeeze(-1)     # [B,128]
        return self.classifier(x)         # [B,n_classes]

