import torch
import torch.nn as nn

from transunet.configs import ModelConfig

# Rest of the code stays the same

class SegmentationHead3D(nn.Module):
    """Final 1x1x1 convolution to produce class predictions"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.conv = nn.Conv3d(config.encoder_channels[0], config.num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            logits: [B, num_classes, D, H, W]
        """
        return self.conv(x)