import torch
import torch.nn as nn

from transunet.model.encoder import ConvBlock3D  # Relative import
from transunet.configs import ModelConfig


class UpConvBlock3D(nn.Module):
    """Upsampling block with skip connection"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )
        
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        """
        Args:
            x: [B, in_channels, H, W, D]
            skip: [B, skip_channels, H*2, W*2, D*2]
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class CNNDecoder3D(nn.Module):
    """3D CNN Decoder with skip connections"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Decoder channels (reverse of encoder, excluding bottleneck)
        encoder_channels = config.encoder_channels
        
        # Build decoder blocks
        self.decoders = nn.ModuleList()
        
        # 512 -> 256 (skip: 256)
        self.decoders.append(
            UpConvBlock3D(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        )
        
        # 256 -> 128 (skip: 128)
        self.decoders.append(
            UpConvBlock3D(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        )
        
        # 128 -> 64 (skip: 64)
        self.decoders.append(
            UpConvBlock3D(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        )
    
    def forward(self, bottleneck, skip_connections):
        """
        Args:
            bottleneck: [B, 512, H/8, W/8, D/8]
            skip_connections: List of 4 feature maps
        Returns:
            x: [B, 64, H, W, D]
        """
        x = bottleneck
        
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skip_connections) - 2 - i  # 2, 1, 0
            skip = skip_connections[skip_idx]
            x = decoder(x, skip)
        
        return x