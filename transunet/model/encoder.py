import torch
import torch.nn as nn

from transunet.configs import ModelConfig

class ConvBlock3D(nn.Module):
    """Double 3D Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class CNNEncoder3D(nn.Module):
    """3D CNN Encoder with 4 downsampling levels"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.channels = [config.in_channels] + config.encoder_channels
        
        self.encoders = nn.ModuleList()
        for i in range(len(config.encoder_channels)):
            self.encoders.append(ConvBlock3D(self.channels[i], self.channels[i+1]))
        
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Returns:
            skip_connections: List[Tensor] - Features for skip connections
            bottleneck: Tensor - Deepest features
        """
        skip_connections = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skip_connections.append(x)
            
            if i < len(self.encoders) - 1:
                x = self.downsample(x)
        
        bottleneck = skip_connections[-1]
        
        return skip_connections, bottleneck
