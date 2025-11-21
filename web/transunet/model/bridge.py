import torch
import torch.nn as nn

from transunet.configs import ModelConfig
from transunet.model.transformer import Transformer

class Tokenizer3D(nn.Module):
    """Convert spatial features to sequence tokens with downsampling"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_channels[-1]
        self.embed_dim = config.hidden_size
        
        # Downsample spatially to reduce sequence length
        self.downsample = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels, 
                     kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, self.in_channels),
            nn.GELU()
        )
        
        self.projection = nn.Linear(self.in_channels, self.embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            tokens: [B, seq_len, embed_dim]
            spatial_shape: (D_down, H_down, W_down) - downsampled shape
        """
        B, C, D, H, W = x.shape
        
        # Downsample: reduces spatial dimensions by 2x
        x = self.downsample(x)  # [B, C, D/2, H/2, W/2]
        _, _, D_down, H_down, W_down = x.shape
        
        # Flatten spatial: [B, C, D_down, H_down, W_down] -> [B, C, D_down*H_down*W_down]
        x = x.flatten(2)
        
        # Transpose: [B, C, seq_len] -> [B, seq_len, C]
        x = x.transpose(1, 2)
        
        # Project: [B, seq_len, C] -> [B, seq_len, embed_dim]
        tokens = self.projection(x)
        
        spatial_shape = (D_down, H_down, W_down)
        
        return tokens, spatial_shape


class DeTokenizer3D(nn.Module):
    """Convert sequence tokens back to spatial features with upsampling"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.out_channels = config.encoder_channels[-1]
        
        self.projection = nn.Linear(self.embed_dim, self.out_channels)
        
        # Upsample to restore original spatial size
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(self.out_channels, self.out_channels,
                             kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, self.out_channels),
            nn.GELU()
        )
    
    def forward(self, tokens, spatial_shape):
        """
        Args:
            tokens: [B, seq_len, embed_dim]
            spatial_shape: (D_down, H_down, W_down) - downsampled shape
        Returns:
            x: [B, C, D, H, W] - original spatial size restored
        """
        B, seq_len, embed_dim = tokens.shape
        D_down, H_down, W_down = spatial_shape
        
        # Project: [B, seq_len, embed_dim] -> [B, seq_len, C]
        x = self.projection(tokens)
        
        # Transpose: [B, seq_len, C] -> [B, C, seq_len]
        x = x.transpose(1, 2)
        
        # Reshape to 3D: [B, C, seq_len] -> [B, C, D_down, H_down, W_down]
        x = x.reshape(B, self.out_channels, D_down, H_down, W_down)
        
        # Upsample back to original size: [B, C, D, H, W]
        x = self.upsample(x)
        
        return x


class TransformerBridge(nn.Module):
    """Complete bridge: Tokenization -> Transformer -> De-tokenization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tokenizer = Tokenizer3D(config)
        self.transformer = Transformer(config)
        self.detokenizer = DeTokenizer3D(config)
    
    def forward(self, bottleneck):
        tokens, spatial_shape = self.tokenizer(bottleneck)
        tokens = self.transformer(tokens)
        output = self.detokenizer(tokens, spatial_shape)
        return output