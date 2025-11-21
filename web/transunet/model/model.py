import torch
import torch.nn as nn


from transunet.configs import ModelConfig
# Use relative imports within package
from transunet.model.encoder import CNNEncoder3D
from transunet.model.decoder import CNNDecoder3D
from transunet.model.bridge import TransformerBridge
from transunet.model.segmentation_head import SegmentationHead3D



class TransUNet3D(nn.Module):
    """
    Complete 3D TransUNet Architecture
    
    Architecture:
        Input [B, 4, 128, 128, 128]
          ↓
        Encoder (4 levels with skip connections)
          ↓
        Bottleneck [B, 512, 16, 16, 16]
          ↓
        Bridge (Tokenization -> Transformer -> De-tokenization)
          ↓
        Decoder (3 upsampling levels with skip connections)
          ↓
        Segmentation Head
          ↓
        Output [B, 4, 128, 128, 128]
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = CNNEncoder3D(config)
        
        # Transformer Bridge
        self.bridge = TransformerBridge(config)
        
        # Decoder
        self.decoder = CNNDecoder3D(config)
        
        # Segmentation Head
        self.seg_head = SegmentationHead3D(config)
        
        print(f"\nTransUNet3D initialized successfully!")
        self._print_architecture()
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, D, H, W] - Input MRI volumes
        
        Returns:
            logits: [B, num_classes, D, H, W] - Segmentation logits
        """
        # Encoder
        skip_connections, bottleneck = self.encoder(x)
        
        # Bridge (Transformer)
        bottleneck = self.bridge(bottleneck)
        
        # Decoder
        x = self.decoder(bottleneck, skip_connections)
        
        # Segmentation head
        logits = self.seg_head(x)
        
        return logits
    
    def _print_architecture(self):
        """Print model architecture summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (fp32)")