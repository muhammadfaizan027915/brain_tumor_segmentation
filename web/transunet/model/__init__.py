# model/__init__.py
"""
TransUNet3D Model Package

Usage in Kaggle:
    from model import TransUNet3D, TransUNetConfig
"""

# Import in correct dependency order to avoid circular imports
from transunet.model.encoder import ConvBlock3D, CNNEncoder3D
from transunet.model.decoder import CNNDecoder3D
from transunet.model.bridge import Tokenizer3D, DeTokenizer3D, TransformerBridge
from transunet.model.transformer import Transformer
from transunet.model.segmentation_head import SegmentationHead3D
from transunet.model.model import TransUNet3D

# Export public API
__all__ = [
    'TransUNet3D',
    'Transformer',
    'CNNEncoder3D',
    'CNNDecoder3D', 
    'TransformerBridge',
    'SegmentationHead3D',
    'ConvBlock3D',
    'Tokenizer3D',
    'DeTokenizer3D'
]