from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Tuple, List, Optional


class DataConfig(BaseModel):
    """
    Minimal inference config.
    Keeps only what is required to run preprocessing and subject → tensor.
    """

    # used only when you want local inference on folders
    data_path: Path = Field(default=Path("data/uploads"))

    # shape your model expects AFTER preprocessing / CropOrPad
    patch_size: Optional[Tuple[int, int, int]] = (224, 224, 160)

    # normalization
    normalize_method: str = "z-score"

    # TorchIO loader settings (these MUST stay)
    num_workers: int = 0
    pin_memory: bool = False




class ModelConfig(BaseModel):
    """
    Minimal TransUNet config for inference.
    All fields required to rebuild the model architecture exactly are kept.
    """

    # basic model IO
    in_channels: int = 4
    num_classes: int = 4

    # model expects this input volume size
    img_size: Tuple[int, int, int] = (160, 224, 224)  # (D, H, W)

    # encoder sizes
    encoder_channels: List[int] = [16, 32, 64, 128]

    # transformer bottleneck
    hidden_size: int = 384
    num_transformer_layers: int = 6
    num_heads: int = 6
    mlp_dim: int = 1536
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # patch embedding
    patch_size: int = 16

    model_config = {"extra": "ignore"}

    @field_validator("img_size")
    @classmethod
    def _validate_img_size(cls, v):
        if len(v) != 3:
            raise ValueError("img_size must be (D, H, W)")
        if any(int(x) <= 0 for x in v):
            raise ValueError("All dimensions must be positive")
        return tuple(int(x) for x in v)

    @model_validator(mode="after")
    def _validate_hidden(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        return self
