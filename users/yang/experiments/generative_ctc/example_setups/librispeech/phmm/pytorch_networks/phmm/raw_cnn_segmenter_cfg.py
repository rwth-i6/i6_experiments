from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    conv_kernel_sizes: Tuple[int, ...] = (10, 8, 4, 4, 4)
    conv_strides: Tuple[int, ...] = (5, 4, 2, 2, 2)
    conv_channels: int = 256
    conv_bias: bool = False
    batch_norm: bool = True
    apply_final_norm_activation: bool = True
    leaky_relu_negative_slope: float = 0.01
    projection_dim: int = 64
    projection_bias: bool = True
    contrastive_num_samples: int = 1
    contrastive_temperature: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
