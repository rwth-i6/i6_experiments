from dataclasses import dataclass
from typing import Sequence


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 128
    conv_channels: int = 256
    conv_kernel_sizes: tuple[int, ...] = (3,)
    conv_dilations: tuple[int, ...] | None = None
    num_conv_layers: int | None = None
    projection_dim: int = 256
    dropout: float = 0.0
    leaky_relu_negative_slope: float = 0.01
    pad_token_id: int = 0
    bos_token_id: int = 40

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "conv_kernel_sizes" in d and not isinstance(d["conv_kernel_sizes"], tuple):
            d["conv_kernel_sizes"] = tuple(d["conv_kernel_sizes"])
        if d.get("conv_dilations") is not None and not isinstance(d["conv_dilations"], tuple):
            d["conv_dilations"] = tuple(d["conv_dilations"])
        return cls(**d)

    def __post_init__(self):
        if isinstance(self.conv_kernel_sizes, Sequence) and not isinstance(self.conv_kernel_sizes, tuple):
            self.conv_kernel_sizes = tuple(self.conv_kernel_sizes)
        if self.conv_dilations is not None and isinstance(self.conv_dilations, Sequence) and not isinstance(
            self.conv_dilations, tuple
        ):
            self.conv_dilations = tuple(self.conv_dilations)
