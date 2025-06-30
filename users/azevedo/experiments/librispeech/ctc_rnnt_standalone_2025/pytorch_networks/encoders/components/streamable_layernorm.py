import torch
from torch import nn

from ...streamable_module import StreamableModule

class StreamableLayerNormV1(StreamableModule):
    def __init__(self, input_dim: torch.Tensor, dual_mode: bool = True):
        super().__init__()
        self.layernorm_off = nn.LayerNorm(input_dim)
        self.layernorm_on = nn.LayerNorm(input_dim) if dual_mode else self.layernorm_off

    def forward_offline(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_off(x)
    
    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_on(x)