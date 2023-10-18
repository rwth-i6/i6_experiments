import torch
import torch.nn.functional
from torch import nn

from i6_models.config import ModelConfiguration

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ConformerConv2dFrontendConfig(ModelConfiguration):
    strides: List[int]
    input_dim: int
    output_dim: int


class ConformerConv2dFrontend(nn.Module):
    def __init__(self, cfg: ConformerConv2dFrontendConfig):
        super().__init__()

        strides = cfg.strides
        assert len(strides) == 2, f"Expected 2 strides, got {len(strides)}"

        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # max pool over features
            # stride over time
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=strides[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=strides[1]),
            nn.ReLU(),
        )
        # TODO: compute input_dim using the formula wo hard-coding the value
        self.linear_proj = nn.Linear(in_features=2560, out_features=cfg.output_dim, bias=False)
        self.linear_proj_dropout = nn.Dropout(p=0.1)

    def forward(self, tensor: torch.Tensor, seq_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Conv2d expects an input of shape [B,C,H=T,W=F]
        tensor = tensor.unsqueeze(1)  # [B,1,T,F]

        tensor = self.conv(tensor)  # [B,64,T',F']
        tensor = tensor.transpose(1, 2)  # [B,T',64,F']
        tensor = tensor.flatten(2)  # [B,T',64*F']

        tensor = self.linear_proj(tensor)  # [B,T',512]
        tensor = self.linear_proj_dropout(tensor)  # [B,T',512]

        # TODO: modify seq_mask w.r.t subsample rate

        return tensor, seq_mask
