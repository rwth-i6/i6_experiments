from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from i6_models.config import ModelConfiguration


@dataclass
class VGGFrontendConfigV1(ModelConfiguration):
    num_inputs: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv_kernel_size: int
    conv1_stride: int
    conv2_stride: int
    conv3_stride: int
    pool_size: int
    linear_size: int
    dropout: float

    def __post__init__(self):
        super().__post_init__()
        assert self.conv_kernel_size % 2 == 1, "Conv kernel size must be odd."


def subsample_mask(sequence_mask: torch.Tensor, subsampling_factor: int):
    if subsampling_factor == 1:
        return sequence_mask

    max_len = sequence_mask.shape[1]

    padding = 0
    if (overhang := max_len % subsampling_factor) != 0:
        padding = subsampling_factor - overhang

    padded_mask = torch.nn.functional.pad(sequence_mask, pad=(0, padding), value=0)

    reshaped_mask = padded_mask.reshape(padded_mask.shape[0], -1, subsampling_factor)

    subsampled_mask = torch.all(reshaped_mask == 1, dim=2)
    subsampled_mask = subsampled_mask.type(sequence_mask.dtype)

    return subsampled_mask


class VGGFrontendV1(torch.nn.Module):
    def __init__(self, config: VGGFrontendConfigV1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=config.conv1_channels,
            kernel_size=(config.conv_kernel_size, 1),
            padding=(config.conv_kernel_size // 2, 0),
            stride=(config.conv1_stride, 1),
        )
        self.pool = torch.nn.MaxPool2d(
            kernel_size=(1, config.pool_size),
            stride=(1, config.pool_size),
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=config.conv1_channels,
            out_channels=config.conv2_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            stride=(config.conv2_stride, 1),
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=config.conv2_channels,
            out_channels=config.conv3_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            stride=(config.conv3_stride, 1),
        )
        self.linear = torch.nn.Linear(
            config.conv3_channels * (config.num_inputs // config.pool_size), config.linear_size
        )
        self.subsample_factor = config.conv1_stride * config.conv2_stride * config.conv3_stride
        self.dropout = config.dropout
        self.layer_norm = torch.nn.LayerNorm(config.linear_size)

    def forward(
        self, x: torch.Tensor, sequence_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x[:, None, :, :]  # [B, 1, T, F]
        x = self.conv1(x)  # [B, C_1, T', F]
        x = torch.nn.functional.silu(x)  # [B, C_1, T', F]
        x = self.pool(x)  # [B, C_1, T', F']
        x = self.conv2(x)  # [B, C_2, T'', F']
        x = torch.nn.functional.silu(x)  # [B, C_2, T'', F']
        x = self.conv3(x)  # [B, C_3, T''', F']
        x = torch.nn.functional.silu(x)  # [B, C_3, T''', F']
        x = torch.transpose(x, 1, 2)  # [B, T''', C_3, F']
        x = torch.flatten(x, start_dim=2)  # [B, T''', C_3 * F']
        x = self.linear(x)  # [B, T''', F'']
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)  # [B, T''', F'']
        x = self.layer_norm(x)  # [B, T''', F'']

        if sequence_mask is None:
            subsampled_mask = None
        else:
            subsampled_mask = subsample_mask(sequence_mask, self.subsample_factor)

        return x, subsampled_mask
