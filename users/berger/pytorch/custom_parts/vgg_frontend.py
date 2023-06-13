from dataclasses import dataclass

import torch

from i6_models.config import ModelConfiguration


@dataclass
class VGGFrontendConfigV1(ModelConfiguration):
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv_kernel_size: int
    conv1_stride: int
    conv2_stride: int
    conv3_stride: int
    pool_size: int

    def __post__init__(self):
        super().__post_init__()
        assert self.conv_kernel_size % 2 == 1, "Conv kernel size must be odd."


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
            stride=config.conv2_stride,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=config.conv2_channels,
            out_channels=config.conv3_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            stride=config.conv3_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, None, :, :]  # [B, 1, T, F]
        x = self.conv1(x)  # [B, C_1, T', F]
        x = torch.nn.functional.silu(x)  # [B, C_1, T', F]
        x = self.pool(x)  # [B, C_1, T', F']
        x = self.conv2(x)  # [B, C_2, T'', F']
        x = torch.nn.functional.silu(x)  # [B, C_2, T'', F']
        x = self.conv3(x)  # [B, C_3, T''', F']
        x = torch.nn.functional.silu(x)  # [B, C_3, T''', F'']
        return x
