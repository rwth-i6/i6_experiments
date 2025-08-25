from __future__ import annotations

from sisyphus.tools import functools

__all__ = [
    "GenericVGGFrontendV1",
    "GenericVGGFrontendV1Config",
]

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration

from i6_models.parts.frontend.common import get_same_padding, mask_pool, calculate_output_dim


@dataclass
class GenericVGGFrontendV1Config(ModelConfiguration):
    in_features: int
    out_features: int
    conv_channels: List[int]
    conv_kernel_sizes: List[Tuple[int, int]]
    conv_strides: List[Tuple[int, int]]
    pool_sizes: List[Optional[Tuple[int, int]]]
    activations: List[Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]]

    def check_valid(self):
        for kernel_size in self.conv_kernel_sizes:
            assert (
                kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
            ), "GenericVGGActFrontendV1 only supports odd conv kernel sizes"

        assert (
            len(self.conv_channels)
            == len(self.conv_kernel_sizes)
            == len(self.conv_strides)
            == len(self.pool_sizes)
            == len(self.activations)
        ), "All layer parameter lists must have the same length"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class GenericVGGFrontendV1(nn.Module):
    """
    Convolutional Front-End

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
         __
        | Conv
    N x | Activation (optional)
        | Pool (optional)
         __

    Uses explicit same-padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: GenericVGGFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        convs = []
        in_channels = 1
        for channels, kernel_size, stride in zip(
            model_cfg.conv_channels, model_cfg.conv_kernel_sizes, model_cfg.conv_strides
        ):
            padding = get_same_padding(kernel_size)
            assert len(padding) == 2
            convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )
            in_channels = channels

        pools = []
        for kernel_size in model_cfg.pool_sizes:
            if kernel_size is None:
                pools.append(None)
            else:
                pools.append(nn.MaxPool2d(kernel_size=kernel_size, padding=(0, 0)))

        activations = []
        for activation in model_cfg.activations:
            if activation is None:
                activations.append(None)
            else:
                activations.append(activation)

        self.ops = torch.nn.Sequential()  # Ops that are applied to the input features
        self.mask_ops = []  # Ops that are applied to the mask
        out_dim = model_cfg.in_features  # For running out-dim calculation as input for final linear layer
        for conv, pool, act in zip(convs, pools, activations):
            self.ops.append(conv)

            self.mask_ops.append(
                functools.partial(
                    mask_pool, kernel_size=conv.kernel_size[0], stride=conv.stride[0], padding=conv.padding[0]
                )
            )

            out_dim = calculate_output_dim(
                in_dim=out_dim, filter_size=conv.kernel_size[1], stride=conv.stride[1], padding=conv.padding[1]
            )

            if pool is not None:
                self.ops.append(pool)

                self.mask_ops.append(
                    functools.partial(
                        mask_pool, kernel_size=pool.kernel_size[0], stride=pool.stride[0], padding=pool.padding[0]
                    )
                )

                out_dim = calculate_output_dim(
                    in_dim=out_dim, filter_size=pool.kernel_size[1], stride=pool.stride[1], padding=pool.padding[1]
                )

            if act is not None:
                self.ops.append(act)

        out_dim *= convs[-1].out_channels

        self.linear = nn.Linear(
            in_features=out_dim,
            out_features=model_cfg.out_features,
            bias=True,
        )

    def forward(
        self, tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        T might be reduced to T' or T'' depending on stride of the layers

        stride is only allowed for the pool1 and pool2 operation.
        other ops do not have stride configurable -> no update of mask sequence required but added anyway

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T",F'] and the shape of the sequence mask
        """
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,1,T,F]

        tensor = self.ops(tensor)  # [B, C, T', F']

        if sequence_mask is not None:
            for fn in self.mask_ops:
                sequence_mask = fn(sequence_mask)  # [B, T']

        tensor = torch.transpose(tensor, 1, 2)  # [B, T', C, F']
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B, T', C*F']

        tensor = self.linear(tensor)

        return tensor, sequence_mask
