from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from i6_models.config import ModelConfiguration
from i6_models.parts.frontend.common import get_same_padding, calculate_output_dim


def mask_pool(seq_mask: torch.Tensor, *, kernel_size: int, stride: int, padding: int, ceil: bool) -> torch.Tensor:
    """
    apply strides to the masking

    :param seq_mask: [B,T]
    :param kernel_size:
    :param stride:
    :param padding:
    :return: [B,T'] using maxpool
    """
    if stride == 1 and 2 * padding == kernel_size - 1:
        return seq_mask

    seq_mask = seq_mask.float()
    seq_mask = torch.unsqueeze(seq_mask, 1)  # [B,1,T]
    seq_mask = torch.nn.functional.max_pool1d(seq_mask, kernel_size, stride, padding, ceil_mode=ceil)  # [B,1,T']
    seq_mask = torch.squeeze(seq_mask, 1)  # [B,T']
    seq_mask = seq_mask.bool()
    return seq_mask


@dataclass
class VGG4LayerActFrontendCeilPoolV1Config(ModelConfiguration):
    """
    Attributes:
        in_features: number of input features to module
        conv1_channels: number of channels for first conv layer
        conv2_channels: number of channels for second conv layer
        conv3_channels: number of channels for third conv layer
        conv4_channels: number of channels for fourth conv layer
        conv_kernel_size: kernel size of conv layers
        conv_padding: padding for the convolution
        pool1_kernel_size: kernel size of first pooling layer
        pool1_stride: stride of first pooling layer
        pool1_padding: padding for first pooling layer
        pool2_kernel_size: kernel size of second pooling layer
        pool2_stride: stride of second pooling layer
        pool2_padding: padding for second pooling layer
        activation: activation function at the end
        out_features: output size of the final linear layer
    """

    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: Tuple[int, int]
    conv_padding: Optional[Tuple[int, int]]
    pool1_kernel_size: Tuple[int, int]
    pool1_stride: Optional[Tuple[int, int]]
    pool1_padding: Optional[Tuple[int, int]]
    pool2_kernel_size: Tuple[int, int]
    pool2_stride: Optional[Tuple[int, int]]
    pool2_padding: Optional[Tuple[int, int]]
    activation: Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    out_features: int

    def check_valid(self):
        if isinstance(self.conv_kernel_size, int):
            assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        if isinstance(self.pool1_kernel_size, int):
            assert self.pool1_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        if isinstance(self.pool2_kernel_size, int):
            assert self.pool2_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class VGG4LayerActFrontendCeilPoolV1(torch.nn.Module):
    """
    Convolutional Front-End

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
      - Conv
      - Conv
      - Activation
      - Pool
      - Conv
      - Conv
      - Activation
      - Pool

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerActFrontendCeilPoolV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        self.cfg = model_cfg

        conv_padding = (
            model_cfg.conv_padding
            if model_cfg.conv_padding is not None
            else get_same_padding(model_cfg.conv_kernel_size)
        )
        pool1_padding = model_cfg.pool1_padding if model_cfg.pool1_padding is not None else (0, 0)
        pool2_padding = model_cfg.pool2_padding if model_cfg.pool2_padding is not None else (0, 0)

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=model_cfg.conv1_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=model_cfg.conv1_channels,
            out_channels=model_cfg.conv2_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=model_cfg.pool1_kernel_size,
            stride=model_cfg.pool1_stride,
            padding=pool1_padding,
            ceil_mode=True,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=model_cfg.conv2_channels,
            out_channels=model_cfg.conv3_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=model_cfg.conv3_channels,
            out_channels=model_cfg.conv4_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=model_cfg.pool2_kernel_size,
            stride=model_cfg.pool2_stride,
            padding=pool2_padding,
            ceil_mode=True,
        )
        self.activation = model_cfg.activation
        self.linear = torch.nn.Linear(
            in_features=self._calculate_dim(),
            out_features=model_cfg.out_features,
            bias=True,
        )

    def forward(
        self, tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        T might be reduced to T' or T'' depending on stride of the layers

        stride is only allowed for the pool1 and pool2 operation.
        other ops do not have stride configurable -> no update of mask sequence required but added anyway

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T",F'] and the shape of the sequence mask
        """
        assert tensor.shape[-1] == self.cfg.in_features, f"shape {tensor.shape} vs in features {self.cfg.in_features}"
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        if sequence_mask is not None:
            sequence_mask = mask_pool(
                seq_mask=sequence_mask,
                kernel_size=self.conv1.kernel_size[0],
                stride=self.conv1.stride[0],
                padding=self.conv1.padding[0],
                ceil=False,
            )

        tensor = self.conv2(tensor)
        if sequence_mask is not None:
            sequence_mask = mask_pool(
                sequence_mask,
                kernel_size=self.conv2.kernel_size[0],
                stride=self.conv2.stride[0],
                padding=self.conv2.padding[0],
                ceil=False,
            )

        tensor = self.activation(tensor)
        tensor = self.pool1(tensor)  # [B,C,T',F']
        if sequence_mask is not None:
            sequence_mask = mask_pool(
                sequence_mask,
                kernel_size=self.pool1.kernel_size[0],
                stride=self.pool1.stride[0],
                padding=self.pool1.padding[0],
                ceil=True,
            )

        tensor = self.conv3(tensor)
        if sequence_mask is not None:
            sequence_mask = mask_pool(
                sequence_mask,
                kernel_size=self.conv3.kernel_size[0],
                stride=self.conv3.stride[0],
                padding=self.conv3.padding[0],
                ceil=False,
            )

        tensor = self.conv4(tensor)
        if sequence_mask is not None:
            sequence_mask = mask_pool(
                sequence_mask,
                kernel_size=self.conv4.kernel_size[0],
                stride=self.conv4.stride[0],
                padding=self.conv4.padding[0],
                ceil=False,
            )

        tensor = self.activation(tensor)
        tensor = self.pool2(tensor)  # [B,C,T",F"]
        if sequence_mask is not None:
            sequence_mask = mask_pool(
                sequence_mask,
                kernel_size=self.pool2.kernel_size[0],
                stride=self.pool2.stride[0],
                padding=self.pool2.padding[0],
                ceil=True,
            )

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F"]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F"]

        tensor = self.linear(tensor)

        return tensor, sequence_mask

    def _calculate_dim(self) -> int:
        # conv1
        out_dim = calculate_output_dim(
            in_dim=self.cfg.in_features,
            filter_size=self.conv1.kernel_size[1],
            stride=self.conv1.stride[1],
            padding=self.conv1.padding[1],
        )
        # conv2
        out_dim = calculate_output_dim(
            in_dim=out_dim,
            filter_size=self.conv2.kernel_size[1],
            stride=self.conv2.stride[1],
            padding=self.conv2.padding[1],
        )
        # pool1
        out_dim = calculate_output_dim(
            in_dim=out_dim,
            filter_size=self.pool1.kernel_size[1],
            stride=self.pool1.stride[1],
            padding=self.pool1.padding[1],
        )
        # conv3
        out_dim = calculate_output_dim(
            in_dim=out_dim,
            filter_size=self.conv3.kernel_size[1],
            stride=self.conv3.stride[1],
            padding=self.conv3.padding[1],
        )
        # conv4
        out_dim = calculate_output_dim(
            in_dim=out_dim,
            filter_size=self.conv4.kernel_size[1],
            stride=self.conv4.stride[1],
            padding=self.conv4.padding[1],
        )
        # pool2
        out_dim = calculate_output_dim(
            in_dim=out_dim,
            filter_size=self.pool2.kernel_size[1],
            stride=self.pool2.stride[1],
            padding=self.pool2.padding[1],
        )
        out_dim *= self.conv4.out_channels
        return out_dim
