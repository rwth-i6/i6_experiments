from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import torch
from torch import nn

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from ....streamable_module import StreamableModule
from ....base_config import BaseConfig




@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config(BaseConfig):
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
    out_features: int
    activation_str: str = ""
    activation: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None


    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation_str = d.pop("activation_str")
        if activation_str == "ReLU":
            from torch.nn import ReLU
            activation = ReLU()
        else:
            assert False, "Unsupported activation %s" % d["activation_str"]
        d["activation"] = activation
        return VGG4LayerActFrontendV1Config(**d)

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

    def module(self):
        return StreamableVGG4LayerActFrontendV1



class StreamableVGG4LayerActFrontendV1(StreamableModule):
    def __init__(self, model_cfg: VGG4LayerActFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        self.vgg4_act_frontend: VGG4LayerActFrontendV1 = VGG4LayerActFrontendV1(model_cfg)

    def forward_offline(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor):
        """
        :param data_tensor: tensor with shape [B, T, F]
        :param sequence_mask: tensor with shape [B, T]

        :return: tensor of shape [B, T', E]
        """
        return self.vgg4_act_frontend(data_tensor, sequence_mask)
    
    def forward_streaming(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor):
        """
        :param data_tensor: tensor with shape [B, N, C, E]
        :param sequence_mask: tensor with shape [B, N, C]

        :return: tensor of shape [B, N, C', E] and sequence mask [B, N, C']
        """
        batch_sz, num_chunks, _, _ = data_tensor.shape

        data_tensor = data_tensor.flatten(0, 1)
        sequence_mask = sequence_mask.flatten(0, 1)

        x, sequence_mask = self.forward_offline(data_tensor, sequence_mask)

        x = x.view(batch_sz, num_chunks, -1, x.size(-1))
        sequence_mask = sequence_mask.view(batch_sz, num_chunks, sequence_mask.size(-1))

        return x, sequence_mask
    
    def infer(self, *args, **kwargs):
        return super().infer(*args, **kwargs)


