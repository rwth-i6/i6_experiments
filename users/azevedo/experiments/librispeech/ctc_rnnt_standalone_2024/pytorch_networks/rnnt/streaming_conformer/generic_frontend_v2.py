from __future__ import annotations

__all__ = [
    "GenericFrontendV2Config",
    "GenericFrontendV2",
]

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Callable, Optional, Tuple, Union, Sequence

import torch
from torch import nn

from i6_models.config import ModelConfiguration

from i6_models.parts.frontend.common import get_same_padding, mask_pool, calculate_output_dim
from i6_models.parts.frontend.generic_frontend import GenericFrontendV1Config, FrontendLayerType, GenericFrontendV1


@dataclass(kw_only=True)
class GenericFrontendV2Config(GenericFrontendV1Config):
    """
    Attributes:
        in_features: number of input features to module
        layer_ordering: the ordering of the front end layer sequences, the ordering element must be selected from FrontendLayerType
            e.g. the ordering of VGG4LayerActFrontendV1 would be [FrontendLayerType.Conv2d, FrontendLayerType.Activation,
            FrontendLayerType.Pool2d, FrontendLayerType.Conv2d, FrontendLayerType.Conv2d, FrontendLayerType.Activation,
            FrontendLayerType.Pool2d]
        conv_kernel_sizes: kernel sizes for each conv layer
        conv_strides: stride sizes for each conv layer
        conv_paddings: paddings sizes for each conv layer
        conv_out_dims: number of out channels for each conv layer
        pool_kernel_sizes: kernel sizes for each pool layer
        pool_strides: stride sizes for each pool layer
        pool_paddings: padding sizes for each pool layer
        activations: activation functions
        out_features: output size of the final linear layer
    """
    layer_ordering: Sequence[Union[FrontendLayerType, str]]
    conv_groups: Optional[Sequence[int]]
    activations: Optional[Sequence[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor], str]]]

    def check_valid(self):
        num_convs = 0 if self.conv_kernel_sizes is None else len(self.conv_kernel_sizes)
        num_pools = 0 if self.pool_kernel_sizes is None else len(self.pool_kernel_sizes)
        num_activations = 0 if self.activations is None else len(self.activations)

        print(f"{num_convs = }, {self.layer_ordering.count(FrontendLayerType.Conv2d) = }")
        assert num_convs == self.layer_ordering.count(
            FrontendLayerType.Conv2d
        ), "Number of convolution layers mismatch!"
        assert num_activations == self.layer_ordering.count(
            FrontendLayerType.Activation
        ), "Number of activation layers mismatch!"
        assert num_pools == self.layer_ordering.count(FrontendLayerType.Pool2d), "Number of pooling layers mismatch!"

        if self.conv_strides is not None:
            assert len(self.conv_strides) == num_convs, "Please specify stride for each convolution layer!"
        if self.conv_paddings is not None:
            assert len(self.conv_paddings) == num_convs, "Please specify padding for each convolution layer!"
        if num_convs != 0:
            assert (
                len(self.conv_out_dims) == num_convs
            ), "Please specify the number of channels for each convolution layer!"
        if self.conv_groups is not None:
            assert len(self.conv_groups) == num_convs, "Please specify groups for each convolution layer!"

        if self.pool_strides is not None:
            assert len(self.pool_strides) == num_pools, "Please specify stride for each pooling layer!"
        if self.pool_paddings is not None:
            assert len(self.pool_paddings) == num_pools, "Please specify padding for each pooling layer!"

        assert len(self.layer_ordering) == num_convs + num_pools + num_activations, "Number of total layers mismatch!"

        for kernel_sizes in filter(None, [self.conv_kernel_sizes, self.pool_kernel_sizes]):
            for kernel_size in kernel_sizes:
                assert all(k % 2 for k in kernel_size), "ConformerVGGFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()

    # FIXME: please don't look. Returnn and dataclasses are giving me no choice...
    def __call__(self) -> GenericFrontendV2Config:
        str_layer_ordering = [str(x) if isinstance(x, FrontendLayerType) else x for x in self.layer_ordering]
        return replace(self, layer_ordering=str_layer_ordering)

    @classmethod
    def from_dict(cls, d):
        layertype_map = FrontendLayerType.__members__   # e.g. {"Activation": FrontendLayerType.Activation, ...}
        activation_map = {"relu": nn.ReLU()}
        
        d["layer_ordering"] = [x.replace(f"{FrontendLayerType.__name__}.", "") for x in d["layer_ordering"]]
        d["layer_ordering"] = [layertype_map[x] if x in layertype_map else x for x in d["layer_ordering"]]

        d["activations"] = [activation_map[x] for x in d["activations"]]
        return GenericFrontendV2Config(**d)



class GenericFrontendV2(nn.Module):
    def __init__(self, model_cfg: GenericFrontendV2Config):
        """
        Generic Front-End
        can be used to generate customized frontend by combining convolutional and pooling layers, as well as activation
        functions differently

        To get the ESPnet case, for example Conv2dSubsampling6, use these options
            layer_ordering = [FrontendLayerType.Conv2d, FrontendLayerType.Conv2d]
            conv_kernel_sizes = [3, 5]
            strides = [2, 3]

        To get the i6_models VGG4LayerActFrontendV1, use the options:
            layer_ordering = [FrontendLayerType.Conv2d, FrontendLayerType.Activation, FrontendLayerType.Pool2d,
                FrontendLayerType.Conv2d, FrontendLayerType.Conv2d, FrontendLayerType.Activation, FrontendLayerType.Pool2d]
            conv_kernel_sizes = [3, 3, 3]
            conv_out_dims = [32, 34, 64]
            pool_kernel_sizes = [3, 3]
            pool_strides = [2, 2]
            activations = [torch.nn.ReLU(), torch.nn.ReLU()]
        """
        super().__init__()

        model_cfg.check_valid()

        self.cfg = model_cfg

        self.frontend_layers = nn.ModuleList([])

        conv_layer_index = 0
        pool_layer_index = 0
        activation_layer_index = 0
        last_channel_dim = 1
        last_feat_dim = model_cfg.in_features
        for layer_type in model_cfg.layer_ordering:
            if layer_type == FrontendLayerType.Conv2d:
                conv_out_dim = model_cfg.conv_out_dims[conv_layer_index]
                conv_kernel_size = model_cfg.conv_kernel_sizes[conv_layer_index]
                conv_stride = 1 if model_cfg.conv_strides is None else model_cfg.conv_strides[conv_layer_index]
                conv_padding = (
                    get_same_padding(conv_kernel_size)
                    if model_cfg.conv_paddings is None
                    else model_cfg.conv_paddings[conv_layer_index]
                )
                conv_groups = 1 if model_cfg.conv_groups is None else model_cfg.conv_groups[conv_layer_index]
                if conv_groups == -1:   # depthwise convolution
                    conv_groups = last_channel_dim
                    conv_out_dim = last_channel_dim if conv_out_dim == -1 else conv_out_dim

                self.frontend_layers.append(
                    nn.Conv2d(
                        in_channels=last_channel_dim,
                        out_channels=conv_out_dim,
                        kernel_size=conv_kernel_size,
                        stride=conv_stride,
                        padding=conv_padding,
                        groups=conv_groups,
                    )
                )

                last_channel_dim = conv_out_dim
                last_feat_dim = calculate_output_dim(
                    in_dim=last_feat_dim,
                    filter_size=conv_kernel_size[1],
                    stride=conv_stride[1],
                    padding=conv_padding[1],
                )
                conv_layer_index += 1

            elif layer_type == FrontendLayerType.Pool2d:
                pool_stride = None if model_cfg.pool_strides is None else model_cfg.pool_strides[pool_layer_index]
                pool_kernel_size = model_cfg.pool_kernel_sizes[pool_layer_index]
                pool_padding = (
                    get_same_padding(pool_kernel_size)
                    if model_cfg.pool_paddings is None
                    else model_cfg.pool_paddings[pool_layer_index]
                )

                self.frontend_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_kernel_size,
                        stride=pool_stride,
                        padding=pool_padding,
                    )
                )
                last_feat_dim = calculate_output_dim(
                    in_dim=last_feat_dim,
                    filter_size=pool_kernel_size[1],
                    stride=pool_stride[1] or pool_kernel_size[1],
                    padding=pool_padding[1],
                )
                pool_layer_index += 1

            elif layer_type == FrontendLayerType.Activation:
                self.frontend_layers.append(model_cfg.activations[activation_layer_index])
                activation_layer_index += 1
            else:
                raise NotImplementedError

            self.linear = nn.Linear(
                in_features=last_feat_dim * last_channel_dim,
                out_features=model_cfg.out_features,
                bias=True,
            )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert tensor.shape[-1] == self.cfg.in_features, f"shape {tensor.shape} vs in features {self.cfg.in_features}"
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        for layer in self.frontend_layers:
            tensor = layer(tensor)

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                sequence_mask = mask_pool(
                    sequence_mask,
                    kernel_size=layer.kernel_size[0],
                    stride=layer.stride[0],
                    padding=layer.padding[0],
                )

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F"]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F"]

        tensor = self.linear(tensor)

        return tensor, sequence_mask