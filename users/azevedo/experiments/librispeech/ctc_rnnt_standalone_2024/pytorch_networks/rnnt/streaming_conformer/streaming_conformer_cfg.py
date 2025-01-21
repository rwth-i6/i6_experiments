from dataclasses import dataclass, field

import torch
from torch import nn
from typing import Any, Callable, Optional, Type, Union, Tuple, Sequence

# FastConformer relevant imports
from i6_models.parts.frontend.generic_frontend import FrontendLayerType
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from .generic_frontend_v2 import GenericFrontendV2Config

# complete Transducer config
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
    ModelConfig as ModelConfigV1,
    SpecaugConfig,
    PredictorConfig,
    VGG4LayerActFrontendV1Config_mod,
    VGG4LayerActFrontendV1Config
)

@dataclass(kw_only=True)
class FastConformerFrontendV1Factory:
    """Implements FastConformer's frontend as a GenericFrontend:

    Rekesh, Dima, et al. "Fast conformer with linearly scalable attention for efficient speech recognition." 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023.
    """

    in_features: int

    n_layers: int
    depthw_sep_convs: Sequence[FrontendLayerType]
    depthw_sep_strides: Sequence[Tuple[int, int]]
    depthw_sep_kernel_sizes: Sequence[Tuple[int, int]]
    out_channels: Optional[Sequence[int]]

    activation_str: str

    out_features: int

    frontend_cfg: GenericFrontendV2Config = field(default=None)

    def __post_init__(self):
        # first layer is normal conv2d with stride 2
        layer_ordering = [FrontendLayerType.Conv2d, FrontendLayerType.Activation]
        conv_strides = [(2, 2)]
        conv_kernel_sizes = [(3, 3)]
        conv_groups = [1]
        out_channels = [self.out_channels[0]]

        layer_ordering += self.depthw_sep_convs * (self.n_layers - 1)
        conv_strides += self.depthw_sep_strides * (self.n_layers - 1)
        conv_kernel_sizes += self.depthw_sep_kernel_sizes * (self.n_layers - 1)
        conv_groups += [-1, 1] * (self.n_layers - 1)
        out_channels += self.out_channels * (self.n_layers - 1)

        num_activations = layer_ordering.count(FrontendLayerType.Activation)
        activations = [self.activation_str.lower() for _ in range(num_activations)]

        self.frontend_cfg = GenericFrontendV2Config(
            in_features=self.in_features,
            layer_ordering=layer_ordering,
            conv_groups=conv_groups,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            conv_out_dims=out_channels,
            conv_paddings=None,
            pool_kernel_sizes=None,
            pool_paddings=None,
            pool_strides=None,
            activations=activations,
            out_features=self.out_features
        )()

    def __call__(self) -> GenericFrontendV2Config:
        return self.frontend_cfg


@dataclass
class ModelConfig(ModelConfigV1):
    frontend_config: Union[GenericFrontendV2Config, VGG4LayerActFrontendV1Config_mod]
    fastemit_lambda: Optional[float]
    chunk_size: Optional[float]

    use_vgg: Optional[bool]

    def __post_init__(self):
        if self.use_vgg is None:
            self.use_vgg = isinstance(self.frontend_config,
                                        Union[VGG4LayerActFrontendV1Config_mod, VGG4LayerActFrontendV1Config])

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])

        if d.get("use_vgg", False):
            print(d["frontend_config"])
            d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        else:
            d["frontend_config"] = GenericFrontendV2Config.from_dict(d["frontend_config"])
            
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["predictor_config"] = PredictorConfig.from_dict(d["predictor_config"])
        return ModelConfig(**d)