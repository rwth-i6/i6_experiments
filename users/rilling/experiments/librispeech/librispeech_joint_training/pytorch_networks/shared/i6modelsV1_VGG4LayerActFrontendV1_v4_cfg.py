"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Type, Union
from sisyphus import tk

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration


@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config_mod(VGG4LayerActFrontendV1Config):
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


@dataclass
class ConformerEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV1Config


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            d = d.copy()
            return SpecaugConfig(**d)
        else:
            return None

@dataclass
class FlowDecoderConfig(ModelConfiguration):
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_blocks: int
    n_layers: int
    p_dropout: float
    n_split: int
    n_sqz: int
    sigmoid_scale: bool

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return FlowDecoderConfig(**d)
        
@dataclass
class TextEncoderConfig(ModelConfiguration):
    n_vocab: Union[tk.Variable, int]
    hidden_channels: int
    filter_channels: int
    filter_channels_dp: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    window_size: int
    block_length: Union[int, None]
    mean_only: bool
    prenet: bool
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return TextEncoderConfig(**d)

@dataclass
class ModelConfig():
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: Union[SpecaugConfig, None]
    decoder_config: FlowDecoderConfig
    text_encoder_config: TextEncoderConfig
    specauc_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    conv_kernel_size: int
    final_dropout: float
    out_channels: int
    gin_channels: int
    n_speakers: Union[tk.Variable, int]
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["decoder_config"] = FlowDecoderConfig.from_dict(d["decoder_config"])
        d["text_encoder_config"] = TextEncoderConfig.from_dict(d["text_encoder_config"])
        return ModelConfig(**d)


