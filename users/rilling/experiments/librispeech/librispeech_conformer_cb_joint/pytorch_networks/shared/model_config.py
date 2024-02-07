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
    hidden_channels: int  # Number of hidden channels
    kernel_size: int  # Kernel Size for convolutions in coupling blocks
    dilation_rate: int  # Dilation Rate to define dilation in convolutions of coupling block
    n_blocks: int  # Number of coupling blocks
    n_layers: int  # Number of layers in CNN of the coupling blocks
    p_dropout: float  # Dropout probability for CNN in coupling blocks. 
    n_split: int  #  Number of splits for the 1x1 convolution for flows in the decoder. 
    n_sqz: int  #  n_sqz (int, optional): Squeeze. 
    sigmoid_scale: bool  # sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. 
    ddi: bool # Define if data-dependent initialization is necessary

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return FlowDecoderConfig(**d)

@dataclass
class ConformerCouplingFlowDecoderConfig(FlowDecoderConfig):
    n_heads: int # Number of heads in coupling block conformer net
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return ConformerCouplingFlowDecoderConfig(**d)


@dataclass
class TextEncoderConfig(ModelConfiguration):
    n_vocab: Union[tk.Variable, int]  #  Size of vocabulary for embeddings
    hidden_channels: int  # hidden_channels (int): Number of hidden channels
    filter_channels: int  # filter_channels (int): Number of filter channels
    filter_channels_dp: int  #  filter_channels_dp (int): Number of filter channels for duration predictor
    n_heads: int  # n_heads (int): Number of heads in encoder's Multi-Head Attention
    n_layers: int  # n_layers (int): Number of layers consisting of Multi-Head Attention and CNNs in encoder
    kernel_size: int  # kernel_size (int): Kernel Size for CNNs in encoder layers
    p_dropout: float  #  p_dropout (float): Dropout probability for both encoder and duration predictor
    window_size: int  # Window size  in Multi-Head Self-Attention for encoder. 
    block_length: Union[
        int, None
    ]  # Block length for optional block masking in Multi-Head Attention for encoder. 
    mean_only: bool  # Boolean to only project text encodings to mean values instead of mean and std. 
    prenet: bool  # Boolean to add ConvReluNorm prenet before encoder . 

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return TextEncoderConfig(**d)

@dataclass
class ConformerASREncoderConfig(ModelConfiguration):
    conformer_size: int # channel size of the conformer input / output
    ff_dim: int # number hidden dimensions in ff
    ff_dropout: float # dropout probability for dropout after ff
    num_heads: int # Number of attention heads in the multi-head self-attention
    att_weights_dropout: float # dropout probability of dropout after attention weights
    mhsa_dropout: float # dropout probability of dropout after multi-head self-attention
    kernel_size: int # kernel size of the convolutional block in conformer
    conv_dropout: float # dropout probability of dropout after convolutional block 
    num_layers: int # Number of Conformer layers

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return ConformerASREncoderConfig(**d)

@dataclass
class ModelConfig:
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: Union[SpecaugConfig, None]
    decoder_config: ConformerCouplingFlowDecoderConfig
    text_encoder_config: TextEncoderConfig
    conformer_asr_encoder_config: ConformerASREncoderConfig
    out_channels: int
    gin_channels: int
    final_dropout: float
    label_target_size: int
    specaug_start_epoch: int
    n_speakers: Union[tk.Variable, int]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["decoder_config"] = ConformerCouplingFlowDecoderConfig.from_dict(d["decoder_config"])
        d["text_encoder_config"] = TextEncoderConfig.from_dict(d["text_encoder_config"])
        d["conformer_asr_encoder_config"] = ConformerASREncoderConfig.from_dict(d["conformer_asr_encoder_config"])
        return ModelConfig(**d)
