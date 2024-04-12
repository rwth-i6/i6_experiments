from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Type, Union, Tuple
from sisyphus import tk

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration


@dataclass
class DbMelFeatureExtractionConfig():
    """

    :param sample_rate: audio sample rate in Hz
    :param win_size: window size in seconds
    :param hop_size: window shift in seconds
    :param f_min: minimum mel filter frequency in Hz
    :param f_max: maximum mel fitler frequency in Hz
    :param min_amp: minimum amplitude for safe log
    :param num_filters: number of mel windows
    :param center: centered STFT with automatic padding
    :param norm: tuple optional of mean & std_dev for feature normalization
    """
    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool
    norm: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, d):
        return DbMelFeatureExtractionConfig(**d)


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
class MultiscaleFlowDecoderConfig(ModelConfiguration):
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_blocks: int
    n_layers: int
    p_dropout: float
    n_split: int
    n_sqz: int
    n_early_every: int
    sigmoid_scale: bool

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return MultiscaleFlowDecoderConfig(**d)


@dataclass
class ConformerFlowDecoderConfig(ModelConfiguration):
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_blocks: int
    n_layers: int
    n_heads: int
    p_dropout: float
    n_split: int
    n_sqz: int
    sigmoid_scale: bool

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return ConformerFlowDecoderConfig(**d)


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
class EmbeddingTextEncoderConfig(ModelConfiguration):
    n_vocab: Union[tk.Variable, int]
    hidden_channels: int
    filter_channels_dp: int
    kernel_size: int
    p_dropout: float
    mean_only: bool

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return EmbeddingTextEncoderConfig(**d)

@dataclass
class PhonemePredictionConfig(ModelConfiguration):
    n_channels: int
    n_layers: int
    p_dropout: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return PhonemePredictionConfig(**d)


@dataclass
class PhonemePredictionConfigCNN(ModelConfiguration):
    n_channels: int
    n_layers: int
    kernel_size: int
    p_dropout: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return PhonemePredictionConfigCNN(**d)

@dataclass
class PhonemePredictionConfigBLSTM(ModelConfiguration):
    n_channels: int
    n_layers: int
    p_dropout: float
    subsampling_factor: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return PhonemePredictionConfigBLSTM(**d)


@dataclass
class ModelConfigV1:
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: Union[SpecaugConfig, None]
    decoder_config: FlowDecoderConfig
    text_encoder_config: TextEncoderConfig
    ffn_layers: Union[int, None]
    ffn_channels: Union[int, None]
    specauc_start_epoch: Union[int, None]
    label_target_size: Union[int, None]
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
        return ModelConfigV1(**d)


@dataclass
class ModelConfigV2:
    decoder_config: FlowDecoderConfig
    text_encoder_config: Union[TextEncoderConfig, EmbeddingTextEncoderConfig]
    label_target_size: int
    out_channels: int
    gin_channels: int
    n_speakers: Union[tk.Variable, int]
    specauc_start_epoch: Optional[int] = None
    phoneme_prediction_config: Optional[Union[PhonemePredictionConfig, PhonemePredictionConfigCNN, PhonemePredictionConfigBLSTM]] = None
    specaug_config: Optional[SpecaugConfig] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        if "n_heads" in d["decoder_config"].keys():
            d["decoder_config"] = ConformerFlowDecoderConfig.from_dict(d["decoder_config"])
        elif "n_early_every" in d["decoder_config"].keys():
            d["decoder_config"] = MultiscaleFlowDecoderConfig.from_dict(d["decoder_config"])
        else:
            d["decoder_config"] = FlowDecoderConfig.from_dict(d["decoder_config"])
        if "n_heads" in d["text_encoder_config"].keys():
            d["text_encoder_config"] = TextEncoderConfig.from_dict(d["text_encoder_config"])
        else: 
            d["text_encoder_config"] = EmbeddingTextEncoderConfig.from_dict(d["text_encoder_config"])
        if "phoneme_prediction_config" in d.keys() and d["phoneme_prediction_config"] is not None:
            if "kernel_size" in d["phoneme_prediction_config"].keys():
                d["phoneme_prediction_config"] = PhonemePredictionConfigCNN.from_dict(d["phoneme_prediction_config"])
            elif "subsampling_factor" in d["phoneme_prediction_config"].keys():
                d["phoneme_prediction_config"] = PhonemePredictionConfigBLSTM.from_dict(d["phoneme_prediction_config"])
            else:
                d["phoneme_prediction_config"] = PhonemePredictionConfig.from_dict(d["phoneme_prediction_config"])
        return ModelConfigV2(**d)
