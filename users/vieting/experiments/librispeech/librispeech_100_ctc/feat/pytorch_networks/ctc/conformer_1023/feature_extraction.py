from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class SupervisedConvolutionalFeatureExtractionV1Config(ModelConfiguration):
    """
    Attributes:
        wave_norm: normalize waveform over time dim, then learnable scale and biases
        num_tf: number of filters in first conv layer (for time-frequency decomposition)
        size_tf: filter size in first conv layer
        stride_tf: stride in first conv layer
        num_env: number of filters in second conv layer (for envelope extraction)
        size_env: filter size in second conv layer
        stride_env: stride in second conv layer
    """

    wave_norm: bool
    num_tf: int
    size_tf: int
    stride_tf: int
    num_env: int
    size_env: int
    stride_env: int

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.num_tf > 0 and self.num_env > 0, "number of filters needs to be positive"
        assert self.size_tf > 0 and self.size_env > 0, "filter sizes need to be positive"
        assert self.stride_tf > 0 and self.stride_env > 0, "strides need to be positive"


class SupervisedConvolutionalFeatureExtractionV1(nn.Module):
    """
    Module which applies conv layers to the raw waveform and pools using multi resolutional learned filters similar to
    Z. Tüske, R. Schlüter, and H. Ney.
    Acoustic modeling of Speech Waveform based on Multi-Resolution, Neural Network Signal Processing.
    ICASSP 2018
    https://www-i6.informatik.rwth-aachen.de/publications/download/1097/Tueske-ICASSP-2018.pdf

    Was also used and referred to as supervised convolutional features (SCF) in
    Peter Vieting, Christoph Lüscher, Wilfried Michel, Ralf Schlüter, Hermann Ney
    ON ARCHITECTURES AND TRAINING FOR RAW WAVEFORM FEATURE EXTRACTION IN ASR
    ASRU 2021
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9688123&tag=1
    """

    def __init__(self, cfg: SupervisedConvolutionalFeatureExtractionV1Config):
        super().__init__()
        self.wave_norm = nn.Conv1d(1, 1, 1) if cfg.wave_norm else None
        self.conv_tf = nn.Conv1d(1, cfg.num_tf, cfg.size_tf, stride=cfg.stride_tf, bias=False)
        self.conv_env = nn.Conv2d(1, cfg.num_env, (1, cfg.size_env), stride=(1, cfg.stride_env), bias=False)
        self.normalization_env = nn.LayerNorm(cfg.num_tf * cfg.num_env)

    def forward(self, raw_audio: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length: in samples [B]
        :return features as [B,T'',F]
        """
        raw_audio = torch.unsqueeze(raw_audio, 1)  # [B,1,T]

        if self.wave_norm is not None:
            raw_audio = raw_audio - torch.mean(raw_audio, dim=-1)[:, None]
            raw_audio = raw_audio * torch.rsqrt(torch.var(raw_audio, dim=-1)[:, None] + 1e-6)
            raw_audio = self.wave_norm(raw_audio)

        feature_data = self.conv_tf(raw_audio)  # [B,F',T]
        feature_data = feature_data.abs()
        feature_data = torch.unsqueeze(feature_data, 1)  # [B,1,F1,T']
        feature_data = self.conv_env(feature_data)  # [B,F2,F1,T'']
        feature_data = torch.flatten(feature_data.transpose(1, 2), 1, 2)  # [B,F,T'']
        feature_data = torch.pow(feature_data.abs() + 1e-5, 1 / 2.5)
        feature_data = feature_data.transpose(1, 2)  # [B,T'',F]
        feature_data = self.normalization_env(feature_data)

        length = ((length - self.conv_tf.kernel_size[-1]) / self.conv_tf.stride[-1] + 1).int()
        length = ((length - self.conv_env.kernel_size[-1]) / self.conv_env.stride[-1] + 1).int()

        return feature_data, length


@dataclass
class FeatureExtractionConfig(ModelConfiguration):
    """
    Attributes:
        module_class:
    """

    module_class: str


@dataclass
class SupervisedConvolutionalFeatureExtractionV2Config(FeatureExtractionConfig):
    """
    Attributes:
        scf_config: config for the default SCF module
        convs: size, channels and groups for subsequent convolutions used to reduce feature dimension
    """
    scf_config: SupervisedConvolutionalFeatureExtractionV1Config
    convs: List[Tuple[int, int, int]]  # [(size, channels, groups)]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["scf_config"] = SupervisedConvolutionalFeatureExtractionV1Config(**d["scf_config"])
        return SupervisedConvolutionalFeatureExtractionV2Config(**d)


class SupervisedConvolutionalFeatureExtractionV2(SupervisedConvolutionalFeatureExtractionV1):
    """
    Like V1, but with additional convolutions to reduce the feature dimension after the envelope extraction.
    """

    def __init__(self, cfg: SupervisedConvolutionalFeatureExtractionV2Config):
        super().__init__(cfg.scf_config)
        dim_in = cfg.scf_config.num_tf * cfg.scf_config.num_env
        self.convs_reduction = []
        for size, channels, groups in cfg.convs:
            self.convs_reduction.append(nn.Conv1d(dim_in, channels, size, groups=groups, bias=False))
            dim_in = channels

    def forward(self, raw_audio: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length: in samples [B]
        :return features as [B,T'',F']
        """
        feature_data, length = super().forward(raw_audio, length)    # [B,T'',F]

        feature_data = feature_data.transpose(1, 2)  # [B,F,T'']
        for conv in self.convs_reduction:
            conv.to(feature_data.device)
            feature_data = conv(feature_data)
        feature_data = feature_data.transpose(1, 2)  # [B,T'',F']

        return feature_data, length
