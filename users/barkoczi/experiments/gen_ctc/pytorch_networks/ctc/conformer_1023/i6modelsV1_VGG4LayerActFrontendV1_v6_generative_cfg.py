from dataclasses import dataclass

from typing import Optional, Tuple


@dataclass
class LogMelFeatureExtractionV1Config:
    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool


@dataclass
class VGG4LayerActFrontendV1ConfigMod:
    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: Tuple[int, int]
    conv_padding: Optional[Tuple[int, int]]
    pool1_kernel_size: Tuple[int, int]
    pool1_stride: Tuple[int, int]
    pool1_padding: Optional[Tuple[int, int]]
    pool2_kernel_size: Tuple[int, int]
    pool2_stride: Tuple[int, int]
    pool2_padding: Optional[Tuple[int, int]]
    out_features: int
    activation_str: str = ""
    activation: Optional[object] = None

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class SpecaugConfig:
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int


@dataclass
class ModelConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1ConfigMod
    specaug_config: SpecaugConfig
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
    sampling_type: str = "batch"
    sampling_ratio: float = 0.1
    share_samples: bool = False
    ratio_corrector: float = 1.0

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        from torch import nn

        from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
        from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config as I6LogMelConfig

        d["feature_extraction_config"] = I6LogMelConfig(**d["feature_extraction_config"])
        frontend_dict = d["frontend_config"].copy()
        activation_str = frontend_dict.pop("activation_str")
        if activation_str == "ReLU":
            frontend_dict["activation"] = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {activation_str!r}")
        d["frontend_config"] = VGG4LayerActFrontendV1Config(**frontend_dict)
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return cls(**d)
