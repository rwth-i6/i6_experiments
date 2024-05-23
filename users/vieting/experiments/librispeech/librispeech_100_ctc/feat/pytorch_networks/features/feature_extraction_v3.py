from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import nn

from i6_models.config import ModelConfiguration
from .specaugment_sorted_v1 import specaugment_v1_by_length


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int


@dataclass
class FeatureExtractionConfig(ModelConfiguration):
    """
    Attributes:
        module_class:
    """

    module_class: str

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        module_class = globals()[d["module_class"] + "Config"]
        return module_class(**d)


@dataclass
class SupervisedConvolutionalFeatureExtractionV3Config(FeatureExtractionConfig):
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
    specaug_config: SpecaugConfig
    specaug_start_epoch: int

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.num_tf > 0 and self.num_env > 0, "number of filters needs to be positive"
        assert self.size_tf > 0 and self.size_env > 0, "filter sizes need to be positive"
        assert self.stride_tf > 0 and self.stride_env > 0, "strides need to be positive"

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return cls(**d)


class SupervisedConvolutionalFeatureExtractionV3(nn.Module):
    """
    Like V1, but this includes SpecAugment where the masks are based on sorted filters.
    """

    def __init__(self, cfg: SupervisedConvolutionalFeatureExtractionV3Config):
        super().__init__()
        self.wave_norm = nn.Conv1d(1, 1, 1) if cfg.wave_norm else None
        self.conv_tf = nn.Conv1d(1, cfg.num_tf, cfg.size_tf, stride=cfg.stride_tf, bias=False)
        self.conv_env = nn.Conv2d(1, cfg.num_env, (1, cfg.size_env), stride=(1, cfg.stride_env), bias=False)
        self.normalization_env = nn.LayerNorm(cfg.num_tf * cfg.num_env)
        self.specaug_config = cfg.specaug_config
        self.specaug_start_epoch = cfg.specaug_start_epoch

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

        from returnn.torch.context import get_run_ctx
        run_ctx = get_run_ctx()

        if self.training and run_ctx.epoch >= self.specaug_start_epoch:
            filter_idcs_tf = self.get_sorted_filter_indices()
            num_env = self.conv_env.out_channels
            filter_idcs_env = torch.stack([filter_idcs_tf * num_env + filter_idx for filter_idx in range(num_env)])
            feature_data_masked = specaugment_v1_by_length(
                feature_data,
                time_min_num_masks=2,
                time_max_mask_per_n_frames=self.specaug_config.repeat_per_n_frames,
                time_mask_max_size=self.specaug_config.max_dim_time,
                freq_min_num_masks=2,
                freq_mask_max_size=self.specaug_config.max_dim_feat,
                freq_max_num_masks=self.specaug_config.num_repeat_feat,
                sorted_indices=filter_idcs_env.T.flatten(),
            )
        else:
            feature_data_masked = feature_data

        length = ((length - self.conv_tf.kernel_size[-1]) / self.conv_tf.stride[-1] + 1).int()
        length = ((length - self.conv_env.kernel_size[-1]) / self.conv_env.stride[-1] + 1).int()

        return feature_data_masked, length

    def get_sorted_filter_indices(self) -> torch.Tensor:
        filters = self.conv_tf.weight.squeeze()  # (C, N)
        num_freqs = 128  # F
        w = torch.linspace(0.0, torch.pi - torch.pi / num_freqs, num_freqs).to(filters)  # (F,)
        zm1 = torch.exp(-1j * w).unsqueeze(1)  # (F, 1)
        exponents = torch.arange(filters.shape[1]).unsqueeze(0).to(filters)  # (1, N)
        zm1_pow = torch.pow(zm1, exponents)  # (F, N)
        f_resp = torch.tensordot(zm1_pow, filters.t().to(zm1_pow), dims=1)  # (F, C)
        f_resp = f_resp.abs()
        # # move to log domain, not needed for center frequencies
        # f_resp = 20 * torch.log10(f_resp)

        # sorted by increasing center frequency
        center_freqs = torch.argmax(f_resp, dim=0)
        sorted_idcs = torch.argsort(center_freqs)
        return sorted_idcs
