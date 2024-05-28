from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from torch import nn

from i6_models.config import ModelConfiguration
from .specaugment_sorted_v1 import specaugment_v1_by_length


@dataclass
class SpecaugConfig(ModelConfiguration):
    start_epoch: int
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int
    freq_mask_max_size_delta_per_epoch: float


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
class SupervisedConvolutionalFeatureExtractionV4Config(FeatureExtractionConfig):
    """
    Attributes:
        wave_norm: normalize waveform over time dim, then learnable scale and biases
        num_tf: number of filters in first conv layer (for time-frequency decomposition)
        size_tf: filter size in first conv layer
        stride_tf: stride in first conv layer
        init_tf: initialization for first conv layer, e.g. None or "gammatone"
        num_env: number of filters in second conv layer (for envelope extraction)
        size_env: filter size in second conv layer
        stride_env: stride in second conv layer
        init_env: initialization for second conv layer, e.g. None or "hann"
        interleaved_resolutions: whether resolutions should be interleaved or stacked (typically interleaved)
        convs: size, channels and groups for subsequent convolutions used to reduce feature dimension
        init_convs: initialization for pooling conv layers, e.g. None or "ones"
        specaug_config: SpecAugment config
        specaug_before_conv_red: whether to apply SpecAugment before convolutions for reduced feature dimension
    """

    wave_norm: bool
    num_tf: int
    size_tf: int
    stride_tf: int
    init_tf: Optional[str]
    num_env: int
    size_env: int
    stride_env: int
    init_env: Optional[str]
    interleaved_resolutions: bool
    convs: List[Tuple[int, int, int]]  # [(size, channels, groups)]
    init_convs: Optional[str]
    specaug_config: SpecaugConfig
    specaug_before_conv_red: bool

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


class SupervisedConvolutionalFeatureExtractionV4(nn.Module):
    """
    Like V2, but this includes SpecAugment where the masks are based on sorted filters.
    """

    def __init__(self, cfg: SupervisedConvolutionalFeatureExtractionV4Config):
        super().__init__()
        self.wave_norm = nn.Conv1d(1, 1, 1) if cfg.wave_norm else None
        self.conv_tf = nn.Conv1d(1, cfg.num_tf, cfg.size_tf, stride=cfg.stride_tf, bias=False)
        self.conv_env = nn.Conv2d(1, cfg.num_env, (1, cfg.size_env), stride=(1, cfg.stride_env), bias=False)
        self.normalization_env = nn.LayerNorm(cfg.num_tf * cfg.num_env)
        self.convs_reduction = []
        dim_in = cfg.num_tf * cfg.num_env
        for size, channels, groups in cfg.convs:
            self.convs_reduction.append(nn.Conv1d(dim_in, channels, size, groups=groups, bias=False))
            dim_in = channels
        self.cfg = cfg

        with torch.no_grad():
            if cfg.init_tf == "gammatone":
                from .gammatone import GammatoneFilterbank
                gt_fbank = GammatoneFilterbank(cfg.num_tf, cfg.size_tf / 16000)
                gt_fbank_tensor = torch.from_numpy(gt_fbank.get_gammatone_filterbank()).float()  # [T, C]
                gt_fbank_tensor = gt_fbank_tensor.transpose(0, 1).unsqueeze(1)  # [C, 1, T]
            elif cfg.init_tf is not None:
                raise NotImplementedError(f"Unknown initialization: {cfg.init_tf}")
            if cfg.init_env == "hann":
                self.conv_tf.weight = nn.Parameter(gt_fbank_tensor)
                hann_win = torch.hann_window(cfg.size_env, periodic=False)
                hann_win = hann_win.repeat(cfg.num_env, 1)[:, None, None, :]
                self.conv_env.weight = nn.Parameter(hann_win)
            elif cfg.init_env is not None:
                raise NotImplementedError(f"Unknown initialization: {cfg.init_env}")
            if cfg.init_convs == "ones":
                for conv in self.convs_reduction:
                    conv.weight = nn.Parameter(torch.ones(conv.weight.shape))
            elif cfg.init_convs is not None:
                raise NotImplementedError(f"Unknown initialization: {cfg.init_convs}")

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
        if self.cfg.interleaved_resolutions:
            feature_data = torch.flatten(feature_data.transpose(1, 2), 1, 2)  # [B,F,T'']
        else:  # stacked resolutions
            feature_data = torch.flatten(feature_data, 1, 2)  # [B,F,T'']
        feature_data = torch.pow(feature_data.abs() + 1e-5, 1 / 2.5)
        feature_data = feature_data.transpose(1, 2)  # [B,T'',F]
        feature_data = self.normalization_env(feature_data)

        if not self.cfg.specaug_before_conv_red:
            feature_data = feature_data.transpose(1, 2)  # [B,F,T'']
            for conv in self.convs_reduction:
                conv.to(feature_data.device)
                feature_data = conv(feature_data)
            feature_data = feature_data.transpose(1, 2)  # [B,T'',F']

        from returnn.torch.context import get_run_ctx
        run_ctx = get_run_ctx()

        if self.training and run_ctx.epoch >= self.cfg.specaug_config.start_epoch:
            filter_idcs_tf = self.get_sorted_filter_indices()
            num_env = self.conv_env.out_channels
            num_tf = self.conv_tf.out_channels
            if self.cfg.interleaved_resolutions:
                filter_idcs_env = torch.stack([filter_idcs_tf * num_env + filter_idx for filter_idx in range(num_env)])
                filter_idcs_env = filter_idcs_env.T.flatten()
            else:
                filter_idcs_env = torch.stack([filter_idcs_tf + num_tf * filter_idx for filter_idx in range(num_env)])
                filter_idcs_env = filter_idcs_env.T.flatten()
            freq_mask_max_size = int(
                self.cfg.specaug_config.freq_mask_max_size +
                self.cfg.specaug_config.freq_mask_max_size_delta_per_epoch * run_ctx.epoch
            )
            feature_data_masked = specaugment_v1_by_length(
                feature_data,
                time_min_num_masks=self.cfg.specaug_config.time_min_num_masks,
                time_max_mask_per_n_frames=self.cfg.specaug_config.time_max_mask_per_n_frames,
                time_mask_max_size=self.cfg.specaug_config.time_mask_max_size,
                freq_min_num_masks=self.cfg.specaug_config.freq_min_num_masks,
                freq_mask_max_size=freq_mask_max_size,
                freq_max_num_masks=self.cfg.specaug_config.freq_max_num_masks,
                sorted_indices=filter_idcs_env,
            )
        else:
            feature_data_masked = feature_data

        if self.cfg.specaug_before_conv_red:
            feature_data_masked = feature_data_masked.transpose(1, 2)  # [B,F,T'']
            for conv in self.convs_reduction:
                conv.to(feature_data_masked.device)
                feature_data_masked = conv(feature_data_masked)
            feature_data_masked = feature_data_masked.transpose(1, 2)  # [B,T'',F']

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
