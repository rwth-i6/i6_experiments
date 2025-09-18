from dataclasses import dataclass
from sisyphus.tools import functools

import torch

from i6_models.config import ModelConfiguration
from i6_models.primitives.specaugment import specaugment_v1, specaugment_v1_by_length


@dataclass
class SpecaugmentConfigV1(ModelConfiguration):
    time_min_num_masks: int
    time_max_num_masks: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


@dataclass
class SpecaugmentByLengthConfigV1(ModelConfiguration):
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


class SpecaugmentModuleV1(torch.nn.Module):
    def __init__(self, cfg: SpecaugmentConfigV1):
        super().__init__()
        self.specaug_func = functools.partial(
            specaugment_v1,
            time_min_num_masks=cfg.time_min_num_masks,
            time_max_num_masks=cfg.time_max_num_masks,
            time_mask_max_size=cfg.time_mask_max_size,
            freq_min_num_masks=cfg.freq_min_num_masks,
            freq_max_num_masks=cfg.freq_max_num_masks,
            freq_mask_max_size=cfg.freq_mask_max_size,
        )

    def forward(self, audio_features: torch.Tensor):
        """
        :param audio_features: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T, F]
        """
        if not self.training:
            return audio_features

        return self.specaug_func(audio_features)


class SpecaugmentByLengthModuleV1(torch.nn.Module):
    def __init__(self, cfg: SpecaugmentByLengthConfigV1):
        super().__init__()
        self.specaug_func = functools.partial(
            specaugment_v1_by_length,
            time_min_num_masks=cfg.time_min_num_masks,
            time_max_mask_per_n_frames=cfg.time_max_mask_per_n_frames,
            time_mask_max_size=cfg.time_mask_max_size,
            freq_min_num_masks=cfg.freq_min_num_masks,
            freq_max_num_masks=cfg.freq_max_num_masks,
            freq_mask_max_size=cfg.freq_mask_max_size,
        )

    def forward(self, audio_features: torch.Tensor):
        """
        :param audio_features: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T, F]
        """
        if not self.training:
            return audio_features

        return self.specaug_func(audio_features)
