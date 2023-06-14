from dataclasses import dataclass
from typing import List, Optional

import torch
import torchaudio

from i6_models.config import ModelConfiguration


@dataclass
class SpecaugmentConfigV1(ModelConfiguration):
    max_time_mask_num: int
    max_time_mask_size: int
    max_feature_mask_num: int
    max_feature_mask_size: int
    increase_steps: Optional[List[int]] = None


class SpecaugmentModuleV1(torch.nn.Module):
    def __init__(self, step: int, cfg: SpecaugmentConfigV1):
        super().__init__()
        self.max_time_num = cfg.max_time_mask_num
        self.max_feature_num = cfg.max_feature_mask_size

        self.max_time = cfg.max_time_mask_size
        self.max_feature = cfg.max_feature_mask_size

        if cfg.increase_steps is None:
            self.warmup_factor = 1
        else:
            self.warmup_factor = 1 + len([inc_step for inc_step in cfg.increase_steps if inc_step > step])

    def forward(self, audio_features: torch.Tensor):
        """
        :param audio_features: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T, F]
        """
        if not self.training:
            return audio_features

        x = audio_features

        max_time_num = (
            max(
                self.max_time_num,
                int(audio_features.shape[1]) // int(1.0 / 0.7 * self.max_time),
            )
            // self.warmup_factor
        )
        random_time_num = int(torch.randint(low=0, high=max_time_num, size=(1,)).item())

        for _ in range(random_time_num):
            x = torchaudio.functional.mask_along_axis(x, mask_param=self.max_time, mask_value=0.0, axis=1)

        max_feature_num = self.max_feature_num // self.warmup_factor
        random_feature_num = int(torch.randint(low=0, high=max_feature_num, size=(1,)).item())

        for _ in range(random_feature_num):
            x = torchaudio.functional.mask_along_axis(x, mask_param=self.max_feature, mask_value=0.0, axis=2)

        return x
