__all__ = ["NoConfig", "PassSamplesModel", "SpecaugmentByLengthConfig", "lengths_to_padding_mask"]

from dataclasses import dataclass
from typing import Tuple

import torch
from i6_models.config import ModelConfiguration


@dataclass
class NoConfig(ModelConfiguration):
    pass


class PassSamplesModel(torch.nn.Module):
    def __init__(self, cfg: NoConfig, **_):
        super().__init__()

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            audio_samples = audio_samples.squeeze(-1)  # [B, T]

            return audio_samples, audio_samples_size


@dataclass
class SpecaugmentByLengthConfig(ModelConfiguration):
    start_epoch: int
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to an equivalent boolean mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    max_length = torch.max(lengths)
    index_range = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)  # type: ignore
    sequence_mask = torch.less(index_range[None, :], lengths[:, None])

    return sequence_mask
