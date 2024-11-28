from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from i6_models.assemblies.conformer import (
    ConformerEncoderV2,
    ConformerEncoderV2Config,
)
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length
from sisyphus import tk


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


@dataclass
class SpecaugmentByLengthConfig(ModelConfiguration):
    start_epoch: int
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    logmel_cfg: RasrCompatibleLogMelFeatureExtractionV1Config
    conformer_cfg: ConformerEncoderV2Config
    dim: int
    target_size: int
    dropout: float
    specaug_cfg: SpecaugmentByLengthConfig


@dataclass
class ConformerCTCRecogConfig(ConformerCTCConfig):
    prior_file: tk.Path
    prior_scale: float
    blank_penalty: float


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, cfg: ConformerCTCConfig, epoch: int, **_):
        super().__init__()
        self.epoch = epoch
        self.feature_extraction = RasrCompatibleLogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.conformer = ConformerEncoderV2(cfg.conformer_cfg)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.target_size = cfg.target_size
        self.final_linear = torch.nn.Linear(cfg.dim, self.target_size)
        self.specaug_config = cfg.specaug_cfg

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            audio_samples = audio_samples.squeeze(-1)  # [B, T]
            features, features_size = self.feature_extraction.forward(
                audio_samples, audio_samples_size
            )  # [B, T, F], [B]
            sequence_mask = lengths_to_padding_mask(features_size)  # [B, T]

            if self.training and self.epoch >= self.specaug_config.start_epoch:
                features = specaugment_v1_by_length(
                    audio_features=features,
                    time_min_num_masks=self.specaug_config.time_min_num_masks,
                    time_max_mask_per_n_frames=self.specaug_config.time_max_mask_per_n_frames,
                    time_mask_max_size=self.specaug_config.time_mask_max_size,
                    freq_min_num_masks=self.specaug_config.freq_min_num_masks,
                    freq_max_num_masks=self.specaug_config.freq_max_num_masks,
                    freq_mask_max_size=self.specaug_config.freq_mask_max_size,
                )  # [B, T, F]

        encoder_states, sequence_mask = self.conformer.forward(features, sequence_mask)  # [B, T, F], [B, T]
        encoder_states = encoder_states[0]

        encoder_states = self.dropout(encoder_states)  # [B, T, F]
        logits = self.final_linear(encoder_states)  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, V]

        return log_probs, torch.sum(sequence_mask, dim=1).type(torch.int32)


class ConformerCTCRecogModel(ConformerCTCModel):
    def __init__(self, cfg: ConformerCTCRecogConfig, epoch: int, **_):
        super().__init__(
            cfg=cfg,
            epoch=epoch,
        )
        self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        log_probs, _ = super().forward(audio_samples=audio_samples, audio_samples_size=audio_samples_size)  # [B, T, V]
        scores = -log_probs  # [B, T, V]
        scores[:, :, -1] += self.blank_penalty  # [B, T, V]
        return scores + self.scaled_priors.to(device=log_probs.device)  # [B, T, V]
