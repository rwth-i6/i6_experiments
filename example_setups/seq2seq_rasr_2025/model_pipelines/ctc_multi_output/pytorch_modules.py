__all__ = [
    "ConformerCTCMultiOutputConfig",
    "ConformerCTCMultiOutputPriorConfig",
    "ConformerCTCMultiOutputScorerConfig",
    "ConformerCTCMultiOutputModel",
    "ConformerCTCMultiOutputPriorModel",
    "ConformerCTCMultiOutputScorerModel",
]

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from i6_models.assemblies.conformer import (
    ConformerRelPosEncoderV1,
    ConformerRelPosEncoderV1Config,
)
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from sisyphus import tk

from ..common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask


@dataclass
class ConformerCTCMultiOutputConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    conformer_cfg: ConformerRelPosEncoderV1Config
    dim: int
    layer_idx_target_size_list: List[Tuple[int, int]]
    dropout: float
    specaug_cfg: SpecaugmentByLengthConfig


@dataclass
class ConformerCTCMultiOutputPriorConfig(ConformerCTCMultiOutputConfig):
    output_idx: int


@dataclass
class ConformerCTCMultiOutputScorerConfig(ConformerCTCMultiOutputConfig):
    output_idx: int
    prior_file: tk.Path
    prior_scale: float
    blank_penalty: float


class ConformerCTCMultiOutputModel(torch.nn.Module):
    def __init__(self, cfg: ConformerCTCMultiOutputConfig, **_):
        super().__init__()
        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.conformer = ConformerRelPosEncoderV1(cfg.conformer_cfg)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.output_layer_indices = [layer_idx for layer_idx, _ in cfg.layer_idx_target_size_list]
        self.output_layers = torch.nn.ModuleList(
            torch.nn.Linear(cfg.dim, target_size) for _, target_size in cfg.layer_idx_target_size_list
        )
        self.specaug_config = cfg.specaug_cfg
        self.enc_dim = cfg.dim

        # conformer outputs are sorted by layer and deduplicated, so we need a mapping of which conformer output corresponds to which item in `self.output_layer_indices`
        unique_sorted = sorted(set(self.output_layer_indices))
        value_to_index = {value: index for index, value in enumerate(unique_sorted)}
        self.matching_encoder_states_index = [value_to_index[x] for x in self.output_layer_indices]

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:

        with torch.no_grad():
            audio_samples = audio_samples.squeeze(-1)  # [B, T]
            features, features_size = self.feature_extraction.forward(
                audio_samples, audio_samples_size
            )  # [B, T, F], [B]
            sequence_mask = lengths_to_padding_mask(features_size)  # [B, T]

            if self.training:
                from returnn.torch.context import get_run_ctx  # type: ignore

                if get_run_ctx().epoch >= self.specaug_config.start_epoch:
                    features = specaugment_v1_by_length(
                        audio_features=features,
                        time_min_num_masks=self.specaug_config.time_min_num_masks,
                        time_max_mask_per_n_frames=self.specaug_config.time_max_mask_per_n_frames,
                        time_mask_max_size=self.specaug_config.time_mask_max_size,
                        freq_min_num_masks=self.specaug_config.freq_min_num_masks,
                        freq_max_num_masks=self.specaug_config.freq_max_num_masks,
                        freq_mask_max_size=self.specaug_config.freq_mask_max_size,
                    )  # [B, T, F]

        encoder_states, sequence_mask = self.conformer.forward(
            features, sequence_mask, return_layers=self.output_layer_indices
        )  # [B, T, E], [B, T]

        out_log_probs = []

        for output_layer, encoder_states_index in zip(self.output_layers, self.matching_encoder_states_index):
            encoder_layer_output = self.dropout.forward(encoder_states[encoder_states_index])  # [B, T, E]
            logits = output_layer.forward(encoder_layer_output)  # [B, T, V]
            log_probs = torch.log_softmax(logits, dim=2)  # [B, T, V]
            out_log_probs.append(log_probs)

        return out_log_probs, torch.sum(sequence_mask, dim=1).type(torch.int32)


class ConformerCTCMultiOutputPriorModel(ConformerCTCMultiOutputModel):
    def __init__(self, cfg: ConformerCTCMultiOutputPriorConfig, epoch: int, **_):
        super().__init__(cfg=cfg, epoch=epoch)
        self.output_idx = cfg.output_idx

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

        encoder_states, sequence_mask = self.conformer.forward(
            features,
            sequence_mask,
            return_layers=[self.output_idx],
        )  # [B, T, E], [B, T]

        encoder_states = encoder_states[0]  # [B, E]

        logits = self.output_layers[self.output_idx].forward(encoder_states)  # [B, V]
        return torch.log_softmax(logits, dim=-1), torch.sum(sequence_mask, dim=1).type(torch.int32)


class ConformerCTCMultiOutputEncoderModel(ConformerCTCMultiOutputModel):
    def __init__(self, cfg: ConformerCTCMultiOutputConfig, epoch: int, **_):
        super().__init__(cfg=cfg, epoch=epoch)

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, O*E]
        with torch.no_grad():
            audio_samples = audio_samples.squeeze(-1)  # [B, T]
            features, features_size = self.feature_extraction.forward(
                audio_samples, audio_samples_size
            )  # [B, T, F], [B]
            sequence_mask = lengths_to_padding_mask(features_size)  # [B, T]

        encoder_states, sequence_mask = self.conformer.forward(
            features,
            sequence_mask,
            return_layers=self.output_layer_indices,
        )  # [B, T, F], [B, T]

        return torch.concat([encoder_states[idx] for idx in self.matching_encoder_states_index], dim=-1)


class ConformerCTCMultiOutputScorerModel(ConformerCTCMultiOutputModel):
    def __init__(self, cfg: ConformerCTCMultiOutputScorerConfig, epoch: int, **_):
        super().__init__(cfg=cfg, epoch=epoch)
        self.output_idx = cfg.output_idx
        self.encoder_state_index = self.matching_encoder_states_index[cfg.output_idx]

        self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        encoder_state: torch.Tensor,  # [B, O*E]
    ) -> torch.Tensor:
        encoder_state = encoder_state[
            :, self.encoder_state_index * self.enc_dim : (self.encoder_state_index + 1) * self.enc_dim
        ]  # [B, E]

        logits = self.output_layers[self.output_idx].forward(encoder_state)  # [B, V]
        scores = -torch.log_softmax(logits, dim=-1)  # [B, V]

        scores[:, -1] += self.blank_penalty  # [B, V]
        return scores + self.scaled_priors.to(device=scores.device)  # [B, V]
