__all__ = [
    "FFNNTransducerConfig",
    "FFNNTransducerRecogConfig",
    "FFNNTransducerModel",
    "FFNNTransducerEncoder",
    "FFNNTransducerScorer",
]

from dataclasses import dataclass
from typing import Tuple

import torch
from i6_models.assemblies.conformer import ConformerRelPosEncoderV1, ConformerRelPosEncoderV1Config
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length

from ..common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask


@dataclass
class FFNNTransducerConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugmentByLengthConfig
    conformer_cfg: ConformerRelPosEncoderV1Config
    enc_dim: int
    pred_num_layers: int
    pred_dim: int
    pred_activation: torch.nn.Module
    dropout: float
    context_history_size: int
    context_embedding_dim: int
    joiner_dim: int
    joiner_activation: torch.nn.Module
    target_size: int


@dataclass
class FFNNTransducerRecogConfig(FFNNTransducerConfig):
    ilm_scale: float
    blank_penalty: float


class FFNNTransducerModel(torch.nn.Module):
    def __init__(self, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.target_size = cfg.target_size

        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerRelPosEncoderV1(cfg.conformer_cfg)

        self.encoder_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.enc_dim, self.target_size),
        )

        self.context_history_size = cfg.context_history_size
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=self.target_size, embedding_dim=cfg.context_embedding_dim, padding_idx=cfg.target_size - 1
        )

        prediction_layers = []
        prev_size = self.context_history_size * cfg.context_embedding_dim
        for _ in range(cfg.pred_num_layers):
            prediction_layers.append(torch.nn.Dropout(cfg.dropout))
            prediction_layers.append(torch.nn.Linear(prev_size, cfg.pred_dim))
            prediction_layers.append(cfg.pred_activation)
            prev_size = cfg.pred_dim
        self.prediction_net = torch.nn.Sequential(*prediction_layers)

        self.prediction_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout), torch.nn.Linear(cfg.pred_dim, self.target_size)
        )

        self.joint_net = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.enc_dim + cfg.pred_dim, cfg.joiner_dim),
            cfg.joiner_activation,
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.joiner_dim, self.target_size),
        )

    def forward_encoder(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor,  # final encoder logits [B, T, V]
        torch.Tensor,  # encoder lengths  [B]
    ]:
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

        encoder_states, sequence_mask = self.conformer.forward(features, sequence_mask)  # [B, T, E], [B, T]
        encoder_states = encoder_states[-1]

        encoder_states_size = torch.sum(sequence_mask, dim=1).type(torch.int32)

        return encoder_states, encoder_states_size

    def forward_prediction_network(
        self,
        targets: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:  # Final prediction network logits [B, S+1, V]
        extended_targets = torch.nn.functional.pad(targets, [self.context_history_size, 0], value=self.target_size - 1)

        # Build context at each position by shifting and cutting label sequence.
        # E.g. for history size 2 and extended targets 0, 0, a_1, ..., a_S we have context
        # 0, a_1, a_2 a_3 a_4 ... a_S
        # 0,   0, a_1 a_2 a_3 ... a_{S-1}
        context = torch.stack(
            [
                extended_targets[:, self.context_history_size - 1 - i : (-i if i != 0 else None)]  # [B, S+1]
                for i in reversed(range(self.context_history_size))
            ],
            dim=-1,
        )  # [B, S+1, H]

        embedding = self.token_embedding.forward(context)  # [B, S+1, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, S+1, H*A]
        pred_states = self.prediction_net.forward(embedding)  # [B, S+1, P]

        return pred_states

    def forward_joint_network(
        self,
        encoder_states: torch.Tensor,  # [B, T, E]
        encoder_states_size: torch.Tensor,  # [B]
        pred_states: torch.Tensor,  # [B, S+1, P]
        targets_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # final logits [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C]
        encoder_states = encoder_states.to(dtype=torch.float32)
        pred_states = pred_states.to(dtype=torch.float32)
        batch_tensors = []
        for b in range(encoder_states.size(0)):
            valid_enc = encoder_states[b, : encoder_states_size[b], :]  # [T_b, E]
            valid_pred = pred_states[b, : targets_size[b] + 1, :]  # [S_b+1, P]

            expanded_enc = valid_enc.unsqueeze(1).expand(-1, int(targets_size[b].item()) + 1, -1)  # [T_b, S_b+1, E]
            expanded_pred = valid_pred.unsqueeze(0).expand(
                int(encoder_states_size[b].item()), -1, -1
            )  # [T_b, S_b+1, P]

            combination = torch.concat([expanded_enc, expanded_pred], dim=-1)  # [T_b, S_b+1, E+P]

            batch_tensors.append(combination.reshape(-1, combination.size(2)))  # [T_b * (S_b+1), E+P]

        joint_input = torch.concat(batch_tensors, dim=0)  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), E+P]
        joint_output = self.joint_net.forward(joint_input)  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), V]

        return joint_output


class FFNNTransducerEncoder(FFNNTransducerModel):
    def __init__(self, cfg: FFNNTransducerConfig, **_):
        super().__init__(cfg=cfg)
        self.enc_output_indices = []

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, V]
        encoder_states, _ = self.forward_encoder(audio_samples, audio_samples_size)  # [B, T, E]
        return encoder_states  # [B, T, E]


class FFNNTransducerScorer(FFNNTransducerModel):
    def __init__(self, cfg: FFNNTransducerRecogConfig, **_):
        super().__init__(cfg=cfg)
        self.ilm_scale = cfg.ilm_scale
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        encoder_state: torch.Tensor,  # [1, E]
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, V]
        embedding = self.token_embedding.forward(history)  # [B, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, H*A]
        pred_state = self.prediction_net.forward(embedding)  # [B, P]

        joint_input = torch.concat([encoder_state.expand([pred_state.size(0), -1]), pred_state], dim=-1)  # [B, E+P]
        joint_output = self.joint_net.forward(joint_input)  # [B, V]
        scores = -torch.nn.functional.log_softmax(joint_output, dim=1)  # [B, V]

        scores[:, -1] += self.blank_penalty

        if self.ilm_scale != 0:
            zero_enc = torch.zeros_like(encoder_state)  # [B, E]
            ilm_joint_input = torch.concat([zero_enc.expand([pred_state.size(0), -1]), pred_state], dim=-1)  # [B, E+P]
            ilm_joint_output = self.joint_net.forward(ilm_joint_input)  # [B, V]
            ilm_log_probs = torch.nn.functional.log_softmax(ilm_joint_output, dim=1)  # [B, V]

            # Set blank scores to zero and re-normalize the other scores
            blank_log_probs = ilm_log_probs[:, -1:]  # [B, 1]
            non_blank_log_probs = ilm_log_probs[:, :-1]  # [B, V-1]
            ilm_log_probs = torch.concat(
                [
                    non_blank_log_probs - torch.log1p(-torch.exp(blank_log_probs)),
                    torch.zeros_like(blank_log_probs),
                ],
                dim=-1,
            )  # [B, V]

            ilm_scores = -ilm_log_probs

            scores -= self.ilm_scale * ilm_scores

        return scores
