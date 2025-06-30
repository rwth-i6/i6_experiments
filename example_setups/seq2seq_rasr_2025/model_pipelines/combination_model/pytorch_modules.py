__all__ = [
    "CombinationModelConfig",
    "CombinationModel",
    "CombinationModelEncoder",
    "CombinationModelCTCScorer",
    "CombinationModelAttentionScorer",
    "CombinationModelTransducerScorer",
    "CombinationModelAttentionStateInitializer",
    "CombinationModelAttentionStateUpdater",
]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1, ConformerRelPosEncoderV1Config
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from sisyphus import tk

from ..bpe_aed.pytorch_modules import AttentionLSTMDecoderV1, AttentionLSTMDecoderV1Config
from ..common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask


@dataclass
class CombinationModelConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugmentByLengthConfig
    conformer_cfg: ConformerRelPosEncoderV1Config
    attention_decoder_config: AttentionLSTMDecoderV1Config
    transducer_pred_num_layers: int
    transducer_pred_dim: int
    transducer_pred_activation: torch.nn.Module
    transducer_context_history_size: int
    transducer_context_embedding_dim: int
    transducer_joiner_dim: int
    transducer_joiner_activation: torch.nn.Module
    transducer_decoder_dropout: float
    enc_dim: int
    ctc_dropout: float
    target_size: int


@dataclass
class CombinationModelCTCRecogConfig(CombinationModelConfig):
    prior_file: tk.Path
    prior_scale: float
    blank_penalty: float


@dataclass
class CombinationModelTransducerRecogConfig(CombinationModelConfig):
    ilm_scale: float
    blank_penalty: float


class CombinationModel(torch.nn.Module):
    def __init__(self, cfg: CombinationModelConfig, **net_kwargs):
        super().__init__()

        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.encoder = ConformerRelPosEncoderV1(cfg.conformer_cfg)
        self.enc_dim = cfg.enc_dim

        self.target_size = cfg.target_size

        self.ctc_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.ctc_dropout), torch.nn.Linear(cfg.enc_dim, self.target_size)
        )

        self.attention_decoder = AttentionLSTMDecoderV1(cfg=cfg.attention_decoder_config)

        self.transducer_context_history_size = cfg.transducer_context_history_size
        self.transducer_token_embedding = torch.nn.Embedding(
            num_embeddings=self.target_size,
            embedding_dim=cfg.transducer_context_embedding_dim,
            padding_idx=0,
        )

        prediction_layers = []
        prev_size = self.transducer_context_history_size * cfg.transducer_context_embedding_dim
        for _ in range(cfg.transducer_pred_num_layers):
            prediction_layers.append(torch.nn.Dropout(cfg.transducer_decoder_dropout))
            prediction_layers.append(torch.nn.Linear(prev_size, cfg.transducer_pred_dim))
            prediction_layers.append(cfg.transducer_pred_activation)
            prev_size = cfg.transducer_pred_dim

        self.transducer_pred_ffnn = torch.nn.Sequential(*prediction_layers)

        self.transducer_joiner = torch.nn.Sequential(
            torch.nn.Dropout(cfg.transducer_decoder_dropout),
            torch.nn.Linear(cfg.enc_dim + cfg.transducer_pred_dim, cfg.transducer_joiner_dim),
            cfg.transducer_joiner_activation,
            torch.nn.Dropout(cfg.transducer_decoder_dropout),
            torch.nn.Linear(cfg.transducer_joiner_dim, self.target_size),
        )

    def forward_encoder(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor,  # final encoder states [B, T, E]
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

        encoder_states, sequence_mask = self.encoder.forward(features, sequence_mask)  # [B, T, E], [B, T]
        encoder_states = encoder_states[-1]

        encoder_states_size = torch.sum(sequence_mask, dim=1).type(torch.int32)

        return encoder_states, encoder_states_size

    def forward_ctc(self, encoder_states: torch.Tensor) -> torch.Tensor:
        ctc_logits = self.ctc_output.forward(encoder_states)  # [B, T, V]
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)

        return ctc_log_probs

    def forward_attention(
        self, encoder_states: torch.Tensor, encoder_states_size: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        attention_logits = self.attention_decoder.forward(
            encoder_states, targets, encoder_states_size.to(device=encoder_states.device)
        )

        return attention_logits

    def forward_transducer_pred(self, targets: torch.Tensor) -> torch.Tensor:
        extended_targets = torch.nn.functional.pad(
            targets, [self.transducer_context_history_size, 0], value=0
        )  # Also remove EOS token

        # Build context at each position by shifting and cutting label sequence.
        # E.g. for history size 2 and extended targets 0, 0, a_1, ..., a_S we have context
        # 0, a_1, a_2 a_3 a_4 ... a_S
        # 0,   0, a_1 a_2 a_3 ... a_{S-1}
        context = torch.stack(
            [
                extended_targets[:, self.transducer_context_history_size - 1 - i : (-i if i != 0 else None)]  # [B, S+1]
                for i in range(self.transducer_context_history_size)
            ],
            dim=-1,
        )  # [B, S+1, H]

        embedding = self.transducer_token_embedding.forward(context)  # [B, S+1, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, S+1, H*A]
        pred_states = self.transducer_pred_ffnn.forward(embedding)  # [B, S+1, P]

        return pred_states

    def forward_transducer_join(
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
        joint_output = self.transducer_joiner.forward(
            joint_input
        )  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), V]

        return joint_output

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B, S]
        targets_size: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        encoder_states, encoder_states_size = self.forward_encoder(audio_samples, audio_samples_size)

        ctc_log_probs = self.forward_ctc(encoder_states)

        attention_logits = self.forward_attention(encoder_states, encoder_states_size, targets)

        pred_states = self.forward_transducer_pred(targets)
        transducer_logits = self.forward_transducer_join(encoder_states, encoder_states_size, pred_states, targets_size)

        return transducer_logits, attention_logits, ctc_log_probs, encoder_states_size


class CombinationModelEncoder(CombinationModel):
    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T', F]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, E+A+1]
        encoder_states, _ = self.forward_encoder(audio_samples, audio_samples_size)
        enc_ctx = self.attention_decoder.enc_ctx.forward(encoder_states)  # [B, T, A]
        enc_inv_fertility = torch.nn.functional.sigmoid(
            self.attention_decoder.inv_fertility.forward(encoder_states)
        )  # [B,T,1]
        return torch.cat([encoder_states, enc_ctx, enc_inv_fertility], dim=-1)  # [B, T, E+A+1]


class CombinationCTCPriorModel(CombinationModel):
    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_states, encoder_states_size = self.forward_encoder(audio_samples, audio_samples_size)

        ctc_log_probs = self.forward_ctc(encoder_states)

        return ctc_log_probs, encoder_states_size


class CombinationModelCTCScorer(CombinationModel):
    def __init__(self, cfg: CombinationModelCTCRecogConfig, epoch: int, **_):
        super().__init__(
            cfg=cfg,
            epoch=epoch,
        )
        self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        encoder_state: torch.Tensor,  # [1, E]
    ) -> torch.Tensor:
        encoder_state, _ = torch.split(
            encoder_state,
            [
                self.enc_dim,
                self.attention_decoder.attention_dim + 1,
            ],
            dim=-1,
        )  # [1, E], [1, A+1]

        log_probs = self.forward_ctc(encoder_state)
        scores = -log_probs  # [1, V]
        scores[:, -1] += self.blank_penalty  # [1, V]
        return scores + self.scaled_priors.to(device=log_probs.device)  # [1, V]


class CombinationModelTransducerScorer(CombinationModel):
    def __init__(self, cfg: CombinationModelTransducerRecogConfig, epoch: int, **_):
        super().__init__(cfg=cfg)
        self.ilm_scale = cfg.ilm_scale
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        encoder_state: torch.Tensor,  # [1, E+A+1]
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, V]
        encoder_state, _ = torch.split(
            encoder_state,
            [
                self.enc_dim,
                self.attention_decoder.attention_dim + 1,
            ],
            dim=-1,
        )  # [1, E], [1, A+1]

        encoder_states = encoder_state.expand([history.size(0), encoder_state.size(1)])  # [B, E]

        embedding = self.transducer_token_embedding.forward(history)  # [B, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, H*A]
        pred_states = self.transducer_pred_ffnn.forward(embedding)  # [B, P]

        combination = torch.concat([encoder_states, pred_states], dim=1)  # [B, E+P]
        joint_output = self.transducer_joiner.forward(combination)  # [B, V]
        scores = -torch.nn.functional.log_softmax(joint_output, dim=1)  # [B, V]

        scores[:, -1] += self.blank_penalty

        zero_enc_combination = torch.concat([torch.zeros_like(encoder_states), pred_states], dim=1)  # [B, E+P]
        joint_output_ilm = self.transducer_joiner.forward(zero_enc_combination)  # [B, V]
        ilm_log_probs = torch.nn.functional.log_softmax(joint_output_ilm, dim=1)  # [B, V]

        # Set blank scores to zero and re-normalize the other scores
        blank_log_probs = ilm_log_probs[:, -1:]  # [B, 1]
        non_blank_log_probs = ilm_log_probs[:, :-1]  # [B, V-1]
        ilm_log_probs = torch.concat(
            [
                non_blank_log_probs + torch.log(1.0 - torch.exp(blank_log_probs)),
                torch.zeros_like(blank_log_probs),
            ],
            dim=-1,
        )  # [B, V]

        ilm_scores = -ilm_log_probs

        return scores + self.ilm_scale * ilm_scores


class CombinationModelAttentionScorer(CombinationModel):
    def forward(
        self,
        token_embedding: torch.Tensor,  # [B, M]
        lstm_state_h: torch.Tensor,  # [B, H]
        att_context: torch.Tensor,  # [B, E]
    ) -> torch.Tensor:
        readout_in = self.attention_decoder.readout_in.forward(
            torch.cat([lstm_state_h, token_embedding, att_context], dim=1)
        )  # [B, D]

        # maxout layer
        readout_in = readout_in.view(readout_in.size(0), -1, 2)  # [B, D/2, 2]
        readout, _ = torch.max(readout_in, dim=2)  # [B, D/2]

        decoder_logits = self.attention_decoder.output.forward(readout)  # [B, V]
        scores = -torch.log_softmax(decoder_logits, dim=1)  # [B, V]

        return scores


class CombinationModelAttentionStateInitializer(CombinationModel):
    def forward(
        self,
        encoder_states: torch.Tensor,  # [1, T, E + A]
        encoder_states_size: torch.Tensor,  # [1]
    ) -> Tuple[torch.Tensor, ...]:
        encoder_out, enc_ctx, enc_inv_fertility = torch.split(
            encoder_states,
            [
                self.enc_dim,
                self.attention_decoder.attention_dim,
                1,
            ],
            dim=2,
        )  # [1, T, E], [1, T, A], [1, T, 1]

        lstm_state_h = encoder_out.new_zeros(1, self.attention_decoder.lstm_hidden_size)  # [1, H]
        lstm_state_c = encoder_out.new_zeros(1, self.attention_decoder.lstm_hidden_size)  # [1, H]
        att_context = encoder_out.new_zeros(1, encoder_out.size(2))  # [1, E]
        accum_att_weights = encoder_out.new_zeros((1, encoder_out.size(1), 1))  # [1, T, 1]

        token_embedding = encoder_out.new_zeros(1, self.attention_decoder.target_embed_dim)  # [1, M]

        new_lstm_state_h, new_lstm_state_c = self.attention_decoder.s.forward(
            torch.cat([token_embedding, att_context], dim=1),
            (lstm_state_h, lstm_state_c),
        )  # [1, H], [1, H]
        s_transformed = self.attention_decoder.s_transformed.forward(new_lstm_state_h)  # [1, A]

        weight_feedback = self.attention_decoder.weight_feedback.forward(accum_att_weights)  # [1, T, A]

        new_att_context, new_att_weights = self.attention_decoder.attention(
            key=enc_ctx,
            value=encoder_out,
            query=s_transformed,
            weight_feedback=weight_feedback,
            enc_seq_len=encoder_states_size,
        )  # [1, E], [1, T, 1]

        new_accum_att_weights = accum_att_weights + new_att_weights * enc_inv_fertility * 0.5  # [1, T, 1]

        return token_embedding, new_lstm_state_h, new_lstm_state_c, new_att_context, new_accum_att_weights


class CombinationModelAttentionStateUpdater(CombinationModel):
    def forward(
        self,
        encoder_states: torch.Tensor,  # [T, E + A]
        encoder_states_size: torch.Tensor,  # []
        token: torch.Tensor,  # [1j
        lstm_state_h: torch.Tensor,  # [1, H]
        lstm_state_c: torch.Tensor,  # [1, H]
        att_context: torch.Tensor,  # [1, E]
        accum_att_weights: torch.Tensor,  # [1, T, 1]
    ) -> Tuple[torch.Tensor, ...]:
        encoder_out, enc_ctx, enc_inv_fertility = torch.split(
            encoder_states,
            [
                self.enc_dim,
                self.attention_decoder.attention_dim,
                1,
            ],
            dim=2,
        )  # [1, T, E], [1, T, A], [1, T, 1]

        new_token_embedding = self.attention_decoder.target_embed.forward(token)  # [1, M]

        new_lstm_state_h, new_lstm_state_c = self.attention_decoder.s.forward(
            torch.cat([new_token_embedding, att_context], dim=1),
            (lstm_state_h, lstm_state_c),
        )  # [1, H], [1, H]
        s_transformed = self.attention_decoder.s_transformed.forward(new_lstm_state_h)  # [1, A]

        weight_feedback = self.attention_decoder.weight_feedback.forward(accum_att_weights)  # [1, T, A]

        new_att_context, new_att_weights = self.attention_decoder.attention(
            key=enc_ctx,
            value=encoder_out,
            query=s_transformed,
            weight_feedback=weight_feedback,
            enc_seq_len=encoder_states_size,
        )  # [1, E], [1, T, 1]

        new_accum_att_weights = accum_att_weights + new_att_weights * enc_inv_fertility * 0.5  # [T, 1]

        return new_token_embedding, new_lstm_state_h, new_lstm_state_c, new_att_context, new_accum_att_weights
