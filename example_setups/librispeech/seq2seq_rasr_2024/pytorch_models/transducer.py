from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from i6_models.assemblies.conformer import (
    ConformerBlockV2Config,
    ConformerEncoderV2,
    ConformerEncoderV2Config,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length


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
class FFNNTransducerConfig(ModelConfiguration):
    logmel_cfg: RasrCompatibleLogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugmentByLengthConfig
    conformer_cfg: ConformerEncoderV2Config
    enc_dim: int
    enc_output_indices: List[int]
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
    def __init__(self, cfg: FFNNTransducerConfig, epoch: int, **_):
        super().__init__()
        self.epoch = epoch
        self.target_size = cfg.target_size

        self.feature_extraction = RasrCompatibleLogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerEncoderV2(cfg.conformer_cfg)
        self.enc_output_indices = cfg.enc_output_indices
        self.enc_outputs = torch.nn.ModuleDict(
            {
                f"output_{layer_idx}": torch.nn.Sequential(
                    torch.nn.Dropout(cfg.dropout), torch.nn.Linear(cfg.enc_dim, cfg.target_size)
                )
                for layer_idx in self.enc_output_indices
            }
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

        self.pred_ffnn = torch.nn.Sequential(*prediction_layers)

        self.joiner = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.enc_dim + cfg.pred_dim, cfg.joiner_dim),
            cfg.joiner_activation,
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.joiner_dim, self.target_size),
        )

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B, S]
        targets_size: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor,  # final logits [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C]
        Dict[int, torch.Tensor],  # encoder layer log_probs Dict(layer: [B, T, C])
        torch.Tensor,  # encoder lengths  [B]
    ]:
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

        encoder_states, sequence_mask = self.conformer.forward(
            features, sequence_mask, return_layers=self.enc_output_indices + [len(self.conformer.module_list) - 1]
        )  # [B, T, E], [B, T]

        intermediate_encoder_outputs = {
            layer_idx: torch.nn.functional.log_softmax(
                self.enc_outputs[f"output_{layer_idx}"].forward(encoder_states[i]), dim=-1
            )
            for i, layer_idx in enumerate(self.enc_output_indices)
        }  # [B, T, V]

        encoder_states = encoder_states[-1]  # [B, T, F]
        encoder_states_size = torch.sum(sequence_mask, dim=1).type(torch.int32)

        extended_targets = torch.nn.functional.pad(targets, [self.context_history_size, 0], value=self.target_size - 1)

        # Build context at each position by shifting and cutting label sequence.
        # E.g. for history size 2 and extended targets 0, 0, a_1, ..., a_S we have context
        # 0, a_1, a_2 a_3 a_4 ... a_S
        # 0,   0, a_1 a_2 a_3 ... a_{S-1}
        context = torch.stack(
            [
                extended_targets[:, self.context_history_size - 1 - i : (-i if i != 0 else None)]  # [B, S+1]
                for i in range(self.context_history_size)
            ],
            dim=-1,
        )  # [B, S+1, H]

        embedding = self.token_embedding.forward(context)  # [B, S+1, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, S+1, H*A]
        pred_states = self.pred_ffnn.forward(embedding)  # [B, S+1, P]

        with torch.autocast(device_type="cuda", enabled=False):
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

            joint_input = torch.concat(
                batch_tensors, dim=0
            )  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), E+P]
            joint_output = self.joiner.forward(
                joint_input
            )  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), V]

        return joint_output, intermediate_encoder_outputs, encoder_states_size


class FFNNTransducerEncoder(FFNNTransducerModel):
    def __init__(self, cfg: FFNNTransducerConfig, epoch: int, **_):
        super().__init__(cfg=cfg, epoch=epoch)

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, E]
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

        encoder_states, _ = self.conformer.forward(features, sequence_mask)  # [B, T, E], [B, T]
        return encoder_states[0]  # [B, T, E]


class FFNNTransducerScorer(FFNNTransducerModel):
    def __init__(self, cfg: FFNNTransducerRecogConfig, epoch: int, **_):
        super().__init__(cfg=cfg, epoch=epoch)
        self.ilm_scale = cfg.ilm_scale
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        encoder_state: torch.Tensor,  # [1, E]
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, V]
        encoder_states = encoder_state.expand([history.size(0), encoder_state.size(1)])

        embedding = self.token_embedding.forward(history)  # [B, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, H*A]
        pred_states = self.pred_ffnn.forward(embedding)  # [B, P]

        combination = torch.concat([encoder_states, pred_states], dim=1)  # [B, E+P]
        joint_output = self.joiner.forward(combination)  # [B, V]
        scores = -torch.nn.functional.log_softmax(joint_output, dim=1)  # [B, V]

        scores[:, -1] += self.blank_penalty

        zero_enc_combination = torch.concat([torch.zeros_like(encoder_states), pred_states], dim=1)  # [B, E+P]
        joint_output_ilm = self.joiner.forward(zero_enc_combination)  # [B, V]
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


def get_model_config(target_size: int) -> FFNNTransducerConfig:
    logmel_cfg = RasrCompatibleLogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        min_amp=1.175494e-38,
        num_filters=80,
        alpha=0.0,
    )

    specaug_cfg = SpecaugmentByLengthConfig(
        start_epoch=11,
        time_min_num_masks=2,
        time_max_mask_per_n_frames=25,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=5,
        freq_mask_max_size=16,
    )

    frontend = ModuleFactoryV1(
        GenericFrontendV1,
        GenericFrontendV1Config(
            in_features=80,
            layer_ordering=[
                FrontendLayerType.Conv2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Activation,
                FrontendLayerType.Pool2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Activation,
                FrontendLayerType.Pool2d,
            ],
            conv_kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],
            conv_out_dims=[32, 64, 64, 32],
            conv_strides=None,
            conv_paddings=None,
            pool_kernel_sizes=[(2, 1), (2, 1)],
            pool_strides=None,
            pool_paddings=None,
            activations=[torch.nn.ReLU(), torch.nn.ReLU()],
            out_features=512,
        ),
    )

    ff_cfg = ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = ConformerMHSAV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    conv_cfg = ConformerConvolutionV1Config(
        channels=512,
        kernel_size=31,
        dropout=0.1,
        activation=torch.nn.SiLU(),
        norm=LayerNormNC(512),
    )

    block_cfg = ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "conv", "mhsa", "ff"],
        scales=[0.5, 1.0, 1.0, 0.5],
    )

    conformer_cfg = ConformerEncoderV2Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return FFNNTransducerConfig(
        logmel_cfg=logmel_cfg,
        specaug_cfg=specaug_cfg,
        conformer_cfg=conformer_cfg,
        dropout=0.1,
        enc_dim=512,
        enc_output_indices=[5, 11],
        pred_num_layers=2,
        pred_dim=640,
        pred_activation=torch.nn.Tanh(),
        context_history_size=1,
        context_embedding_dim=256,
        joiner_dim=1024,
        joiner_activation=torch.nn.Tanh(),
        target_size=target_size,
    )
