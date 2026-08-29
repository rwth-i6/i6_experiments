__all__ = [
    "QATFFNNTransducerConfig",
    "QATFFNNTransducerRecogConfig",
    "QATFFNNTransducerModel",
    "QATFFNNTransducerEncoder",
    "QATFFNNTransducerScorer",
]

from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Dict, List, Callable

import torch

from ...common.assemblies.conformer import ConformerEncoderQuantV1Config
from ...common.assemblies.conformer.mem_inited import ConformerEncoderQuant
from ...common.memristor_layers import LinearQuant, EmbeddingQuant, ActivationQuantizer

from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length

from ...common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.embedding import TiledMemristorEmbedding

import synaptogen_ml
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear


@dataclass
class QATFFNNTransducerConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugmentByLengthConfig
    conformer_cfg: ConformerEncoderQuantV1Config
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
    weight_bit_prec: int
    weight_quant_dtype: Union[str, torch.dtype]
    weight_quant_method: str
    # v2
    activation_bit_prec: int
    activation_quant_dtype: Union[str, torch.dtype]
    activation_quant_method: str
    moving_average: Union[float, None]
    converter_hardware_settings: DacAdcHardwareSettings
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings
    correction_settings: Union[CycleCorrectionSettings, None]
    num_cycles: int
    version_control: Union[str, None]

    def __sis_state__(self):
        import dataclasses, torch
        from sisyphus import tk

        def _sanitize(v):
            if isinstance(v, torch.dtype):
                return str(v)
                # return (str(v) + self.hash_control) if self.hash_control is not None else str(v)
            if isinstance(v, tk.Path):
                return v  # keep for path extraction
            if dataclasses.is_dataclass(v):
                return {f.name: _sanitize(getattr(v, f.name)) for f in dataclasses.fields(v)}
            if isinstance(v, dict):
                return {k: _sanitize(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(_sanitize(x) for x in v)
            return v

        return {f.name: _sanitize(getattr(self, f.name)) for f in dataclasses.fields(self) if f.name != "hash_control"}

    def __sis_hash__(self):
        return str(type(self))

    def with_replaced(self, **kwargs):
        import dataclasses

        consumed = set()

        def _recurse(obj):
            # 1. Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                new_seq = type(obj)(_recurse(v) for v in obj)
                if any(old is not new for old, new in zip(obj, new_seq)):
                    return new_seq
                return obj

            # 2. Base case: not a dataclass
            if not dataclasses.is_dataclass(obj):
                return obj

            # 3. Handle dataclasses
            changes = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                if f.name in kwargs:
                    changes[f.name] = kwargs[f.name]
                    consumed.add(f.name)
                else:
                    new_val = _recurse(val)
                    if new_val is not val:
                        changes[f.name] = new_val
            if changes:
                return dataclasses.replace(obj, **changes)
            return obj

        result = _recurse(self)
        unconsumed = set(kwargs) - consumed
        assert not unconsumed, f"with_replaced: keys not found in config tree: {unconsumed}"
        return result


@dataclass
class QATFFNNTransducerRecogConfig(QATFFNNTransducerConfig):
    ilm_scale: float
    blank_penalty: float


class QATFFNNTransducerModelBase(torch.nn.Module):
    def prep_quant(self):
        pass

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
            features, features_size = self.feature_extraction(audio_samples, audio_samples_size)  # [B, T, F], [B]
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

        encoder_states, sequence_mask = self.conformer(features, sequence_mask)  # [B, T, E], [B, T]
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

        embedding = self.token_embedding(context)  # [B, S+1, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, S+1, H*A]
        pred_states = self.prediction_net(embedding)  # [B, S+1, P]

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
        joint_output = self.joint_net(joint_input)  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), V]

        return joint_output


class QATFFNNTransducerModel(QATFFNNTransducerModelBase):
    def __init__(self, cfg: QATFFNNTransducerConfig, **_):
        super().__init__()
        self.target_size = cfg.target_size

        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerEncoderQuant(cfg.conformer_cfg)

        self.enc_output_in_q = torch.nn.Identity()

        self.enc_output_out_q = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.encoder_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.enc_output_in_q,
            LinearQuant(
                cfg.enc_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.enc_output_out_q,
        )  # only for loss, but quantized just for consistency

        self.context_history_size = cfg.context_history_size
        self.token_embedding = TiledMemristorEmbedding(
            num_embeddings=self.target_size,
            embedding_dim=cfg.context_embedding_dim,
            padding_idx=cfg.target_size - 1,
            weight_precision=cfg.weight_bit_prec,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        # self.token_embedding = EmbeddingQuant(
        #     num_embeddings=self.target_size,
        #     embedding_dim=cfg.context_embedding_dim,
        #     padding_idx=cfg.target_size - 1,
        #     weight_bit_prec=cfg.weight_bit_prec,
        #     weight_quant_dtype=cfg.weight_quant_dtype,
        #     weight_quant_method=cfg.weight_quant_method,
        # )

        prediction_layers = []
        prev_size = self.context_history_size * cfg.context_embedding_dim
        for _ in range(cfg.pred_num_layers):
            prediction_layers.append(torch.nn.Dropout(cfg.dropout))
            prediction_layers.append(torch.nn.Identity())
            prediction_layers.append(
                TiledMemristorLinear(
                    in_features=prev_size,
                    out_features=cfg.pred_dim,
                    weight_precision=cfg.weight_bit_prec,
                    converter_hardware_settings=cfg.converter_hardware_settings,
                    memristor_inputs=128,
                    memristor_outputs=128,
                )
                # LinearQuant(
                #     prev_size,
                #     cfg.pred_dim,
                #     weight_bit_prec=cfg.weight_bit_prec,
                #     weight_quant_dtype=cfg.weight_quant_dtype,
                #     weight_quant_method=cfg.weight_quant_method,
                #     bias=True,
                # )
            )
            prediction_layers.append(
                ActivationQuantizer(
                    bit_precision=cfg.activation_bit_prec,
                    dtype=cfg.activation_quant_dtype,
                    method=cfg.activation_quant_method,
                    channel_axis=1,
                    moving_avrg=cfg.moving_average,
                )
            )
            prediction_layers.append(cfg.pred_activation)
            prev_size = cfg.pred_dim
        self.prediction_net = torch.nn.Sequential(*prediction_layers)

        self.prediction_output_in_q = torch.nn.Identity()

        self.prediction_output_out_q = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.prediction_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.prediction_output_in_q,
            LinearQuant(
                cfg.pred_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.prediction_output_out_q,
        )  # only for loss, but quantized just for consistency

        self.joint_net_q1_in = torch.nn.Identity()

        self.joint_net_q1_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joint_net_q2_in = torch.nn.Identity()

        self.joint_net_q2_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joint_net_q2_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joint_net = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.joint_net_q1_in,
            TiledMemristorLinear(
                in_features=cfg.enc_dim + cfg.pred_dim,
                out_features=cfg.joiner_dim,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=cfg.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            # LinearQuant(
            #     cfg.enc_dim + cfg.pred_dim,
            #     cfg.joiner_dim,
            #     weight_bit_prec=cfg.weight_bit_prec,
            #     weight_quant_dtype=cfg.weight_quant_dtype,
            #     weight_quant_method=cfg.weight_quant_method,
            #     bias=True,
            # ),
            self.joint_net_q1_out,
            cfg.joiner_activation,
            torch.nn.Dropout(cfg.dropout),
            self.joint_net_q2_in,
            TiledMemristorLinear(
                in_features=cfg.joiner_dim,
                out_features=self.target_size,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=cfg.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            # LinearQuant(
            #     cfg.joiner_dim,
            #     self.target_size,
            #     weight_bit_prec=cfg.weight_bit_prec,
            #     weight_quant_dtype=cfg.weight_quant_dtype,
            #     weight_quant_method=cfg.weight_quant_method,
            #     bias=True,
            # ),
            self.joint_net_q2_out,
        )
        self.converter_hardware_settings = cfg.converter_hardware_settings
        self.correction_settings = cfg.correction_settings
        self.num_cycles = cfg.num_cycles
        self.prep_quant()

    def _convert_linear(self, lin: LinearQuant, in_quant: ActivationQuantizer) -> torch.nn.Module:

        lin.weight_quantizer.set_scale_and_zp()
        in_quant.set_scale_and_zp()
        if lin.pruning_config is not None:
            with torch.no_grad():
                lin.weight.data = lin.pruning_config.apply(lin.weight.data, training=False)
        weight_prec = lin.weight_bit_prec if lin.weight_bit_prec != 1.5 else 2
        mem_lin = TiledMemristorLinear(
            in_features=lin.in_features,
            out_features=lin.out_features,
            weight_precision=weight_prec,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(
            activation_quant=in_quant,
            linear_quant=lin,
            num_cycles_init=self.num_cycles,
            correction_settings=self.correction_settings,
        )
        return mem_lin

    def _convert_embedding(self, emb: EmbeddingQuant) -> torch.nn.Module:
        emb.weight_quantizer.set_scale_and_zp()
        if emb.pruning_config is not None:
            with torch.no_grad():
                emb.weight.data = emb.pruning_config.apply(emb.weight.data, training=False)
        weight_prec = emb.weight_bit_prec if emb.weight_bit_prec != 1.5 else 2
        mem_emb = TiledMemristorEmbedding(
            num_embeddings=emb.num_embeddings,
            embedding_dim=emb.embedding_dim,
            weight_precision=weight_prec,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
            padding_idx=emb.padding_idx,
        )
        mem_emb.init_from_embedding_quant(
            embedding_quant=emb,
            num_cycles_init=self.num_cycles,
            correction_settings=self.correction_settings,
        )
        return mem_emb

    def prep_quant(self):
        synaptogen_ml.set_fast_inference(True)
        self.conformer.prep_quant()

        self.token_embedding.initialized = True

        for base in range(0, len(self.prediction_net), 5):
            self.prediction_net[base + 2].initialized = True
        self.joint_net[2].initialized = True
        self.joint_net[7].initialized = True


class QATFFNNTransducerEncoder(QATFFNNTransducerModelBase):
    def __init__(self, cfg: QATFFNNTransducerConfig, **_):
        super().__init__()
        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerEncoderQuant(cfg.conformer_cfg)
        self.enc_output_indices = []
        self.prep_quant()

    def prep_quant(self):
        synaptogen_ml.set_fast_inference(True)
        self.conformer.prep_quant()

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T', E], [B]
        encoder_states, encoder_states_len = self.forward_encoder(audio_samples=audio_samples, audio_samples_size=audio_samples_size)
        return encoder_states, encoder_states_len


class QATFFNNTransducerScorer(QATFFNNTransducerModelBase):
    def __init__(self, cfg: QATFFNNTransducerRecogConfig, **_):
        super().__init__()
        self.target_size = cfg.target_size
        self.context_history_size = cfg.context_history_size
        self.token_embedding = TiledMemristorEmbedding(
            num_embeddings=self.target_size,
            embedding_dim=cfg.context_embedding_dim,
            padding_idx=cfg.target_size - 1,
            weight_precision=cfg.weight_bit_prec,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        prediction_layers = []
        prev_size = self.context_history_size * cfg.context_embedding_dim
        for _ in range(cfg.pred_num_layers):
            prediction_layers.append(torch.nn.Dropout(cfg.dropout))
            prediction_layers.append(torch.nn.Identity())
            prediction_layers.append(
                TiledMemristorLinear(
                    in_features=prev_size,
                    out_features=cfg.pred_dim,
                    weight_precision=cfg.weight_bit_prec,
                    converter_hardware_settings=cfg.converter_hardware_settings,
                    memristor_inputs=128,
                    memristor_outputs=128,
                )
            )
            prediction_layers.append(
                ActivationQuantizer(
                    bit_precision=cfg.activation_bit_prec,
                    dtype=cfg.activation_quant_dtype,
                    method=cfg.activation_quant_method,
                    channel_axis=1,
                    moving_avrg=cfg.moving_average,
                )
            )
            prediction_layers.append(cfg.pred_activation)
            prev_size = cfg.pred_dim
        self.prediction_net = torch.nn.Sequential(*prediction_layers)

        self.prediction_output_in_q = torch.nn.Identity()

        self.prediction_output_out_q = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.prediction_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.prediction_output_in_q,
            LinearQuant(
                cfg.pred_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.prediction_output_out_q,
        )  # only for loss, but quantized just for consistency

        self.joint_net_q1_in = torch.nn.Identity()

        self.joint_net_q1_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joint_net_q2_in = torch.nn.Identity()

        self.joint_net_q2_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joint_net_q2_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joint_net = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.joint_net_q1_in,
            TiledMemristorLinear(
                in_features=cfg.enc_dim + cfg.pred_dim,
                out_features=cfg.joiner_dim,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=cfg.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            self.joint_net_q1_out,
            cfg.joiner_activation,
            torch.nn.Dropout(cfg.dropout),
            self.joint_net_q2_in,
            TiledMemristorLinear(
                in_features=cfg.joiner_dim,
                out_features=self.target_size,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=cfg.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            self.joint_net_q2_out,
        )
        self.converter_hardware_settings = cfg.converter_hardware_settings
        self.correction_settings = cfg.correction_settings
        self.num_cycles = cfg.num_cycles
        self.prep_quant()

        self.ilm_scale = cfg.ilm_scale
        self.blank_penalty = cfg.blank_penalty

    def prep_quant(self):
        synaptogen_ml.set_fast_inference(True)
        self.token_embedding.initialized = True
        for base in range(0, len(self.prediction_net), 5):
            self.prediction_net[base + 2].initialized = True
        self.joint_net[2].initialized = True
        self.joint_net[7].initialized = True

    def forward(
        self,
        encoder_state: torch.Tensor,  # [1, E]
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, V]
        embedding = self.token_embedding(history)  # [B, H, A]
        embedding = torch.reshape(
            embedding, shape=[*(embedding.shape[:-2]), embedding.shape[-2] * embedding.shape[-1]]
        )  # [B, H*A]
        pred_state = self.prediction_net(embedding)  # [B, P]

        joint_input = torch.concat([encoder_state.expand([pred_state.size(0), -1]), pred_state], dim=-1)  # [B, E+P]
        joint_output = self.joint_net(joint_input)  # [B, V]
        scores = -torch.nn.functional.log_softmax(joint_output, dim=1)  # [B, V]

        scores[:, -1] += self.blank_penalty

        if self.ilm_scale != 0:
            zero_enc = torch.zeros_like(encoder_state)  # [B, E]
            ilm_joint_input = torch.concat([zero_enc.expand([pred_state.size(0), -1]), pred_state], dim=-1)  # [B, E+P]
            ilm_joint_output = self.joint_net(ilm_joint_input)  # [B, V]
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
