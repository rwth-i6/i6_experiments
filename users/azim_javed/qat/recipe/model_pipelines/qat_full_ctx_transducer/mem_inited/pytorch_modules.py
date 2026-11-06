__all__ = [
    "QATLstmTransducerConfig",
    "QATLstmTransducerRecogConfig",
    "QATLstmTransducerModel",
    "QATLstmTransducerEncoder",
    "QATLstmTransducerScorer",
    "QATLstmTransducerStateInitializer",
    "QATLstmTransducerStateUpdater",
]

from dataclasses import dataclass
from typing import Tuple, Union

import torch
from ...common.assemblies.conformer import ConformerEncoderQuantV1Config
from ...common.assemblies.conformer.mem_inited import ConformerEncoderQuant

from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length

from ...common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask

from ...common.memristor_layers import LSTMQuant, LinearQuant, EmbeddingQuant, ActivationQuantizer
from ...common.memristor_layers_mem_inited import LSTMQuant, ActivationQuantizer

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

import synaptogen_ml
from synaptogen_ml.memristor_modules.embedding import TiledMemristorEmbedding
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear


@dataclass
class QATLstmTransducerConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugmentByLengthConfig
    conformer_cfg: ConformerEncoderQuantV1Config
    enc_dim: int
    pred_num_layers: int
    pred_dim: int
    pred_activation: torch.nn.Module
    dropout: float
    context_embedding_dim: int
    joiner_dim: int
    joiner_activation: torch.nn.Module
    target_size: int
    weight_bit_prec: int
    weight_quant_dtype: Union[str, torch.dtype]
    weight_quant_method: str
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
class QATLstmTransducerRecogConfig(QATLstmTransducerConfig):
    ilm_scale: float
    blank_penalty: float

@dataclass
class QATLstmTransducerJointADCRecogConfig(QATLstmTransducerRecogConfig):
    joint_l1_converter_hardware_settings: DacAdcHardwareSettings
    joint_l2_converter_hardware_settings: DacAdcHardwareSettings

class QATLstmTransducerModelBase(torch.nn.Module):
    def __init__(self, cfg: QATLstmTransducerConfig, **_):
        super().__init__()
        self.target_size = cfg.target_size
        self.target_size = cfg.target_size
        self.converter_hardware_settings = cfg.converter_hardware_settings
        self.correction_settings = cfg.correction_settings
        self.num_cycles = cfg.num_cycles
        self.embedding_cache = {}

    def cached_embedding_forward(self, indices: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if not use_cache:
            return self.token_embedding(indices)

        label_ids = indices.reshape(-1).tolist()
        cache = self.embedding_cache
        misses = [label_id for label_id in dict.fromkeys(label_ids) if label_id not in cache]
        if misses:
            miss_indices = torch.tensor(misses, dtype=indices.dtype, device=indices.device)
            miss_embeddings = self.token_embedding(miss_indices)
            for label_id, embedding in zip(misses, miss_embeddings):
                cache[label_id] = embedding.detach()

        embeddings = torch.stack([cache[label_id] for label_id in label_ids], dim=0)
        return embeddings.reshape(*indices.shape, embeddings.size(-1))

    def forward_encoder(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor,  # final encoder states [B, T, E]
        torch.Tensor,  # ctc log_probs [B, T, C]
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

        ctc_logits = self.enc_output(encoder_states)  # [B, T, V]
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=2)

        encoder_states_size = torch.sum(sequence_mask, dim=1).type(torch.int32)

        return encoder_states, ctc_log_probs, encoder_states_size

    def forward_prediction_network(
        self,
        targets: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:  # Final prediction network states [B, S+1, P]

        context = torch.nn.functional.pad(targets, [1, 0], value=self.target_size - 1)  # [B, S+1]

        embedding = self.token_embedding(context)  # [B, S+1, A]
        pred_states, _ = self.pred_lstm(embedding)  # [B, S+1, P]
        pred_states = self.pred_act(pred_states)  # [B, S+1, P]

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
        joint_output = self.joiner(joint_input)  # [T_1 * (S_1+1) + T_2 * (S_2 + 1) + ... + T_B * (S_B+1), V]

        return joint_output


class QATLstmTransducerModel(QATLstmTransducerModelBase):
    def __init__(self, cfg: QATLstmTransducerConfig, **_):
        super().__init__(cfg=cfg)

        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerEncoderQuant(cfg.conformer_cfg)

        self.enc_output_q_in = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.enc_output_q_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.enc_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.enc_output_q_in,
            LinearQuant(
                cfg.enc_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.enc_output_q_out,
        )

        self.token_embedding = EmbeddingQuant(
            num_embeddings=self.target_size,
            embedding_dim=cfg.context_embedding_dim,
            padding_idx=cfg.target_size - 1,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
        )

        self.pred_lstm = LSTMQuant(
            input_size=cfg.context_embedding_dim,
            hidden_size=cfg.pred_dim,
            num_layers=cfg.pred_num_layers,
            bias=True,
            batch_first=True,
            dropout=cfg.dropout,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            activation_bit_prec=cfg.activation_bit_prec,
            activation_quant_dtype=cfg.activation_quant_dtype,
            activation_quant_method=cfg.activation_quant_method,
            moving_average=cfg.moving_average,
            converter_hardware_settings=cfg.converter_hardware_settings,
            correction_settings=cfg.correction_settings,
            num_cycles=cfg.num_cycles,
        )
        self.pred_act = cfg.pred_activation

        self.pred_output_q_in = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.pred_output_q_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.pred_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.pred_output_q_in,
            LinearQuant(
                cfg.pred_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.pred_output_q_out,
        )

        self.joiner_q1_in = torch.nn.Identity() 

        self.joiner_q1_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joiner_q2_in = torch.nn.Identity()

        self.joiner_q2_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joiner = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.joiner_q1_in,
            TiledMemristorLinear(
                in_features=cfg.enc_dim + cfg.pred_dim,
                out_features=cfg.joiner_dim,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=cfg.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            self.joiner_q1_out,
            cfg.joiner_activation,
            torch.nn.Dropout(cfg.dropout),
            self.joiner_q2_in,
            TiledMemristorLinear(
                in_features=cfg.joiner_dim,
                out_features=self.target_size,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=cfg.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            self.joiner_q2_out,
        )
        
        self.prep_quant()

    def prep_quant(self):
        synaptogen_ml.set_fast_inference(True)
        self.conformer.prep_quant()

        self.token_embedding.initialized = True
        self.pred_lstm.prep_quant()

        self.joiner[2].initialized = True
        self.joiner[7].initialized = True

class QATLstmTransducerEncoder(QATLstmTransducerModelBase):
    def __init__(self, cfg: QATLstmTransducerConfig, **_):
        super().__init__(cfg=cfg)
        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerEncoderQuant(cfg.conformer_cfg)

        self.enc_output_q_in = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.enc_output_q_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.enc_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.enc_output_q_in,
            LinearQuant(
                cfg.enc_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.enc_output_q_out,
        )

        self.enc_output_indices = []
        self.prep_quant()

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, E], [B]
        encoder_states, _, encoder_states_size = self.forward_encoder(audio_samples=audio_samples, audio_samples_size=audio_samples_size)

        return encoder_states, encoder_states_size  # [B, T, E]

    def prep_quant(self):
        synaptogen_ml.set_fast_inference(True)
        self.conformer.prep_quant()

class QATLstmTransducerScorer(QATLstmTransducerModelBase):
    def __init__(self, cfg: QATLstmTransducerRecogConfig, **_):
        super().__init__(cfg=cfg)
        self.ilm_scale = cfg.ilm_scale
        self.blank_penalty = cfg.blank_penalty

        self.joiner_q1_in = torch.nn.Identity() 

        self.joiner_q1_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )
        self.joiner_q2_in = torch.nn.Identity()

        self.joiner_q2_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.joiner = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.joiner_q1_in,
            TiledMemristorLinear(
                in_features=cfg.enc_dim + cfg.pred_dim,
                out_features=cfg.joiner_dim,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=getattr(
                    cfg, "joint_l1_converter_hardware_settings", cfg.converter_hardware_settings
                ),
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            self.joiner_q1_out,
            cfg.joiner_activation,
            torch.nn.Dropout(cfg.dropout),
            self.joiner_q2_in,
            TiledMemristorLinear(
                in_features=cfg.joiner_dim,
                out_features=self.target_size,
                weight_precision=cfg.weight_bit_prec,
                converter_hardware_settings=getattr(
                    cfg, "joint_l2_converter_hardware_settings", cfg.converter_hardware_settings
                ),
                memristor_inputs=128,
                memristor_outputs=128,
            ),
            self.joiner_q2_out,
        )
        self.prep_quant()

    def prep_quant(self):
        synaptogen_ml.set_fast_inference(True)

        self.joiner[2].initialized = True
        self.joiner[7].initialized = True
        


    def forward(
        self,
        encoder_state: torch.Tensor,  # [1, E]
        lstm_out: torch.Tensor,  # [B, P]
    ) -> torch.Tensor:  # [B, V]
        encoder_states = encoder_state.expand([lstm_out.size(0), encoder_state.size(1)])
        combination = torch.concat([encoder_states, lstm_out], dim=1)  # [B, E+P]

        if self.ilm_scale != 0:
            zero_enc_combination = torch.concat([torch.zeros_like(encoder_states), lstm_out], dim=1)
            joint_output, joint_output_ilm = self.joiner(
                torch.concat([combination, zero_enc_combination], dim=0)
            ).chunk(2, dim=0)  # each [B, V]
        else:
            joint_output = self.joiner(combination)  # [B, V]

        scores = -torch.nn.functional.log_softmax(joint_output, dim=1)  # [B, V]
        scores[:, -1] += self.blank_penalty

        if self.ilm_scale != 0:
            ilm_log_probs = torch.nn.functional.log_softmax(joint_output_ilm, dim=1)  # [B, V]
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

            scores -= self.ilm_scale * (-ilm_log_probs)

        return scores

    def forward_separate(
        self,
        encoder_state: torch.Tensor,  # [1, E]
        lstm_out: torch.Tensor,  # [B, P]
    ) -> torch.Tensor:  # [B, V]
        encoder_states = encoder_state.expand([lstm_out.size(0), encoder_state.size(1)])

        combination = torch.concat([encoder_states, lstm_out], dim=1)  # [B, E+P]
        joint_output = self.joiner(combination)  # [B, V]
        scores = -torch.nn.functional.log_softmax(joint_output, dim=1)  # [B, V]

        scores[:, -1] += self.blank_penalty

        if self.ilm_scale != 0:
            zero_enc_combination = torch.concat([torch.zeros_like(encoder_states), lstm_out], dim=1)  # [B, E+P]
            joint_output_ilm = self.joiner(zero_enc_combination)  # [B, V]
            ilm_log_probs = torch.nn.functional.log_softmax(joint_output_ilm, dim=1)  # [B, V]

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


class QATLstmTransducerStateInitializer(QATLstmTransducerModelBase):
    def __init__(self, cfg: QATLstmTransducerConfig, **_):
        super().__init__(cfg=cfg)

        self.token_embedding = TiledMemristorEmbedding(
            num_embeddings=self.target_size,
            embedding_dim=cfg.context_embedding_dim,
            padding_idx=cfg.target_size - 1,
            weight_precision=cfg.weight_bit_prec,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        self.pred_lstm = LSTMQuant(
            input_size=cfg.context_embedding_dim,
            hidden_size=cfg.pred_dim,
            num_layers=cfg.pred_num_layers,
            bias=True,
            batch_first=True,
            dropout=cfg.dropout,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            activation_bit_prec=cfg.activation_bit_prec,
            activation_quant_dtype=cfg.activation_quant_dtype,
            activation_quant_method=cfg.activation_quant_method,
            moving_average=cfg.moving_average,
            converter_hardware_settings=cfg.converter_hardware_settings,
            correction_settings=cfg.correction_settings,
            num_cycles=cfg.num_cycles,
        )
        self.pred_act = cfg.pred_activation

        self.pred_output_q_in = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.pred_output_q_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.pred_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.pred_output_q_in,
            LinearQuant(
                cfg.pred_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.pred_output_q_out,
        )
        self.prep_quant()

    def forward(self, use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.token_embedding.input_factor.device
        token = torch.tensor([[self.target_size - 1]], dtype=torch.int32, device=device)  # [1, 1]
        embed = self.cached_embedding_forward(token, use_cache=use_cache)  # [1, 1, E]
        h_0 = torch.zeros((self.pred_lstm.num_layers, 1, self.pred_lstm.hidden_size), dtype=embed.dtype, device=embed.device)  # [L, 1, P]
        c_0 = torch.zeros_like(h_0)  # [L, 1, P]

        lstm_out, (h_0, c_0) = self.pred_lstm(embed, (h_0, c_0), use_cache=use_cache)  # [1, 1, P], [L, 1, P], [L, 1, P]
        lstm_out = self.pred_act(lstm_out)  # [1, 1, P]

        lstm_out = lstm_out.reshape([1, self.pred_lstm.hidden_size])  # [1, P]
        h_0 = h_0.transpose(0, 1)  # [1, L, P]
        c_0 = c_0.transpose(0, 1)  # [1, L, P]

        return lstm_out, h_0, c_0  # [1, P], [1, L, P], [1, L, P]

    def prep_quant(self):
        self.pred_lstm.prep_quant()
        self.token_embedding.initialized = True


class QATLstmTransducerStateUpdater(QATLstmTransducerModelBase):
    def __init__(self, cfg: QATLstmTransducerConfig, **_):
        super().__init__(cfg=cfg)

        self.token_embedding = TiledMemristorEmbedding(
            num_embeddings=self.target_size,
            embedding_dim=cfg.context_embedding_dim,
            padding_idx=cfg.target_size - 1,
            weight_precision=cfg.weight_bit_prec,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        self.pred_lstm = LSTMQuant(
            input_size=cfg.context_embedding_dim,
            hidden_size=cfg.pred_dim,
            num_layers=cfg.pred_num_layers,
            bias=True,
            batch_first=True,
            dropout=cfg.dropout,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            activation_bit_prec=cfg.activation_bit_prec,
            activation_quant_dtype=cfg.activation_quant_dtype,
            activation_quant_method=cfg.activation_quant_method,
            moving_average=cfg.moving_average,
            converter_hardware_settings=cfg.converter_hardware_settings,
            correction_settings=cfg.correction_settings,
            num_cycles=cfg.num_cycles,
        )
        self.pred_act = cfg.pred_activation

        self.pred_output_q_in = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.pred_output_q_out = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=2,
            moving_avrg=cfg.moving_average,
        )

        self.pred_output = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            self.pred_output_q_in,
            LinearQuant(
                cfg.pred_dim,
                self.target_size,
                weight_bit_prec=cfg.weight_bit_prec,
                weight_quant_dtype=cfg.weight_quant_dtype,
                weight_quant_method=cfg.weight_quant_method,
                bias=True,
            ),
            self.pred_output_q_out,
        )
        self.prep_quant()

    def forward(
        self,
        token: torch.Tensor,  # [B]
        lstm_h: torch.Tensor,  # [B, L, H]
        lstm_c: torch.Tensor,  # [B, L, H]
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.cached_embedding_forward(token, use_cache=use_cache)  # [B, E]
        embed = embed.reshape([embed.size(0), 1, embed.size(1)])  # [B, 1, E]

        lstm_h = lstm_h.transpose(0, 1)  # [B, L, P]
        lstm_c = lstm_c.transpose(0, 1)  # [B, L, P]

        lstm_out, (new_lstm_h, new_lstm_c) = self.pred_lstm(embed, (lstm_h, lstm_c), use_cache=use_cache)  # [B, 1, P] [L, 1, P] [L, 1, P]
        lstm_out = self.pred_act(lstm_out)  # [B, 1, P]

        lstm_out = lstm_out.reshape([-1, self.pred_lstm.hidden_size])  # [B, P]
        new_lstm_h = new_lstm_h.transpose(0, 1)  # [B, L, P]
        new_lstm_c = new_lstm_c.transpose(0, 1)  # [B, L, P]

        return lstm_out, new_lstm_h, new_lstm_c

    def prep_quant(self):
        self.pred_lstm.prep_quant()
        self.token_embedding.initialized = True
