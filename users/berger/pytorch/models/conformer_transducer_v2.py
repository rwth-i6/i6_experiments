from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import i6_models.assemblies.conformer as conformer_i6
import i6_models.parts.conformer as conformer_parts_i6
import torch
from i6_core.returnn.config import CodeWrapper
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport, ExternalImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from i6_experiments.users.berger.pytorch.custom_parts.sequential import SequentialModuleV1, SequentialModuleV1Config
from i6_experiments.users.berger.pytorch.custom_parts.speed_perturbation import (
    SpeedPerturbationModuleV1,
    SpeedPerturbationModuleV1Config,
)

from ..custom_parts.specaugment import (
    SpecaugmentByLengthConfigV1,
    SpecaugmentByLengthModuleV1,
)
from .util import lengths_to_padding_mask


@dataclass
class TransducerTranscriberConfig(ModelConfiguration):
    feature_extraction: ModuleFactoryV1
    specaugment: ModuleFactoryV1
    encoder: ModuleFactoryV1
    layer_size: int
    target_size: int
    enc_loss_layers: List[int] = field(default_factory=list)


class TransducerTranscriber(torch.nn.Module):
    def __init__(self, cfg: TransducerTranscriberConfig, **_) -> None:
        super().__init__()
        self.feature_extraction = cfg.feature_extraction()
        self.specaugment = cfg.specaugment()
        self.encoder = cfg.encoder()

        self.enc_loss_layers = cfg.enc_loss_layers
        self.output_layers = torch.nn.ModuleDict(
            {
                f"output_{layer_idx}": torch.nn.Linear(cfg.layer_size, cfg.target_size)
                for layer_idx in cfg.enc_loss_layers
            }
        )

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:  # [B, T, C], Dict(l: [B, T, C]), [B]
        with torch.no_grad():
            sources = sources.squeeze(-1)
            x, source_lengths = self.feature_extraction(sources, source_lengths)
            sequence_mask = lengths_to_padding_mask(source_lengths)

            x = self.specaugment(x)  # [B, T, F]

        return_layers = self.enc_loss_layers.copy()
        return_layers.append(len(self.encoder.module_list) - 1)

        intermediate_encodings, sequence_mask = self.encoder(
            x, sequence_mask=sequence_mask, return_layers=return_layers
        )  # List([B, T, E]), [B, T]

        source_encodings = intermediate_encodings[-1]

        intermediate_logits = {
            layer_idx: self.output_layers[f"output_{layer_idx}"](encoding)
            for layer_idx, encoding in zip(self.enc_loss_layers, intermediate_encodings)
        }  # Dict(l: [B, T, C])

        return (
            source_encodings,
            intermediate_logits,
            torch.sum(sequence_mask, dim=1).to(torch.int32),
        )  # [B, T, E], Dict(l: [B, T, C]), [B]


# class TransducerTranscriberNoFeatExtr(torch.nn.Module):
#     def __init__(self, cfg: TransducerTranscriberConfig, **_) -> None:
#         super().__init__()
#         self.specaugment = cfg.specaugment()
#         self.encoder = cfg.encoder()
#
#     def forward(
#         self,
#         sources: torch.Tensor,  # [B, T, F]
#         source_lengths: torch.Tensor,  # [B]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
#         with torch.no_grad():
#             sequence_mask = lengths_to_padding_mask(source_lengths)
#
#             x = self.specaugment(sources)  # [B, T, F]
#
#         x, sequence_mask = self.encoder(x, sequence_mask)  # [B, T, E], [B, T]
#
#         return x, torch.sum(sequence_mask, dim=1).to(torch.int32)  # [B, T, C], [B]


@dataclass
class FFNNTransducerPredictorConfig(ModelConfiguration):
    layers: int
    layer_size: int
    activation: Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    dropout: float
    context_history_size: int
    context_embedding_size: int
    target_size: int
    blank_id: int


class FFNNTransducerPredictor(torch.nn.Module):
    def __init__(self, cfg: FFNNTransducerPredictorConfig, **_) -> None:
        super().__init__()
        self.blank_id = cfg.blank_id
        self.embedding = torch.nn.Embedding(
            num_embeddings=cfg.target_size, embedding_dim=cfg.context_embedding_size, padding_idx=self.blank_id
        )
        self.context_history_size = cfg.context_history_size
        prediction_layers = []
        prev_size = self.context_history_size * cfg.context_embedding_size
        for _ in range(cfg.layers):
            prediction_layers.append(torch.nn.Dropout(cfg.dropout))
            prediction_layers.append(torch.nn.Linear(prev_size, cfg.layer_size))
            prediction_layers.append(cfg.activation)
            prev_size = cfg.layer_size

        self.network = torch.nn.Sequential(*prediction_layers)

    def forward(
        self,
        targets: torch.Tensor,  # [B, S],
        target_lengths: torch.Tensor,  # [B],
        history: Optional[torch.Tensor] = None,  # [B, H], use all blanks if None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, S + 1, C], [B], [B, H]
        # extend input by prepending either the state if it's given or some history consisting of blanks
        if history is None:
            history = torch.full(
                (targets.size(0), self.context_history_size),
                fill_value=self.blank_id,
                dtype=targets.dtype,
                device=targets.device,
            )  # [B, H]

        extended_targets = torch.concat([history, targets], dim=1)  # [B, S+H]

        return_history = extended_targets[:, -(self.context_history_size) :]  # [B, H]

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

        a = self.embedding(context)  # [B, S+1, H, E]
        a = torch.reshape(a, shape=[*(a.shape[:-2]), a.shape[-2] * a.shape[-1]])  # [B, S+1, H*E]
        a = self.network(a)  # [B, S+1, P]
        return a, target_lengths, return_history

    def forward_viterbi(
        self,
        targets: torch.Tensor,  # [B, T],
        target_lengths: torch.Tensor,  # [B],
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
        # Get alignment sequence including blanks as labels, e.g.
        # [0, 0, 1, 0, 2, 0, 0, 3, 0]

        B, T = targets.shape

        history = torch.zeros(
            (B, T, self.context_history_size), dtype=targets.dtype, device=targets.device
        )  # [B, T, H]

        # start with all-blank history
        recent_labels = torch.full(
            (B, self.context_history_size), fill_value=self.blank_id, dtype=targets.dtype, device=targets.device
        )  # [B, H]

        for t in range(T):
            # set context at frame t
            history[:, t, :] = recent_labels

            current_labels = targets[:, t]  # [B]
            non_blank_positions = current_labels != self.blank_id  # [B]

            # shift recent_labels and append next one if we see a non-blank
            recent_labels[non_blank_positions, 1:] = recent_labels[non_blank_positions, :-1]
            recent_labels[non_blank_positions, 0] = current_labels[non_blank_positions]

        a = self.embedding(history)  # [B, T, H, E]
        a = torch.reshape(a, shape=[*(a.shape[:-2]), a.shape[-2] * a.shape[-1]])  # [B, T, H*E]
        a = self.network(a)  # [B, T, P]
        return a, target_lengths


class CombinationMode(Enum):
    CONCAT = auto()
    SUM = auto()


@dataclass
class TransducerJoinerConfig(ModelConfiguration):
    layer_size: int
    act: torch.nn.Module
    input_size: int
    target_size: int
    combination_mode: CombinationMode


class TransducerJoiner(torch.nn.Module):
    def __init__(self, cfg: TransducerJoinerConfig, **_) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_size, cfg.layer_size),
            cfg.act,
            torch.nn.Linear(cfg.layer_size, cfg.target_size),
        )
        self.combination_mode = cfg.combination_mode

    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, T, E],
        source_lengths: torch.Tensor,  # [B],
        target_encodings: torch.Tensor,  # [B, S+1, P],
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, T, S+1, C], [B], [B],
        source_encodings = source_encodings.unsqueeze(2).expand(
            target_encodings.size(0), -1, target_encodings.size(1), -1
        )  # [B, T, S+1, E]
        target_encodings = target_encodings.unsqueeze(1).expand(-1, source_encodings.size(1), -1, -1)  # [B, T, S+1, P]

        if self.combination_mode == CombinationMode.CONCAT:
            joiner_inputs = torch.concat([source_encodings, target_encodings], dim=-1)  # [B, T, S+1, E + P]
        elif self.combination_mode == CombinationMode.SUM:
            joiner_inputs = source_encodings + target_encodings  # [B, T, S+1, E=P]

        output = self.network(joiner_inputs)  # [B, T, S+1, C]

        if not self.training:
            output = torch.log_softmax(output, dim=-1)  # [B, T, S+1, C]

        return output, source_lengths, target_lengths

    def forward_viterbi(
        self,
        source_encodings: torch.Tensor,  # [B, T, E],
        source_lengths: torch.Tensor,  # [B],
        target_encodings: torch.Tensor,  # [B, T, P],
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, T, C], [B], [B],
        if self.combination_mode == CombinationMode.CONCAT:
            joiner_inputs = torch.concat([source_encodings, target_encodings], dim=-1)  # [B, T, E + P]
        elif self.combination_mode == CombinationMode.SUM:
            joiner_inputs = source_encodings + target_encodings  # [B, T, E=P]

        output = self.network(joiner_inputs)  # [B, T, C]

        if not self.training:
            output = torch.log_softmax(output, dim=-1)  # [B, T, C]

        return output, source_lengths, target_lengths


class PackedTransducerJoiner(torch.nn.Module):
    def __init__(self, cfg: TransducerJoinerConfig, **_) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_size, cfg.layer_size),
            cfg.act,
            torch.nn.Linear(cfg.layer_size, cfg.target_size),
        )
        self.combination_mode = cfg.combination_mode

    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, T, E],
        source_lengths: torch.Tensor,  # [B],
        target_encodings: torch.Tensor,  # [B, S+1, P],
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C], [B], [B],
        batch_tensors = []

        for b in range(source_encodings.size(0)):
            valid_source = source_encodings[b, : source_lengths[b], :]  # [T_b, E]
            valid_target = target_encodings[b, : target_lengths[b] + 1, :]  # [S_b+1, P]

            expanded_source = valid_source.unsqueeze(1).expand(
                -1, int(target_lengths[b].item()) + 1, -1
            )  # [T_b, S_b+1, E]
            expanded_target = valid_target.unsqueeze(0).expand(int(source_lengths[b].item()), -1, -1)  # [T_b, S_b+1, P]

            if self.combination_mode == CombinationMode.CONCAT:
                combination = torch.concat([expanded_source, expanded_target], dim=-1)  # [T_b, S_b+1, E + P]
            elif self.combination_mode == CombinationMode.SUM:
                combination = expanded_source + expanded_target  # [T_b, S_b+1, E (=P)]
            else:
                raise NotImplementedError

            batch_tensors.append(combination.reshape(-1, combination.size(2)))  # [T_b * (S_b+1), E(+P)]

        joint_encodings = torch.concat(
            batch_tensors, dim=0
        )  # [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), E(+P)]
        output = self.network(joint_encodings)  # [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C]

        if not self.training:
            output = torch.log_softmax(output, dim=-1)  # [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C]

        return output, source_lengths, target_lengths


@dataclass
class FFNNTransducerConfig(ModelConfiguration):
    transcriber_cfg: TransducerTranscriberConfig
    predictor_cfg: FFNNTransducerPredictorConfig
    joiner_cfg: TransducerJoinerConfig


@dataclass
class FFNNTransducerWithIlmConfig(FFNNTransducerConfig):
    ilm_scale: float


class FFNNTransducer(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.transcriber = TransducerTranscriber(cfg.transcriber_cfg)
        self.predictor = FFNNTransducerPredictor(cfg.predictor_cfg)
        self.joiner = PackedTransducerJoiner(cfg.joiner_cfg)

    def transcribe(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:  # [B, T, E], Dict(l: [B, T, E]), [B]
        return self.transcriber.forward(sources=sources, source_lengths=source_lengths)

    def predict(
        self,
        targets: torch.Tensor,  # [B, S]
        target_lengths: torch.Tensor,  # [B]
        history: Optional[torch.Tensor] = None,  # [B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, S+1, P], [B], [B, H]
        return self.predictor.forward(targets=targets, target_lengths=target_lengths, history=history)

    def join(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        source_lengths: torch.Tensor,  # [B]
        target_encodings: torch.Tensor,  # [B, S+1, P]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C], [B], [B],
        return self.joiner.forward(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B, S]
        target_lengths: torch.Tensor,  # [B]
        history: Optional[torch.Tensor] = None,  # [B, H]
    ) -> Tuple[
        torch.Tensor, Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # [T_1 * (S_1+1) + T_2 * (S_2+1) + ... + T_B * (S_B+1), C], Dict(l: [B, T, C]), [B], [B], [B, H]
        source_encodings, intermediate_logits, source_lengths = self.transcribe(
            sources=sources, source_lengths=source_lengths
        )
        label_encodings, target_lengths, history = self.predict(
            targets=targets, target_lengths=target_lengths, history=history
        )

        output, source_lengths, target_lengths = self.join(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=label_encodings,
            target_lengths=target_lengths,
        )

        return output, intermediate_logits, source_lengths, target_lengths, history


class FFNNTransducerViterbi(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.transcriber = TransducerTranscriber(cfg.transcriber_cfg)
        self.predictor = FFNNTransducerPredictor(cfg.predictor_cfg)
        self.joiner = TransducerJoiner(cfg.joiner_cfg)

    def transcribe(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:  # [B, T, E], Dict(l: [B, T, C]), [B]
        return self.transcriber.forward(sources=sources, source_lengths=source_lengths)

    def predict(
        self,
        targets: torch.Tensor,  # [B, T]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, P], [B]
        return self.predictor.forward_viterbi(targets=targets, target_lengths=target_lengths)

    def join(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        source_lengths: torch.Tensor,  # [B]
        target_encodings: torch.Tensor,  # [B, T, P]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, T, C], [B], [B]
        return self.joiner.forward_viterbi(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B, T]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[
        torch.Tensor, Dict[int, torch.Tensor], torch.Tensor, torch.Tensor
    ]:  # [B, T, C], Dict(l: [B, T, C]), [B], [B]
        source_encodings, intermediate_logits, source_lengths = self.transcribe(
            sources=sources, source_lengths=source_lengths
        )
        target_encodings, target_lengths = self.predict(targets=targets, target_lengths=target_lengths)

        output, source_lengths, target_lengths = self.join(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

        return output, intermediate_logits, source_lengths, target_lengths


class FFNNTransducerEncoderOnly(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.transcriber = TransducerTranscriber(cfg.transcriber_cfg)

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T', E]
        source_encodings, _, source_lengths = self.transcriber.forward(sources=sources, source_lengths=source_lengths)
        return source_encodings, source_lengths


class FFNNTransducerDecoderOnly(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.predictor = FFNNTransducerPredictor(cfg.predictor_cfg)
        self.joiner = TransducerJoiner(cfg.joiner_cfg)

        if isinstance(cfg, FFNNTransducerWithIlmConfig):
            self.ilm_scale = cfg.ilm_scale
        else:
            self.ilm_scale = 0.0

    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, E]
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, C]
        device = source_encodings.device
        B = history.size(0)
        source_lengths = torch.tensor([1] * B, device=device)  # [B]
        target_lengths = torch.tensor([0] * B, device=device)  # [B]

        history_encoding, _, _ = self.predictor.forward(
            targets=torch.empty((B, 0), dtype=history.dtype, device=device),
            target_lengths=target_lengths,
            history=history,
        )  # [B, 1, P]

        source_encodings = source_encodings.unsqueeze(1)  # [B, 1, E]

        joint_output, _, _ = self.joiner.forward(
            source_encodings=source_encodings,
            source_lengths=source_lengths,  # [B]
            target_encodings=history_encoding,
            target_lengths=target_lengths,
        )  # [B, 1, 1, C]
        joint_output = joint_output.squeeze(2).squeeze(1)  # [B, C]

        if self.ilm_scale != 0:
            assert self.predictor.blank_id == 0
            joint_output_ilm, _, _ = self.joiner.forward(
                source_encodings=torch.zeros_like(source_encodings),
                source_lengths=source_lengths,
                target_encodings=history_encoding,
                target_lengths=target_lengths,
            )  # [B, 1, 1, C]
            joint_output_ilm = joint_output_ilm.squeeze(2).squeeze(1)  # [B, C]

            blank_log_probs = joint_output_ilm[:, :1]  # [B, 1]
            non_blank_log_probs = joint_output_ilm[:, 1:]  # [B, C-1]

            joint_output_ilm = torch.concat(
                [
                    torch.zeros_like(blank_log_probs),
                    non_blank_log_probs - torch.log(1.0 - torch.exp(blank_log_probs)),
                ],
                dim=-1,
            )  # [B, C]

            joint_output -= self.ilm_scale * joint_output_ilm

        return joint_output  # [B, C]


def get_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    loss_repo = CloneGitRepositoryJob("https://github.com/SimBe195/monotonic-rnnt.git").out_repository
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            ExternalImport(import_path=loss_repo),
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer.train_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
        ],
    )


def get_viterbi_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducerViterbi.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer.train_step_viterbi",
                import_as="train_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
        ],
    )


def get_align_restrict_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    loss_repo = CloneGitRepositoryJob(
        "https://github.com/SimBe195/monotonic-rnnt.git", commit="3fbb480107f7379347b25953be0727dcc4d0e57b"
    ).out_repository
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            ExternalImport(import_path=loss_repo),
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer.train_step_align_restrict",
                import_as="train_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
        ],
    )


def get_torchaudio_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer_torchaudio.train_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
        ],
    )


def get_k2_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer_torchaudio.train_step_k2",
                import_as="train_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
        ],
    )


def get_pruned_k2_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer_torchaudio.train_step_k2_pruned",
                import_as="train_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
        ],
    )


def get_encoder_recog_serializer(model_config: FFNNTransducerConfig, **_) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducerEncoderOnly.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.forward.transducer.encoder_forward_step", import_as="forward_step")
        ],
    )


def get_decoder_recog_serializer(model_config: FFNNTransducerConfig, **_) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducerDecoderOnly.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.forward.transducer.decoder_forward_step", import_as="forward_step")
        ],
    )


def get_beam_search_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    kwargs.setdefault("lexicon_file", CodeWrapper("lexicon_file"))
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.forward.transducer.monotonic_timesync_beam_search_forward_step",
                import_as="forward_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
            Import(f"{pytorch_package}.forward.search_callback.SearchCallback", import_as="forward_callback"),
        ],
    )


def get_default_config_v1(num_outputs: int) -> FFNNTransducerConfig:
    feature_extraction = ModuleFactoryV1(
        module_class=RasrCompatibleLogMelFeatureExtractionV1,
        cfg=RasrCompatibleLogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            min_amp=1.175494e-38,
            num_filters=80,
            alpha=0.97,
        ),
    )

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentByLengthModuleV1,
        cfg=SpecaugmentByLengthConfigV1(
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=16,
            freq_mask_max_size=5,
        ),
    )

    frontend = ModuleFactoryV1(
        VGG4LayerActFrontendV1,
        VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(2, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation=torch.nn.ReLU(),
            out_features=384,
        ),
    )

    ff_cfg = conformer_parts_i6.ConformerPositionwiseFeedForwardV1Config(
        input_dim=384,
        hidden_dim=1536,
        dropout=0.2,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = conformer_parts_i6.ConformerMHSAV1Config(
        input_dim=384,
        num_att_heads=6,
        att_weights_dropout=0.2,
        dropout=0.2,
    )

    conv_cfg = conformer_parts_i6.ConformerConvolutionV1Config(
        channels=384,
        kernel_size=31,
        dropout=0.2,
        activation=torch.nn.SiLU(),
        norm=torch.nn.BatchNorm1d(num_features=384, affine=False),
    )

    block_cfg = conformer_i6.ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = conformer_i6.ConformerEncoderV2Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    transcriber_cfg = TransducerTranscriberConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        encoder=ModuleFactoryV1(module_class=conformer_i6.ConformerEncoderV2, cfg=conformer_cfg),
        layer_size=384,
        target_size=num_outputs,
        enc_loss_layers=[5, 11],
    )

    predictor_cfg = FFNNTransducerPredictorConfig(
        layers=2,
        layer_size=640,
        activation=torch.nn.Tanh(),
        dropout=0.2,
        context_history_size=1,
        context_embedding_size=256,
        blank_id=0,
        target_size=num_outputs,
    )

    joiner_cfg = TransducerJoinerConfig(
        input_size=1024,
        layer_size=1024,
        act=torch.nn.Tanh(),
        target_size=num_outputs,
        combination_mode=CombinationMode.CONCAT,
    )

    return FFNNTransducerConfig(
        transcriber_cfg=transcriber_cfg,
        predictor_cfg=predictor_cfg,
        joiner_cfg=joiner_cfg,
    )


def get_default_config_v2(num_outputs: int) -> FFNNTransducerConfig:
    feature_extraction = ModuleFactoryV1(
        module_class=RasrCompatibleLogMelFeatureExtractionV1,
        cfg=RasrCompatibleLogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            min_amp=1.175494e-38,
            num_filters=80,
            alpha=0.97,
        ),
    )

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentByLengthModuleV1,
        cfg=SpecaugmentByLengthConfigV1(
            time_min_num_masks=2,
            time_max_mask_per_n_frames=20,
            time_mask_max_size=30,
            freq_min_num_masks=2,
            freq_max_num_masks=16,
            freq_mask_max_size=5,
        ),
    )

    frontend = ModuleFactoryV1(
        VGG4LayerActFrontendV1,
        VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(2, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation=torch.nn.ReLU(),
            out_features=768,
        ),
    )

    ff_cfg = conformer_parts_i6.ConformerPositionwiseFeedForwardV1Config(
        input_dim=768,
        hidden_dim=3072,
        dropout=0.3,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = conformer_parts_i6.ConformerMHSAV1Config(
        input_dim=768,
        num_att_heads=8,
        att_weights_dropout=0.3,
        dropout=0.3,
    )

    conv_cfg = conformer_parts_i6.ConformerConvolutionV1Config(
        channels=768,
        kernel_size=7,
        dropout=0.3,
        activation=torch.nn.SiLU(),
        norm=torch.nn.BatchNorm1d(num_features=768, affine=False),
    )

    block_cfg = conformer_i6.ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "conv", "mhsa", "ff"],
    )

    conformer_cfg = conformer_i6.ConformerEncoderV2Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    transcriber_cfg = TransducerTranscriberConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        encoder=ModuleFactoryV1(module_class=conformer_i6.ConformerEncoderV2, cfg=conformer_cfg),
        layer_size=768,
        target_size=num_outputs,
        enc_loss_layers=[5, 11],
    )

    predictor_cfg = FFNNTransducerPredictorConfig(
        layers=2,
        layer_size=384,
        activation=torch.nn.Tanh(),
        dropout=0.3,
        context_history_size=1,
        context_embedding_size=256,
        blank_id=0,
        target_size=num_outputs,
    )

    joiner_cfg = TransducerJoinerConfig(
        input_size=1152,
        layer_size=1024,
        act=torch.nn.Tanh(),
        target_size=num_outputs,
        combination_mode=CombinationMode.CONCAT,
    )

    return FFNNTransducerConfig(
        transcriber_cfg=transcriber_cfg,
        predictor_cfg=predictor_cfg,
        joiner_cfg=joiner_cfg,
    )


def get_default_config_v3(num_outputs: int) -> FFNNTransducerConfig:
    feature_extraction = ModuleFactoryV1(
        module_class=RasrCompatibleLogMelFeatureExtractionV1,
        cfg=RasrCompatibleLogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            min_amp=1.175494e-38,
            num_filters=80,
            alpha=0.97,
        ),
    )

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentByLengthModuleV1,
        cfg=SpecaugmentByLengthConfigV1(
            time_min_num_masks=2,
            time_max_mask_per_n_frames=35,
            time_mask_max_size=25,
            freq_min_num_masks=2,
            freq_max_num_masks=8,
            freq_mask_max_size=10,
        ),
    )

    frontend = ModuleFactoryV1(
        VGG4LayerActFrontendV1,
        VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(2, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation=torch.nn.ReLU(),
            out_features=512,
        ),
    )

    ff_cfg = conformer_parts_i6.ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.3,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = conformer_parts_i6.ConformerMHSAV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.3,
        dropout=0.3,
    )

    conv_cfg = conformer_parts_i6.ConformerConvolutionV1Config(
        channels=512,
        kernel_size=7,
        dropout=0.3,
        activation=torch.nn.SiLU(),
        norm=torch.nn.BatchNorm1d(num_features=512, affine=False),
    )

    block_cfg = conformer_i6.ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "conv", "mhsa", "ff"],
    )

    conformer_cfg = conformer_i6.ConformerEncoderV2Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    transcriber_cfg = TransducerTranscriberConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        encoder=ModuleFactoryV1(module_class=conformer_i6.ConformerEncoderV2, cfg=conformer_cfg),
        layer_size=512,
        target_size=num_outputs,
        enc_loss_layers=[5, 11],
    )

    predictor_cfg = FFNNTransducerPredictorConfig(
        layers=2,
        layer_size=384,
        activation=torch.nn.Tanh(),
        dropout=0.3,
        context_history_size=1,
        context_embedding_size=256,
        blank_id=0,
        target_size=num_outputs,
    )

    joiner_cfg = TransducerJoinerConfig(
        input_size=896,
        layer_size=1024,
        act=torch.nn.Tanh(),
        target_size=num_outputs,
        combination_mode=CombinationMode.CONCAT,
    )

    return FFNNTransducerConfig(
        transcriber_cfg=transcriber_cfg,
        predictor_cfg=predictor_cfg,
        joiner_cfg=joiner_cfg,
    )
