from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Union

import i6_models.assemblies.conformer as conformer_i6
import i6_models.parts.conformer as conformer_parts_i6
import torch
from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
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


class TransducerTranscriber(torch.nn.Module):
    def __init__(self, cfg: TransducerTranscriberConfig, **_) -> None:
        super().__init__()
        self.feature_extraction = cfg.feature_extraction()
        self.specaugment = cfg.specaugment()
        self.encoder = cfg.encoder()

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
        with torch.no_grad():
            sources = sources.squeeze(-1)
            x, source_lengths = self.feature_extraction(sources, source_lengths)
            print("Features: ", x[0, :3, :5])
            sequence_mask = lengths_to_padding_mask(source_lengths)

            x = self.specaugment(x)  # [B, T, F]

        x, sequence_mask = self.encoder(x, sequence_mask)  # [B, T, E], [B, T]

        return x, torch.sum(sequence_mask, dim=1).to(torch.int32)  # [B, T, C], [B]


class TransducerTranscriberNoFeatExtr(torch.nn.Module):
    def __init__(self, cfg: TransducerTranscriberConfig, **_) -> None:
        super().__init__()
        self.specaugment = cfg.specaugment()
        self.encoder = cfg.encoder()

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
        with torch.no_grad():
            sequence_mask = lengths_to_padding_mask(source_lengths)

            x = self.specaugment(sources)  # [B, T, F]

        x, sequence_mask = self.encoder(x, sequence_mask)  # [B, T, E], [B, T]

        return x, torch.sum(sequence_mask, dim=1).to(torch.int32)  # [B, T, C], [B]


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
        state: Optional[
            List[List[torch.Tensor]]
        ] = None,  # Most recently fed inputs, used for higher order context, shape [[[B, H-1]]]; list of lists for compatibility with torchaudio
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:  # [B, S, C], [B], [[[B, H-1]]]
        # print("")
        # print("Enter predictor forward")
        # print("Predictor received input", input.deeper())

        # extend input by prepending either the state if it's given or some history consisting of blanks
        if state is None:
            prepend = torch.full(
                (targets.size(0), self.context_history_size - 1),
                fill_value=self.blank_id,
                dtype=targets.dtype,
                device=targets.device,
            )  # [B, H-1]
            # print("Predictor received no state. Use", prepend.deeper())
        else:
            prepend = state[0][0]  # [B, H-1]
            # print("Predictor received state", prepend.deeper())
        extended_input = torch.concat([prepend, targets], dim=1)  # [B, S+H-1]
        # print("extended input", extended_input.deeper())

        if self.context_history_size > 1:
            return_state = extended_input[:, -(self.context_history_size - 1) :]  # [B, H-1]
        else:
            return_state = torch.empty(
                size=(targets.size(0), 0), dtype=targets.dtype, device=targets.device
            )  # [B, 0] = [B, H-1]
        # print("New state is ", return_state.deeper())

        # Build context at each position by shifting and cutting label sequence.
        # E.g. for history size 2 and label sequence a_1, ..., a_S we have context
        # a_2 a_3 a_4 ... a_S
        # a_1 a_2 a_3 ... a_{S-1}
        context = torch.stack(
            [
                extended_input[:, self.context_history_size - 1 - i : (-i if i != 0 else None)]  # [B, S]
                for i in range(self.context_history_size)
            ],
            dim=-1,
        )  # [B, S, H]

        # print("Predict based on context", context)
        a = self.embedding(context)  # [B, S, H, E]
        a = torch.reshape(a, shape=[*(a.shape[:-2]), a.shape[-2] * a.shape[-1]])  # [B, S, H*E]
        a = self.network(a)  # [B, S, P]
        # topk = torch.topk(torch.nn.functional.softmax(a, dim=-1), k=4)
        # print("Result probabilities:")
        # print(topk.indices.deeper())
        # print(topk.values.deeper())
        # print(
        #     "Repeat probabilities:",
        #     torch.gather(torch.nn.functional.softmax(a, dim=-1), dim=-1, index=context[:, :, -2:-1]).deeper(),
        # )
        return a, target_lengths, [[return_state]]


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
        source_encodings: torch.Tensor,  # [B, T, C_1],
        source_lengths: torch.Tensor,  # [B],
        target_encodings: torch.Tensor,  # [B, S, C_2],
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, T, S, F], [B], [B],
        source_encodings = source_encodings.unsqueeze(2).expand(
            target_encodings.size(0), -1, target_encodings.size(1), -1
        )  # [B, T, S, C_1]
        target_encodings = target_encodings.unsqueeze(1).expand(-1, source_encodings.size(1), -1, -1)  # [B, T, S, C_2]

        if self.combination_mode == CombinationMode.CONCAT:
            joiner_inputs = torch.concat([source_encodings, target_encodings], dim=-1)  # [B, T, S, C_1 + C_2]
        elif self.combination_mode == CombinationMode.SUM:
            joiner_inputs = source_encodings + target_encodings  # [B, T, S, C_1=C_2]

        output = self.network(joiner_inputs)  # [B, T, S, C]

        if not self.training:
            output = torch.log_softmax(output, dim=-1)

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
        source_encodings: torch.Tensor,  # [B, T, C_1],
        source_lengths: torch.Tensor,  # [B],
        target_encodings: torch.Tensor,  # [B, S, C_2],
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [T_1 * S_1 + T_2 * S_2 + ... + T_B * S_B, F], [B], [B],
        batch_tensors = []

        for b in range(source_encodings.size(0)):
            valid_source = source_encodings[b, : source_lengths[b], :]  # [T_b, C_1]
            valid_target = target_encodings[b, : target_lengths[b], :]  # [S_b, C_2]

            expanded_source = valid_source.unsqueeze(1).expand(-1, int(target_lengths[b].item()), -1)  # [T_b, S_b, C_1]
            expanded_target = valid_target.unsqueeze(0).expand(int(source_lengths[b].item()), -1, -1)  # [T_b, S_b, C_2]

            if self.combination_mode == CombinationMode.CONCAT:
                combination = torch.concat([expanded_source, expanded_target], dim=-1)  # [T_b, S_b, C_1 + C_2]
            elif self.combination_mode == CombinationMode.SUM:
                combination = expanded_source + expanded_target  # [T_b, S_b, C_1 (=C_2)]
            else:
                raise NotImplementedError

            batch_tensors.append(combination.reshape(-1, combination.size(2)))  # [T_b * S_b, C']

        joint_encodings = torch.concat(batch_tensors, dim=0)  # [T_1 * S_1 + T_2 * S_2 + ... + T_B * S_B, C']
        output = self.network(joint_encodings)  # [T_1 * S_1 + T_2 * S_2 + ... + T_B * S_B, F]

        if not self.training:
            output = torch.log_softmax(output, dim=-1)

        return output, source_lengths, target_lengths


@dataclass
class FFNNTransducerConfig(ModelConfiguration):
    transcriber_cfg: TransducerTranscriberConfig
    predictor_cfg: FFNNTransducerPredictorConfig
    joiner_cfg: TransducerJoinerConfig


class FFNNTransducer(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.transcriber = TransducerTranscriber(cfg.transcriber_cfg)
        self.predictor = FFNNTransducerPredictor(cfg.predictor_cfg)
        self.joiner = PackedTransducerJoiner(cfg.joiner_cfg)

    def transcribe(self, sources: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transcriber(sources=sources, source_lengths=source_lengths)

    def predict(
        self, targets: torch.Tensor, target_lengths: torch.Tensor, state: Optional[List[List[torch.Tensor]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        return self.predictor(targets=targets, target_lengths=target_lengths, state=state)

    def join(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        source_encodings, source_lengths = self.transcribe(sources=sources, source_lengths=source_lengths)
        target_encodings, target_lengths, state = self.predict(
            targets=targets, target_lengths=target_lengths, state=predictor_state
        )

        output, source_lengths, target_lengths = self.join(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

        return output, source_lengths, target_lengths, state


class FFNNTransducerEncoderOnly(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.transcriber = TransducerTranscriberNoFeatExtr(cfg.transcriber_cfg)

    def forward(
        self,
        sources: torch.Tensor,  # [B, T, F]
        source_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T', E]
        return self.transcriber(sources=sources, source_lengths=source_lengths)


class FFNNTransducerDecoderOnly(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.predictor = FFNNTransducerPredictor(cfg.predictor_cfg)
        self.joiner = TransducerJoiner(cfg.joiner_cfg)

    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, E]
        targets: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, C]
        dec_state = [[targets[:, :-1]]]  # [[[B, H-1]]]
        dec_current_label = targets[:, -1:]  # [B, 1]
        dec_length = torch.tensor([1] * targets.size(0), device=targets.device)  # [B]

        decoder, _, _ = self.predictor(
            targets=dec_current_label, target_lengths=dec_length, state=dec_state
        )  # [B, 1, P]

        source_encodings = source_encodings.unsqueeze(1)  # [B, 1, E]

        joint_output, _, _ = self.joiner(
            source_encodings=source_encodings,
            source_lengths=dec_length,
            target_encodings=decoder,
            target_lengths=dec_length,
        )  # [B, 1, 1, C]

        return joint_output.squeeze(2).squeeze(1)  # [B, C]


def get_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.train_steps.transducer.train_step",
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
        GenericFrontendV1,
        GenericFrontendV1Config(
            in_features=80,
            layer_ordering=[
                FrontendLayerType.Conv2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Pool2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Pool2d,
                FrontendLayerType.Activation,
            ],
            conv_kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],
            conv_paddings=None,
            conv_out_dims=[32, 64, 64, 32],
            conv_strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
            pool_kernel_sizes=[(2, 1), (2, 1)],
            pool_strides=None,
            pool_paddings=None,
            activations=[torch.nn.ReLU()],
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

    block_cfg = conformer_i6.ConformerBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = conformer_i6.ConformerEncoderV1Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    transcriber_cfg = TransducerTranscriberConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        encoder=ModuleFactoryV1(module_class=conformer_i6.ConformerEncoderV1, cfg=conformer_cfg),
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
