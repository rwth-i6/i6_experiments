from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union, Callable, Tuple

import torch
from torchaudio.models import rnnt

from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
import i6_models.parts.conformer as conformer_parts_i6
import i6_models.assemblies.conformer as conformer_i6
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from .util import lengths_to_padding_mask

from ..custom_parts.specaugment import (
    SpecaugmentConfigV1,
    SpecaugmentModuleV1,
)


@dataclass
class TransducerTranscriberConfig(ModelConfiguration):
    feature_extraction: ModuleFactoryV1
    specaugment: ModuleFactoryV1
    encoder: ModuleFactoryV1
    dim: int
    target_size: int


class TransducerTranscriber(rnnt._Transcriber, torch.nn.Module):
    def __init__(self, cfg: TransducerTranscriberConfig, **_) -> None:
        super().__init__()
        self.feature_extraction = cfg.feature_extraction()
        self.specaugment = cfg.specaugment()
        self.encoder = cfg.encoder()
        self.final_linear = torch.nn.Linear(cfg.dim, cfg.target_size)

    def forward(
        self,
        input: torch.Tensor,  # [B, T, F]
        lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
        with torch.no_grad():
            input = input.squeeze(-1)
            x, lengths = self.feature_extraction(input, lengths)
            sequence_mask = lengths_to_padding_mask(lengths)

            x = self.specaugment(x)  # [B, T, F]

        x, sequence_mask = self.encoder(x, sequence_mask)  # [B, T, E], [B, T]
        x = self.final_linear(x)  # [B, T, C]

        return x, torch.sum(sequence_mask, dim=1).to(torch.int32)  # [B, T, C], [B]

    def infer(
        self,
        input: torch.Tensor,  # [B, T, F],
        lengths: torch.Tensor,  # [B]
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise NotImplementedError


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
        self.embedding = torch.nn.Embedding(num_embeddings=cfg.target_size, embedding_dim=cfg.context_embedding_size)

        self.blank_id = cfg.blank_id

        self.context_history_size = cfg.context_history_size
        prediction_layers = []
        prev_size = self.context_history_size * cfg.context_embedding_size
        for _ in range(cfg.layers):
            prediction_layers.append(torch.nn.Dropout(cfg.dropout))
            prediction_layers.append(torch.nn.Linear(prev_size, cfg.layer_size))
            prediction_layers.append(cfg.activation)
            prev_size = cfg.layer_size

        self.network = torch.nn.Sequential(
            *prediction_layers, torch.nn.Dropout(cfg.dropout), torch.nn.Linear(prev_size, cfg.target_size)
        )

    def forward(
        self,
        input: torch.Tensor,  # [B, S],
        lengths: torch.Tensor,  # [B],
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
                (input.size(0), self.context_history_size - 1),
                fill_value=self.blank_id,
                dtype=input.dtype,
                device=input.device,
            )  # [B, H-1]
            # print("Predictor received no state. Use", prepend.deeper())
        else:
            prepend = state[0][0]  # [B, H-1]
            # print("Predictor received state", prepend.deeper())
        extended_input = torch.concat([prepend, input], dim=1)  # [B, S+H-1]
        # print("extended input", extended_input.deeper())

        if self.context_history_size > 1:
            return_state = extended_input[:, -(self.context_history_size - 1) :]  # [B, H-1]
        else:
            return_state = torch.empty(
                size=(input.size(0), 0), dtype=input.dtype, device=input.device
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
        return a, lengths, [[return_state]]


class CombinationMode(Enum):
    CONCAT = auto()
    SUM = auto()


@dataclass
class TransducerJoinerConfig(ModelConfiguration):
    layer_size: int
    act: torch.nn.Module
    target_size: int
    combination_mode: CombinationMode


class TransducerJoiner(torch.nn.Module):
    def __init__(self, cfg: TransducerJoinerConfig, **_) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(cfg.target_size, cfg.layer_size),
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
            torch.nn.Linear(cfg.target_size, cfg.layer_size),
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


class FFNNTransducer(rnnt.RNNT):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__(
            transcriber=TransducerTranscriber(cfg.transcriber_cfg),
            predictor=FFNNTransducerPredictor(cfg.predictor_cfg),
            joiner=PackedTransducerJoiner(cfg.joiner_cfg),
        )


class FFNNTransducerEncoderOnly(rnnt.RNNT):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__(
            transcriber=TransducerTranscriber(cfg.transcriber_cfg),
            predictor=None,
            joiner=None,
        )

    def forward(
        self,
        features: torch.Tensor,  # [B, T, F]
        features_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T', E]
        return self.transcribe(sources=features, source_lengths=features_size)


class FFNNTransducerDecoderOnly(rnnt.RNNT):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__(
            transcriber=None,
            predictor=FFNNTransducerPredictor(cfg.predictor_cfg),
            joiner=TransducerJoiner(cfg.joiner_cfg),
        )

    def forward(
        self,
        encoder: torch.Tensor,  # [B, E]
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, C]
        dec_history_state = [[history[:, :-1]]]  # [[[B, H-1]]]
        dec_current_label = history[:, -1:]  # [B, 1]
        dec_length = torch.tensor([1] * history.size(0), device=history.device)  # [B]

        decoder, _, _ = self.predict(
            targets=dec_current_label, target_lengths=dec_length, state=dec_history_state
        )  # [B, 1, P]

        encoder = encoder.unsqueeze(1)  # [B, 1, E]

        joint_output, _, _ = self.join(
            source_encodings=encoder, source_lengths=dec_length, target_encodings=decoder, target_lengths=dec_length
        )  # [B, 1, 1, C]

        return joint_output.squeeze((1, 2))  # [B, C]


def get_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
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
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducerEncoderOnly.__name__}",
        model_config=model_config,
        additional_serializer_objects=[Import(f"{pytorch_package}.forward.transducer.encoder_forward_step")],
    )


def get_decoder_recog_serializer(model_config: FFNNTransducerConfig, **_) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducerDecoderOnly.__name__}",
        model_config=model_config,
        additional_serializer_objects=[Import(f"{pytorch_package}.forward.transducer.decoder_forward_step")],
    )


def get_beam_search_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
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
        module_class=LogMelFeatureExtractionV1,
        cfg=LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
            n_fft=400,
        ),
    )

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentModuleV1,
        cfg=SpecaugmentConfigV1(
            time_min_num_masks=1,
            time_max_num_masks=1,
            time_mask_max_size=15,
            freq_min_num_masks=1,
            freq_max_num_masks=8,
            freq_mask_max_size=5,
        ),
    )

    frontend_cfg = VGG4LayerActFrontendV1Config(
        in_features=80,
        conv1_channels=32,
        conv2_channels=32,
        conv3_channels=64,
        conv4_channels=64,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 2),
        pool1_stride=None,
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=None,
        pool2_padding=None,
        activation=torch.nn.SiLU(),
        out_features=512,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = conformer_parts_i6.ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = conformer_parts_i6.ConformerMHSAV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    conv_cfg = conformer_parts_i6.ConformerConvolutionV1Config(
        channels=512,
        kernel_size=31,
        dropout=0.1,
        activation=torch.nn.SiLU(),
        norm=torch.nn.BatchNorm1d(num_features=512, affine=False),
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
        dim=512,
        target_size=num_outputs,
    )

    predictor_cfg = FFNNTransducerPredictorConfig(
        layers=2,
        layer_size=640,
        activation=torch.nn.Tanh(),
        dropout=0.1,
        context_history_size=1,
        context_embedding_size=256,
        target_size=num_outputs,
        blank_id=0,
    )

    joiner_cfg = TransducerJoinerConfig(
        layer_size=1024,
        act=torch.nn.Tanh(),
        target_size=num_outputs,
        combination_mode=CombinationMode.SUM,
    )

    return FFNNTransducerConfig(
        transcriber_cfg=transcriber_cfg,
        predictor_cfg=predictor_cfg,
        joiner_cfg=joiner_cfg,
    )
