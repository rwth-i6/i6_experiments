from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Tuple

import torch
from torchaudio.models import rnnt
from torchaudio.prototype.models.rnnt import conformer_rnnt_model

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
import i6_models.parts.conformer as conformer_parts_i6
import i6_models.assemblies.conformer as conformer_i6
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from .util import lengths_to_padding_mask

from ..custom_parts.specaugment import (
    SpecaugmentConfigV1,
    SpecaugmentModuleV1,
)


@dataclass
class TransducerTranscriberConfig(ModelConfiguration):
    specaugment: ModuleFactoryV1
    conformer: ModuleFactoryV1
    dim: int
    target_size: int


class TransducerTranscriber(rnnt._Transcriber, torch.nn.Module):
    def __init__(self, cfg: TransducerTranscriberConfig, **_) -> None:
        super().__init__()
        self.specaugment = cfg.specaugment()
        self.conformer = cfg.conformer()
        self.final_linear = torch.nn.Linear(cfg.dim, cfg.target_size)

    def forward(
        self,
        input: torch.Tensor,  # [B, T, F]
        lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
        with torch.no_grad():
            assert lengths is not None
            sequence_mask = lengths_to_padding_mask(lengths)
            sequence_mask = torch.nn.functional.pad(sequence_mask, (0, input.size(1) - sequence_mask.size(1)), value=0)

            x = self.specaugment(input)  # [B, T, F]

        x, sequence_mask = self.conformer(x, sequence_mask)  # [B, T, E], [B, T]

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
    act: Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]
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
            prediction_layers.append(cfg.act)
            prev_size = cfg.layer_size

        self.network = torch.nn.Sequential(*prediction_layers, torch.nn.Linear(prev_size, cfg.target_size))

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
        # print("Predictor received input", input[:5].deeper())
        if state is None:
            prepend = torch.full(
                (input.size(0), self.context_history_size - 1),
                fill_value=self.blank_id,
                dtype=input.dtype,
                device=input.device,
            )  # [B, H-1]
            # print("Predictor received no state. Use", prepend[:5].deeper())
        else:
            prepend = state[0][0]  # [B, H-1]
            # print("Predictor received state", prepend[:5].deeper())
        extended_input = torch.concat([prepend, input], dim=1)  # [B, S+H-1]
        # print("extended input", extended_input[:5].deeper())

        if self.context_history_size > 1:
            new_state = extended_input[:, -self.context_history_size + 1 :]  # [B, H-1]
        else:
            new_state = torch.empty(
                size=(input.size(0), 0), dtype=input.dtype, device=input.device
            )  # [B, 0] = [B, H-1]
        # print("New state is ", new_state[:5].deeper())

        context = torch.stack(
            [
                extended_input[:, self.context_history_size - i - 1 : (-i if i != 0 else None)]  # [B, S]
                for i in range(self.context_history_size - 1, -1, -1)
            ],
            dim=-1,
        )  # [B, S, H]

        # print("Predict logits based on context", context[:5])
        a = self.embedding(context)  # [B, S, H, E]
        a = torch.reshape(a, shape=[*(a.shape[:-2]), a.shape[-2] * a.shape[-1]])  # [B, S, H*E]
        a = self.network(a)  # [B, S, P]
        topk = torch.topk(torch.nn.functional.softmax(a, dim=-1), k=4)
        # print("Result probabilities:")
        # print(topk.indices[:5].deeper())
        # print(topk.values[:5].deeper())
        # print(
        #     "Repeat probabilities:",
        #     torch.gather(torch.nn.functional.softmax(a, dim=-1), dim=-1, index=context[:, :, -2:-1])[:5].deeper(),
        # )
        return a, lengths, [[new_state]]


@dataclass
class TransducerJoinerConfig(ModelConfiguration):
    layer_size: int
    act: torch.nn.Module
    target_size: int


class TransducerJoiner(torch.nn.Module):
    def __init__(self, cfg: TransducerJoinerConfig, **_) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(cfg.target_size, cfg.layer_size),
            cfg.act,
            torch.nn.Linear(cfg.layer_size, cfg.target_size),
        )

    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, T, C],
        source_lengths: torch.Tensor,  # [B],
        target_encodings: torch.Tensor,  # [B, S, C],
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, T, S, C], [B], [B],
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        output = self.network(joint_encodings)
        return output, source_lengths, target_lengths

    def forward_samelength(
        self,
        source_encodings: torch.Tensor,  # [B, T, C],
        target_encodings: torch.Tensor,  # [B, T, C],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_encodings = source_encodings + target_encodings
        output = self.network(joint_encodings)
        return output


@dataclass
class FFNNTransducerConfig(ModelConfiguration):
    transcriber_cfg: TransducerTranscriberConfig
    predictor_cfg: FFNNTransducerPredictorConfig
    joiner_cfg: TransducerJoinerConfig


class FFNNTransducer(rnnt.RNNT):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__(
            transcriber=TransducerTranscriber(cfg.transcriber_cfg),
            predictor=FFNNTransducerPredictor(cfg.predictor_cfg),  # type: ignore
            joiner=TransducerJoiner(cfg.joiner_cfg),  # type: ignore
        )


def get_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
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


def get_beam_search_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.forward.transducer_torchaudio.beam_search_forward_step",
                import_as="forward_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
            Import(f"{pytorch_package}.forward.search_callback.SearchCallback", import_as="forward_callback"),
        ],
    )


def get_default_config_v1(num_inputs: int, num_outputs: int) -> FFNNTransducerConfig:
    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentModuleV1,
        cfg=SpecaugmentConfigV1(
            time_min_num_masks=1,
            time_max_num_masks=1,
            time_mask_max_size=15,
            freq_min_num_masks=1,
            freq_max_num_masks=num_inputs // 10,
            freq_mask_max_size=5,
        ),
    )

    frontend_cfg = VGG4LayerActFrontendV1Config(
        in_features=num_inputs,
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
        specaugment=specaugment,
        conformer=ModuleFactoryV1(module_class=conformer_i6.ConformerEncoderV1, cfg=conformer_cfg),
        dim=512,
        target_size=num_outputs,
    )

    predictor_cfg = FFNNTransducerPredictorConfig(
        layers=2,
        layer_size=640,
        act=torch.nn.Tanh(),
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
    )

    return FFNNTransducerConfig(
        transcriber_cfg=transcriber_cfg,
        predictor_cfg=predictor_cfg,
        joiner_cfg=joiner_cfg,
    )
