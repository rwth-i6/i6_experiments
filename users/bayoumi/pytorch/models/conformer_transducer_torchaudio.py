from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torchaudio.models.rnnt import RNNT

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1

from ..custom_parts.specaugment import (
    SpecaugmentConfigV1,
    SpecaugmentModuleV1,
)


@dataclass
class TorchaudioConformerTransducerConfigV1(ModelConfiguration):
    specaugment: ModuleFactoryV1
    input_dim: int
    encoding_dim: int
    time_reduction_stride: int
    conformer_input_dim: int
    conformer_ffn_dim: int
    conformer_num_layers: int
    conformer_num_heads: int
    conformer_depthwise_conv_kernel_size: int
    conformer_dropout: float
    num_symbols: int
    symbol_embedding_dim: int
    num_lstm_layers: int
    lstm_hidden_dim: int
    lstm_layer_norm: bool
    lstm_layer_norm_epsilon: float
    lstm_dropout: float
    joiner_activation: str


class TorchaudioConformerTransducerV1(RNNT):
    def __init__(self, step: int, cfg: TorchaudioConformerTransducerConfigV1, **_) -> None:
        from torchaudio.prototype.models.rnnt import _ConformerEncoder, _Predictor, _Joiner

        class Joiner(_Joiner):
            def forward_samelength(
                self,
                source_encodings: torch.Tensor,
                target_encodings: torch.Tensor,
            ) -> torch.Tensor:
                joint_encodings = source_encodings + target_encodings
                activation_out = self.activation(joint_encodings)
                output = self.linear(activation_out)
                return output

        encoder = _ConformerEncoder(
            input_dim=cfg.input_dim,
            output_dim=cfg.encoding_dim,
            time_reduction_stride=cfg.time_reduction_stride,
            conformer_input_dim=cfg.conformer_input_dim,
            conformer_ffn_dim=cfg.conformer_ffn_dim,
            conformer_num_layers=cfg.conformer_num_layers,
            conformer_num_heads=cfg.conformer_num_heads,
            conformer_depthwise_conv_kernel_size=cfg.conformer_depthwise_conv_kernel_size,
            conformer_dropout=cfg.conformer_dropout,
        )

        predictor = _Predictor(
            num_symbols=cfg.num_symbols,
            output_dim=cfg.encoding_dim,
            symbol_embedding_dim=cfg.symbol_embedding_dim,
            num_lstm_layers=cfg.num_lstm_layers,
            lstm_hidden_dim=cfg.lstm_hidden_dim,
            lstm_layer_norm=cfg.lstm_layer_norm,
            lstm_layer_norm_epsilon=cfg.lstm_layer_norm_epsilon,
            lstm_dropout=cfg.lstm_dropout,
        )

        joiner = Joiner(
            input_dim=cfg.encoding_dim,
            output_dim=cfg.num_symbols,
            activation=cfg.joiner_activation,
        )

        super().__init__(encoder, predictor, joiner)

        self.specaug = cfg.specaugment()

    def forward(self, sources: torch.Tensor, *args, **kwargs):
        sources = self.specaug(sources)
        return super().forward(sources, *args, **kwargs)


def get_train_serializer(model_config: TorchaudioConformerTransducerConfigV1, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{TorchaudioConformerTransducerV1.__name__}",
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


def get_k2_train_serializer(model_config: TorchaudioConformerTransducerConfigV1, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{TorchaudioConformerTransducerV1.__name__}",
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


def get_pruned_k2_train_serializer(model_config: TorchaudioConformerTransducerConfigV1, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{TorchaudioConformerTransducerV1.__name__}",
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


def get_beam_search_serializer(model_config: TorchaudioConformerTransducerConfigV1, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{TorchaudioConformerTransducerV1.__name__}",
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


def get_torchaudio_default_config_v1(num_inputs: int, num_outputs: int) -> TorchaudioConformerTransducerConfigV1:
    # source: https://pytorch.org/audio/main/_modules/torchaudio/prototype/models/rnnt.html#conformer_rnnt_base

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentModuleV1,
        cfg=SpecaugmentConfigV1(
            time_min_num_masks=0,
            time_max_num_masks=0,
            time_mask_max_size=15,
            freq_min_num_masks=0,
            freq_max_num_masks=0,  # num_inputs // 10,
            freq_mask_max_size=5,
        ),
    )

    return TorchaudioConformerTransducerConfigV1(
        specaugment=specaugment,
        input_dim=num_inputs,
        encoding_dim=num_outputs,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=num_outputs,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-05,
        lstm_dropout=0.3,
        joiner_activation='"tanh"',
    )


def get_i6_default_config_v1(num_inputs: int, num_outputs: int) -> TorchaudioConformerTransducerConfigV1:
    # source: https://pytorch.org/audio/main/_modules/torchaudio/prototype/models/rnnt.html#conformer_rnnt_base

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

    return TorchaudioConformerTransducerConfigV1(
        specaugment=specaugment,
        input_dim=num_inputs,
        encoding_dim=num_outputs,
        time_reduction_stride=4,
        conformer_input_dim=512,
        conformer_ffn_dim=2048,
        conformer_num_layers=12,
        conformer_num_heads=8,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=num_outputs,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-05,
        lstm_dropout=0.3,
        joiner_activation='"tanh"',
    )
