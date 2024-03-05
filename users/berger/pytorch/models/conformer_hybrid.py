from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.assemblies.conformer import (
    ConformerEncoderV1,
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from .util import lengths_to_padding_mask

from ..custom_parts import specaugment


@dataclass
class ConformerHybridConfig(ModelConfiguration):
    specaugment_cfg: specaugment.SpecaugmentConfigV1
    conformer_cfg: ConformerEncoderV1Config
    target_size: int


class ConformerHybridModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerHybridConfig, **kwargs):
        super().__init__()
        self.specaugment = specaugment.SpecaugmentModuleV1(step=step, cfg=cfg.specaugment_cfg)
        self.conformer = ConformerEncoderV1(cfg.conformer_cfg)
        self.final_linear = torch.nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size)

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        x = self.specaugment(audio_features)  # [B, T, F]
        sequence_mask = lengths_to_padding_mask(audio_features_len)
        x, sequence_mask = self.conformer(x, sequence_mask)  # [B, T, F]
        logits = self.final_linear(x)  # [B, T, F]

        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, F]

        if self.training:
            return log_probs, sequence_mask
        return log_probs


def get_train_serializer(
    model_config: ConformerHybridConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.train_steps.hybrid_viterbi.train_step"),
        ],
    )


def get_prior_serializer(
    model_config: ConformerHybridConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.forward.basic.forward_step"),
            Import(f"{pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"),
        ],
    )


def get_recog_serializer(
    model_config: ConformerHybridConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.export.ctc.export"),
        ],
    )


def get_serializer(model_config: ConformerHybridConfig, variant: ConfigVariant) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def get_default_config_v1(num_inputs: int, num_outputs: int) -> ConformerHybridConfig:
    specaugment_cfg = specaugment.SpecaugmentConfigV1(
        max_time_mask_num=1,
        max_time_mask_size=15,
        max_feature_mask_num=num_inputs // 10,
        max_feature_mask_size=5,
        increase_steps=[2000],
    )

    frontend_cfg = VGG4LayerActFrontendV1Config(
        in_features=num_inputs,
        conv1_channels=32,
        conv2_channels=32,
        conv3_channels=64,
        conv4_channels=64,
        conv_kernel_size=3,
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
        norm=torch.nn.BatchNorm1d(num_features=512, affine=False),
    )

    block_cfg = ConformerBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = ConformerEncoderV1Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return ConformerHybridConfig(
        specaugment_cfg=specaugment_cfg,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
    )
