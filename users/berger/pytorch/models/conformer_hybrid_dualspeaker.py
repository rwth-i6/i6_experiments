from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.parts.frontend.vgg_act import (
    VGG4LayerActFrontendV1,
    VGG4LayerActFrontendV1Config,
)
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.assemblies.conformer import (
    ConformerBlockV1Config,
    ConformerBlockV1,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from .util import lengths_to_padding_mask

from ..custom_parts import specaugment


@dataclass
class ConformerHybridDualSpeakerConfig(ModelConfiguration):
    specaugment_cfg: specaugment.SpecaugmentConfigV1

    primary_frontend: ModuleFactoryV1
    secondary_frontend: Optional[ModuleFactoryV1]

    conformer_block_cfg: ConformerBlockV1Config

    num_primary_layers: int
    num_secondary_layers: int
    num_mixture_aware_speaker_layers: int

    use_secondary_audio: bool

    target_size: int


class ConformerHybridDualSpeakerModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerHybridDualSpeakerConfig, **kwargs):
        super().__init__()

        base_dim = cfg.conformer_block_cfg.ff_cfg.input_dim

        self.use_secondary_audio = cfg.use_secondary_audio

        self.specaugment = specaugment.SpecaugmentModuleV1(step=step, cfg=cfg.specaugment_cfg)

        self.primary_frontend = cfg.primary_frontend()
        if cfg.secondary_frontend is None:
            self.secondary_frontend = cfg.primary_frontend()
        else:
            self.secondary_frontend = cfg.secondary_frontend()

        self.primary_encoder = torch.nn.ModuleList(
            [ConformerBlockV1(cfg.conformer_block_cfg) for _ in range(cfg.num_primary_layers)]
        )

        self.secondary_encoder = torch.nn.ModuleList(
            [ConformerBlockV1(cfg.conformer_block_cfg) for _ in range(cfg.num_secondary_layers)]
        )

        self.prim_sec_linear = torch.nn.Linear(2 * base_dim, base_dim)

        self.mas_encoder = torch.nn.ModuleList(
            [ConformerBlockV1(cfg.conformer_block_cfg) for _ in range(cfg.num_mixture_aware_speaker_layers)]
        )

        self.final_linear = torch.nn.Linear(base_dim, cfg.target_size)

    def forward(
        self,
        primary_audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
        secondary_audio_features: Optional[torch.Tensor] = None,
        mix_audio_features: Optional[torch.Tensor] = None,
    ):
        sequence_mask = lengths_to_padding_mask(audio_features_len)

        x_prim = primary_audio_features

        if secondary_audio_features is None and mix_audio_features is None:  # primary_audio_featues shape = [B, T, 3*F]
            primary_audio_features, secondary_audio_features, mix_audio_features = torch.split(
                primary_audio_features, primary_audio_features.shape[2] // 3, dim=2
            )  # [B, T, F], [B, T, F], [B, T, F]

        if self.use_secondary_audio:
            x_sec = secondary_audio_features
        else:
            x_sec = mix_audio_features
        assert x_sec is not None

        x = self.specaugment(torch.concat((x_prim, x_sec), dim=0))  # [2*B, T, F]
        x_prim, x_sec = torch.split(x, x.shape[0] // 2, dim=0)  # [B, T, F], [B, T, F]

        x_prim, _ = self.primary_frontend(x_prim, sequence_mask)  # [B, T, F], [B, T, F]
        x_sec, sequence_mask = self.secondary_frontend(x_sec, sequence_mask)  # [B, T, F], [B, T, F]

        for module in self.primary_encoder:
            x_prim = module(x_prim, sequence_mask)  # [B, T, F]
        for module in self.secondary_encoder:
            x_sec = module(x_sec, sequence_mask)  # [B, T, F]

        x_prim_sec = torch.concat((x_prim, x_sec), dim=2)  # [B, T, 2*F]
        x_prim_sec = self.prim_sec_linear(x_prim_sec)  # [B, T, F]

        for module in self.mas_encoder:
            x_prim_sec = module(x_prim_sec, sequence_mask)  # [B, T, F]

        logits = self.final_linear(x_prim_sec)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, F]

        if self.training:
            return log_probs, sequence_mask
        return log_probs


def get_train_serializer(
    model_config: ConformerHybridDualSpeakerConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridDualSpeakerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.train_steps.hybrid_dualspeaker_viterbi.train_step"),
        ],
    )


def get_prior_serializer(
    model_config: ConformerHybridDualSpeakerConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridDualSpeakerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.forward.hybrid_dualspeaker.forward_step"),
            Import(
                f"{pytorch_package}.forward.prior_callback.ComputePriorCallback",
                import_as="forward_callback",
            ),
        ],
    )


def get_recog_serializer(
    model_config: ConformerHybridDualSpeakerConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridDualSpeakerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.export.hybrid_dualspeaker.export"),
        ],
    )


def get_serializer(model_config: ConformerHybridDualSpeakerConfig, variant: ConfigVariant) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def get_default_config_v1(num_inputs: int, num_outputs: int) -> ConformerHybridDualSpeakerConfig:
    specaugment_cfg = specaugment.SpecaugmentConfigV1(
        max_time_mask_num=20,
        max_time_mask_size=20,
        max_feature_mask_num=1,
        max_feature_mask_size=15,
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
        pool1_kernel_size=(1, 1),
        pool1_stride=None,
        pool1_padding=None,
        pool2_kernel_size=(1, 2),
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

    return ConformerHybridDualSpeakerConfig(
        specaugment_cfg=specaugment_cfg,
        primary_frontend=frontend,
        secondary_frontend=None,
        conformer_block_cfg=block_cfg,
        num_primary_layers=8,
        num_secondary_layers=4,
        num_mixture_aware_speaker_layers=4,
        use_secondary_audio=False,
        target_size=num_outputs,
    )
