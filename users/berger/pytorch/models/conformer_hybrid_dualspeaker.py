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
from i6_models.parts.blstm import BlstmEncoderV1Config, BlstmEncoderV1
from .util import lengths_to_padding_mask

from ..custom_parts import specaugment


@dataclass
class ConformerHybridDualSpeakerConfig(ModelConfiguration):
    use_mix_audio: bool

    specaugment_cfg: specaugment.SpecaugmentConfigV1

    separation_frontend: ModuleFactoryV1
    mixture_frontend: Optional[ModuleFactoryV1]

    conformer_block_cfg: ConformerBlockV1Config

    num_separation_layers: int
    num_mixture_layers: int
    num_mixture_aware_speaker_layers: int

    combination_encoder_cfg: Optional[BlstmEncoderV1Config]

    target_size: int


class ConformerHybridDualSpeakerModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerHybridDualSpeakerConfig, **kwargs):
        super().__init__()

        base_dim = cfg.conformer_block_cfg.ff_cfg.input_dim

        self.specaugment = specaugment.SpecaugmentModuleV1(
            step=step, cfg=cfg.specaugment_cfg
        )

        self.separation_frontend = cfg.separation_frontend()
        if cfg.mixture_frontend is None:
            self.mixture_frontend = cfg.separation_frontend()
        else:
            self.mixture_frontend = cfg.mixture_frontend()

        self.separation_encoder = torch.nn.Sequential(
            *[
                ConformerBlockV1(cfg.conformer_block_cfg)
                for _ in range(cfg.num_separation_layers)
            ]
        )

        self.mixture_encoder = torch.nn.Sequential(
            *[
                ConformerBlockV1(cfg.conformer_block_cfg)
                for _ in range(cfg.num_mixture_layers)
            ]
        )

        self.sep_mix_linear = torch.nn.Linear(2 * base_dim, base_dim)

        self.mas_encoder = torch.nn.Sequential(
            *[
                ConformerBlockV1(cfg.conformer_block_cfg)
                for _ in range(cfg.num_mixture_aware_speaker_layers)
            ]
        )

        if cfg.combination_encoder_cfg is not None:
            self.combination_encoder = BlstmEncoderV1(cfg.combination_encoder_cfg)
            final_in_dim = cfg.combination_encoder_cfg.hidden_dim * 2
        else:
            self.combination_encoder = None
            final_in_dim = base_dim

        self.final_linear = torch.nn.Linear(final_in_dim, 2 * cfg.target_size)

    def forward(
        self,
        sep_0_audio_features: torch.Tensor,
        sep_1_audio_features: torch.Tensor,
        mix_audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        sequence_mask = lengths_to_padding_mask(audio_features_len)

        x = self.specaugment(
            torch.concat(
                (sep_0_audio_features, sep_1_audio_features, mix_audio_features), dim=0
            )
        )  # [3*B, T, F]
        x_01, x_mix = torch.split(
            x, [2 * (x.shape[0] // 3), x.shape[0] // 3], dim=0
        )  # [2*B, T, F], [B, T, F]

        sequence_mask_twice = torch.concat(
            (sequence_mask, sequence_mask), dim=0
        )  # [2*B, T, F]

        x_01, sequence_mask_twice = self.separation_frontend(
            x_01, sequence_mask_twice
        )  # [2*B, T, F], [2*B, T, F]
        x_mix, sequence_mask = self.mixture_frontend(
            x_mix, sequence_mask
        )  # [B, T, F], [B, T, F]

        x_01, sequence_mask_twice = self.separation_encoder(
            x_01, sequence_mask_twice
        )  # [2*B, T, F]
        x_mix, sequence_mask = self.mixture_encoder(x_mix, sequence_mask)  # [B, T, F]

        x_mix_twice = torch.concat((x_mix, x_mix), dim=0)  # [2*B, T, F]
        x_01_mix = torch.concat((x_01, x_mix_twice), dim=2)  # [2*B, T, 2*F]
        x_01_mix = self.sep_mix_linear(x_01_mix)  # [2*B, T, F]

        x_01_mix = self.mas_encoder(x_01_mix, sequence_mask_twice)  # [2*B, T, F]

        x_0_mix, x_1_mix = torch.split(
            x_01_mix, x_01_mix.shape[0] // 2, dim=0
        )  # [B, T, F], [B, T, F]

        x_0_mix_1_mix = torch.concat((x_0_mix, x_1_mix), dim=2)  # [B, T, 2*F]

        if self.combination_encoder is not None:
            x_0_mix_1_mix = self.combination_encoder(x_0_mix_1_mix)  # [B, T, F]

        logits = self.final_linear(x_0_mix_1_mix)  # [B, T, 2*F]

        logits_0, logits_1 = torch.split(
            logits, logits.shape[0] // 2, dim=2
        )  # [B, T, F], [B, T, F]

        log_probs_0 = torch.log_softmax(logits_0, dim=2)  # [B, T, F]
        log_probs_1 = torch.log_softmax(logits_1, dim=2)  # [B, T, F]

        if self.training:
            return log_probs_0, log_probs_1, sequence_mask
        return log_probs_0, log_probs_1


def get_train_serializer(
    model_config: ConformerHybridDualSpeakerConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerHybridDualSpeakerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(
                f"{pytorch_package}.train_steps.hybrid_dual_speaker_viterbi.train_step"
            ),
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
            Import(f"{pytorch_package}.forward.basic.forward_step"),
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
            Import(f"{pytorch_package}.export.ctc.export"),
        ],
    )


def get_serializer(
    model_config: ConformerHybridDualSpeakerConfig, variant: ConfigVariant
) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def get_default_config_v1(
    num_inputs: int, num_outputs: int
) -> ConformerHybridDualSpeakerConfig:
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

    combination_cfg = BlstmEncoderV1Config(
        num_layers=1,
        input_dim=1024,
        hidden_dim=512,
        dropout=0.1,
        enforce_sorted=True,
    )

    return ConformerHybridDualSpeakerConfig(
        use_mix_audio=True,
        specaugment_cfg=specaugment_cfg,
        separation_frontend=frontend,
        mixture_frontend=None,
        conformer_block_cfg=block_cfg,
        num_separation_layers=6,
        num_mixture_layers=6,
        num_mixture_aware_speaker_layers=6,
        combination_encoder_cfg=combination_cfg,
        target_size=num_outputs,
    )
