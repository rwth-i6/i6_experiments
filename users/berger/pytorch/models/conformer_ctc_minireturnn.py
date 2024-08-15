from dataclasses import dataclass
from enum import Enum, auto

import i6_models.assemblies.conformer as conformer_i6
import i6_models.parts.conformer as conformer_parts_i6
import torch
from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from returnn.torch.context import get_run_ctx

from ..custom_parts.sequential import SequentialModuleV1, SequentialModuleV1Config
from ..custom_parts.specaugment import (
    SpecaugmentByLengthConfigV1,
    SpecaugmentByLengthModuleV1,
)
from ..custom_parts.speed_perturbation import SpeedPerturbationModuleV1, SpeedPerturbationModuleV1Config
from .util import lengths_to_padding_mask


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction: ModuleFactoryV1
    specaugment: ModuleFactoryV1
    conformer: ModuleFactoryV1
    dim: int
    target_size: int
    dropout: float
    skip_specaug_epochs: int


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, cfg: ConformerCTCConfig, **_):
        super().__init__()
        self.feature_extraction = cfg.feature_extraction()
        self.conformer = cfg.conformer()
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.final_linear = torch.nn.Linear(cfg.dim, cfg.target_size)
        self.specaugment = cfg.specaugment()
        self.skip_specaug_epochs = cfg.skip_specaug_epochs

    def forward(self, audio_features: torch.Tensor, audio_features_len: torch.Tensor):
        with torch.no_grad():
            audio_features = audio_features.squeeze(-1)
            x, input_len = self.feature_extraction(audio_features, audio_features_len)
            sequence_mask = lengths_to_padding_mask(input_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch > self.skip_specaug_epochs:
                x = self.specaugment(x)  # [B, T, F]

        x, sequence_mask = self.conformer(x, sequence_mask)  # [B, T, F]
        if isinstance(x, list):  # e.g. for return value of ConformerEncoderV2. Currently only support last layer
            assert len(x) == 1
            x = x[0]
        x = self.dropout(x)
        logits = self.final_linear(x)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(sequence_mask, dim=1).type(torch.int32)


def get_train_serializer(
    model_config: ConformerCTCConfig,
    **_,
) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.train_steps_minireturnn.ctc.train_step"),
        ],
    )


def get_prior_serializer(
    model_config: ConformerCTCConfig,
    **_,
) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.forward_minireturnn.prior_hooks.prior_init_hook", import_as="forward_init_hook"),
            Import(
                f"{pytorch_package}.forward_minireturnn.prior_hooks.prior_finish_hook", import_as="forward_finish_hook"
            ),
            Import(f"{pytorch_package}.forward_minireturnn.prior_hooks.prior_step", import_as="forward_step"),
        ],
    )


class RecogType(Enum):
    RASR = auto()
    FLASHLIGHT = auto()


def get_rasr_recog_serializer(
    model_config: ConformerCTCConfig,
    **_,
) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[Import(f"{pytorch_package}.export.ctc")],
    )


def get_flashlight_recog_serializer(
    model_config: ConformerCTCConfig,
    **kwargs,
) -> Collection:
    assert __package__ is not None
    pytorch_package = __package__.rpartition(".")[0]

    # Try to get some values from the returnn config at runtime since they will be set in the systems recognition step
    kwargs.setdefault("lexicon_file", CodeWrapper("lexicon_file"))
    kwargs.setdefault("vocab_file", CodeWrapper("vocab_file"))
    kwargs.setdefault("lm_file", CodeWrapper("lm_file"))
    kwargs.setdefault("prior_file", CodeWrapper("prior_file"))
    kwargs.setdefault("prior_scale", CodeWrapper("prior_scale"))
    kwargs.setdefault("lm_scale", CodeWrapper("lm_scale"))

    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.forward.ctc.flashlight_ctc_decoder_forward_step",
                import_as="forward_step",
                hashed_arguments=kwargs,
                unhashed_arguments={},
                unhashed_package_root="",
            ),
            Import(f"{pytorch_package}.forward.search_callback.SearchCallback", import_as="forward_callback"),
        ],
    )


def get_recog_serializer(
    model_config: ConformerCTCConfig,
    recog_type: RecogType = RecogType.RASR,
    **kwargs,
) -> Collection:
    if recog_type == RecogType.RASR:
        return get_rasr_recog_serializer(model_config, **kwargs)
    if recog_type == RecogType.FLASHLIGHT:
        return get_flashlight_recog_serializer(model_config, **kwargs)
    raise NotImplementedError


def get_serializer(model_config: ConformerCTCConfig, variant: ConfigVariant, **kwargs) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config, **kwargs)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config, **kwargs)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config, **kwargs)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config, **kwargs)
    raise NotImplementedError


def get_default_config_v1(num_outputs: int) -> ConformerCTCConfig:
    feature_extraction = ModuleFactoryV1(
        module_class=SequentialModuleV1,
        cfg=SequentialModuleV1Config(
            submodules=[
                ModuleFactoryV1(
                    module_class=SpeedPerturbationModuleV1,
                    cfg=SpeedPerturbationModuleV1Config(
                        min_speed_factor=0.9,
                        max_speed_factor=1.1,
                    ),
                ),
                ModuleFactoryV1(
                    module_class=RasrCompatibleLogMelFeatureExtractionV1,
                    cfg=RasrCompatibleLogMelFeatureExtractionV1Config(
                        sample_rate=16000,
                        win_size=0.025,
                        hop_size=0.01,
                        min_amp=1.175494e-38,
                        num_filters=80,
                        alpha=0.97,
                    ),
                ),
            ]
        ),
    )

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentByLengthModuleV1,
        cfg=SpecaugmentByLengthConfigV1(
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=5,
            freq_mask_max_size=16,
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
        norm=LayerNormNC(512),
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

    return ConformerCTCConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        conformer=ModuleFactoryV1(conformer_i6.ConformerEncoderV2, cfg=conformer_cfg),
        dim=512,
        target_size=num_outputs,
        dropout=0.1,
        skip_specaug_epochs=10,
    )
