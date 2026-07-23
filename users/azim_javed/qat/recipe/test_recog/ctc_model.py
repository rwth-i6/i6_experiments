import numpy as np
import torch
from torch.nn.modules.activation import ReLU, SiLU

from i6_experiments.users.berger.pytorch.custom_parts.specaugment import (
    SpecaugmentByLengthConfigV1,
    SpecaugmentByLengthModuleV1,
)
from i6_experiments.users.berger.pytorch.models.conformer_ctc_minireturnn import (
    ConformerCTCConfig,
    ConformerCTCModel,
)
from i6_models.assemblies.conformer.conformer_v2 import (
    ConformerBlockV2Config,
    ConformerEncoderV2,
    ConformerEncoderV2Config,
)
from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)


def get_ctc_model() -> ConformerCTCModel:
    ctc_model = ConformerCTCModel(
        cfg=ConformerCTCConfig(
            feature_extraction=ModuleFactoryV1(
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
            specaugment=ModuleFactoryV1(
                module_class=SpecaugmentByLengthModuleV1,
                cfg=SpecaugmentByLengthConfigV1(
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=25,
                    time_mask_max_size=20,
                    freq_min_num_masks=2,
                    freq_max_num_masks=5,
                    freq_mask_max_size=16,
                ),
            ),
            conformer=ModuleFactoryV1(
                module_class=ConformerEncoderV2,
                cfg=ConformerEncoderV2Config(
                    num_layers=12,
                    frontend=ModuleFactoryV1(
                        module_class=GenericFrontendV1,
                        cfg=GenericFrontendV1Config(
                            in_features=80,
                            layer_ordering=[
                                FrontendLayerType.Conv2d,
                                FrontendLayerType.Conv2d,
                                FrontendLayerType.Activation,
                                FrontendLayerType.Pool2d,
                                FrontendLayerType.Conv2d,
                                FrontendLayerType.Conv2d,
                                FrontendLayerType.Activation,
                                FrontendLayerType.Pool2d,
                            ],
                            conv_kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],
                            conv_strides=None,
                            conv_paddings=None,
                            conv_out_dims=[32, 64, 64, 32],
                            pool_kernel_sizes=[(2, 1), (2, 1)],
                            pool_strides=None,
                            pool_paddings=None,
                            activations=[ReLU(), ReLU()],
                            out_features=512,
                        ),
                    ),
                    block_cfg=ConformerBlockV2Config(
                        ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                            input_dim=512,
                            hidden_dim=2048,
                            dropout=0.1,
                            activation=SiLU(),
                        ),
                        mhsa_cfg=ConformerMHSAV1Config(
                            input_dim=512,
                            num_att_heads=8,
                            att_weights_dropout=0.1,
                            dropout=0.1,
                        ),
                        conv_cfg=ConformerConvolutionV1Config(
                            channels=512,
                            kernel_size=31,
                            dropout=0.1,
                            activation=SiLU(),
                            norm=LayerNormNC(512, eps=1e-05, elementwise_affine=True),
                        ),
                        modules=["ff", "conv", "mhsa", "ff"],
                        scales=[0.5, 1.0, 1.0, 0.5],
                    ),
                ),
            ),
            dim=512,
            target_size=185,
            dropout=0.1,
            specaug_start_epoch=11,
        )
    )

    checkpoint = torch.load(
        "/work/asr4/berger/rasr_dev/label_scorer/setup/dependencies/models/bpe_ctc/epoch.1000.pt",
        map_location=torch.device("cpu"),
    )
    ctc_model.load_state_dict(checkpoint["model"], strict=False)

    ctc_model.eval()

    return ctc_model
