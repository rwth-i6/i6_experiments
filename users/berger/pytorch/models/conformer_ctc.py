from dataclasses import dataclass

import torch

from i6_models.assemblies.conformer import conformer
from i6_models.parts.conformer.frontend import (
    ConformerVGGFrontendV1,
    ConformerVGGFrontendV1Config,
)
from i6_models.config import ModelConfiguration
from i6_models.config import SubassemblyWithOptions
from ..custom_parts import specaugment, vgg_frontend
from functools import partial


@dataclass
class Config(ModelConfiguration):
    specaugment_cfg: specaugment.SpecaugmentConfigV1
    conformer_cfg: conformer.ConformerEncoderV1Config
    target_size: int


class Model(torch.nn.Module):
    def __init__(self, step: int, config: Config, **kwargs):
        super().__init__()
        self.specaugment = specaugment.SpecaugmentModuleV1(
            step=step, config=config.specaugment_cfg
        )
        self.conformer = conformer.ConformerEncoderV1(config.conformer_cfg)
        self.final_linear = torch.nn.Linear(
            config.conformer_cfg.block_cfg.ff_cfg.input_dim, config.target_size
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        x = self.specaugment(audio_features)  # [B, T, F]
        x = self.conformer(x, audio_features_len)  # [B, T, F]
        logits = self.final_linear(x)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs


def get_default_config_v1(num_inputs: int, num_outputs: int) -> Config:
    specaugment_cfg = specaugment.SpecaugmentConfigV1(
        max_time_mask_num=1,
        max_time_mask_size=15,
        max_feature_mask_num=num_inputs // 10,
        max_feature_mask_size=5,
    )

    frontend_cfg = vgg_frontend.VGGFrontendConfigV1(
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv_kernel_size=3,
        conv1_stride=1,
        conv2_stride=2,
        conv3_stride=2,
        pool_size=2,
    )
    frontend = SubassemblyWithOptions(vgg_frontend.VGGFrontendV1, frontend_cfg)

    ff_cfg = conformer.ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        activation=torch.nn.functional.silu,
    )

    mhsa_cfg = conformer.ConformerMHSAV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    conv_cfg = conformer.ConformerConvolutionV1Config(
        channels=32,
        kernel_size=3,
        dropout=0.1,
        activation=torch.nn.functional.silu,
        norm=torch.nn.BatchNorm1d(num_features=32, affine=False),
    )

    block_cfg = conformer.ConformerBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = conformer.ConformerEncoderV1Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return Config(
        specaugment_cfg=specaugment_cfg,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
    )
