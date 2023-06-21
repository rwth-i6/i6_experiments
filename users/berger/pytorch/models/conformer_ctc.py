from dataclasses import dataclass

import torch
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.assemblies.conformer import conformer
from i6_models.config import ModelConfiguration, SubassemblyWithOptions

from ..custom_parts import specaugment, vgg_frontend


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    specaugment_cfg: specaugment.SpecaugmentConfigV1
    conformer_cfg: conformer.ConformerEncoderV1Config
    target_size: int


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.specaugment = specaugment.SpecaugmentModuleV1(step=step, cfg=cfg.specaugment_cfg)
        self.conformer = conformer.ConformerEncoderV1(cfg.conformer_cfg)
        self.final_linear = torch.nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size)

        self.export_mode = False

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        x = self.specaugment(audio_features)  # [B, T, F]
        encoder_padding_mask = None if self.export_mode else _lengths_to_padding_mask(audio_features_len)
        x = self.conformer(x, encoder_padding_mask)  # [B, T, F]
        logits = self.final_linear(x)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs


def get_serializer(
    model_config: ConformerCTCConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.ConformerCTCModel",
        train_step_import_path=f"{pytorch_package}.train_steps.ctc.train_step",
        model_config=model_config,
    )


def get_default_config_v1(num_inputs: int, num_outputs: int) -> ConformerCTCConfig:
    specaugment_cfg = specaugment.SpecaugmentConfigV1(
        max_time_mask_num=1,
        max_time_mask_size=15,
        max_feature_mask_num=num_inputs // 10,
        max_feature_mask_size=5,
    )

    frontend_cfg = vgg_frontend.VGGFrontendConfigV1(
        num_inputs=num_inputs,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv_kernel_size=3,
        conv1_stride=1,
        conv2_stride=2,
        conv3_stride=2,
        pool_size=2,
        linear_size=512,
        dropout=0.1,
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
        channels=512,
        kernel_size=31,
        dropout=0.1,
        activation=torch.nn.functional.silu,
        norm=torch.nn.BatchNorm1d(num_features=512, affine=False),
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

    return ConformerCTCConfig(
        specaugment_cfg=specaugment_cfg,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
    )
