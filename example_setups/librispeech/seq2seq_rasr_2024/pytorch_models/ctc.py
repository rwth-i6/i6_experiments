from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from i6_models.assemblies.conformer import (
    ConformerBlockV2Config,
    ConformerEncoderV2,
    ConformerEncoderV2Config,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import (
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length
from sisyphus import tk


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to an equivalent boolean mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    max_length = torch.max(lengths)
    index_range = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)  # type: ignore
    sequence_mask = torch.less(index_range[None, :], lengths[:, None])

    return sequence_mask


@dataclass
class SpecaugmentByLengthConfig(ModelConfiguration):
    start_epoch: int
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    logmel_cfg: RasrCompatibleLogMelFeatureExtractionV1Config
    conformer_cfg: ConformerEncoderV2Config
    dim: int
    target_size: int
    dropout: float
    specaug_cfg: SpecaugmentByLengthConfig


@dataclass
class ConformerCTCRecogConfig(ConformerCTCConfig):
    prior_file: tk.Path
    prior_scale: float
    blank_penalty: float


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, cfg: ConformerCTCConfig, epoch: int, **_):
        super().__init__()
        self.epoch = epoch
        self.feature_extraction = RasrCompatibleLogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.conformer = ConformerEncoderV2(cfg.conformer_cfg)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.target_size = cfg.target_size
        self.final_linear = torch.nn.Linear(cfg.dim, self.target_size)
        self.specaug_config = cfg.specaug_cfg

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            audio_samples = audio_samples.squeeze(-1)  # [B, T]
            features, features_size = self.feature_extraction.forward(
                audio_samples, audio_samples_size
            )  # [B, T, F], [B]
            sequence_mask = lengths_to_padding_mask(features_size)  # [B, T]

            if self.training and self.epoch >= self.specaug_config.start_epoch:
                features = specaugment_v1_by_length(
                    audio_features=features,
                    time_min_num_masks=self.specaug_config.time_min_num_masks,
                    time_max_mask_per_n_frames=self.specaug_config.time_max_mask_per_n_frames,
                    time_mask_max_size=self.specaug_config.time_mask_max_size,
                    freq_min_num_masks=self.specaug_config.freq_min_num_masks,
                    freq_max_num_masks=self.specaug_config.freq_max_num_masks,
                    freq_mask_max_size=self.specaug_config.freq_mask_max_size,
                )  # [B, T, F]

        encoder_states, sequence_mask = self.conformer.forward(features, sequence_mask)  # [B, T, F], [B, T]
        encoder_states = encoder_states[0]

        encoder_states = self.dropout(encoder_states)  # [B, T, F]
        logits = self.final_linear(encoder_states)  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, V]

        return log_probs, torch.sum(sequence_mask, dim=1).type(torch.int32)


class ConformerCTCRecogModel(ConformerCTCModel):
    def __init__(self, cfg: ConformerCTCRecogConfig, epoch: int, **_):
        super().__init__(
            cfg=cfg,
            epoch=epoch,
        )
        self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        log_probs, _ = super().forward(audio_samples=audio_samples, audio_samples_size=audio_samples_size)
        scores = -log_probs
        scores[:, -1] += self.blank_penalty
        return scores + self.scaled_priors.to(device=log_probs.device)


def get_model_config(target_size: int) -> ConformerCTCConfig:
    logmel_cfg = RasrCompatibleLogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        min_amp=1.175494e-38,
        num_filters=80,
        alpha=0.0,
    )

    specaug_cfg = SpecaugmentByLengthConfig(
        start_epoch=11,
        time_min_num_masks=2,
        time_max_mask_per_n_frames=25,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=5,
        freq_mask_max_size=16,
    )

    frontend = ModuleFactoryV1(
        GenericFrontendV1,
        GenericFrontendV1Config(
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
            conv_out_dims=[32, 64, 64, 32],
            conv_strides=None,
            conv_paddings=None,
            pool_kernel_sizes=[(2, 1), (2, 1)],
            pool_strides=None,
            pool_paddings=None,
            activations=[torch.nn.ReLU(), torch.nn.ReLU()],
            out_features=512,
        ),
    )

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
        norm=LayerNormNC(512),
    )

    block_cfg = ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "conv", "mhsa", "ff"],
        scales=[0.5, 1.0, 1.0, 0.5],
    )

    conformer_cfg = ConformerEncoderV2Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return ConformerCTCConfig(
        logmel_cfg=logmel_cfg,
        specaug_cfg=specaug_cfg,
        conformer_cfg=conformer_cfg,
        dim=512,
        target_size=target_size,
        dropout=0.1,
    )
