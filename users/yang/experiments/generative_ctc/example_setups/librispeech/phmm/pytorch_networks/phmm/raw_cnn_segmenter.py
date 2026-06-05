import torch
from torch import nn

from .raw_cnn_segmenter_cfg import ModelConfig
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.loss.segment_contrastive_loss import (
    segment_contrastive_loss,
)


def _conv1d_output_lengths(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    lengths = torch.div(
        lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1
    return torch.clamp(lengths, min=0)


class Model(nn.Module):
    """
    Raw-waveform CNN segmenter following Kreuk et al. Interspeech 2020.

    Architecture: 5 strided Conv1d blocks with kernels (10, 8, 4, 4, 4),
    strides (5, 4, 2, 2, 2), 256 channels, BatchNorm1d and LeakyReLU, followed
    by a per-frame linear projection.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)

        if len(self.cfg.conv_kernel_sizes) != len(self.cfg.conv_strides):
            raise ValueError("conv_kernel_sizes and conv_strides must have the same length")
        if len(self.cfg.conv_kernel_sizes) < 1:
            raise ValueError("At least one convolutional block is required")
        if self.cfg.conv_channels < 1:
            raise ValueError("conv_channels must be >= 1")
        if self.cfg.projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")

        blocks = []
        in_channels = 1
        last_block_idx = len(self.cfg.conv_kernel_sizes) - 1
        for block_idx, (kernel_size, stride) in enumerate(zip(self.cfg.conv_kernel_sizes, self.cfg.conv_strides)):
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    self.cfg.conv_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    bias=self.cfg.conv_bias,
                )
            )
            apply_norm_activation = self.cfg.apply_final_norm_activation or block_idx != last_block_idx
            if apply_norm_activation:
                if self.cfg.batch_norm:
                    blocks.append(nn.BatchNorm1d(self.cfg.conv_channels))
                blocks.append(nn.LeakyReLU(negative_slope=self.cfg.leaky_relu_negative_slope))
            in_channels = self.cfg.conv_channels

        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Linear(self.cfg.conv_channels, self.cfg.projection_dim, bias=self.cfg.projection_bias)

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        features = torch.squeeze(raw_audio, dim=-1).unsqueeze(1)
        output_lengths = raw_audio_len.to(device=features.device, dtype=torch.long)

        features = self.encoder(features).transpose(1, 2)
        for kernel_size, stride in zip(self.cfg.conv_kernel_sizes, self.cfg.conv_strides):
            output_lengths = _conv1d_output_lengths(output_lengths, kernel_size=kernel_size, stride=stride)

        features = self.projection(features)
        output_lengths = torch.clamp(output_lengths, max=features.shape[1])
        return features, output_lengths


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    features, feature_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    loss = segment_contrastive_loss(
        features,
        feature_len,
        num_samples=model.cfg.contrastive_num_samples,
        temperature=model.cfg.contrastive_temperature,
    )
    inv_norm_factor = torch.sum(torch.clamp(feature_len - 1, min=0))
    if loss.numel() == 0:
        loss_sum = features.sum() * 0.0
        inv_norm_factor = torch.ones_like(inv_norm_factor)
    else:
        loss_sum = loss.sum()
    run_ctx.mark_as_loss(
        name="segment_contrastive_loss",
        loss=loss_sum,
        inv_norm_factor=inv_norm_factor,
    )
