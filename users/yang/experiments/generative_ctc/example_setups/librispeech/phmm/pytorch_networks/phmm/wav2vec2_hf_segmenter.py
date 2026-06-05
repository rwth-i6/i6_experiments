import os

import torch
from torch import nn

from .wav2vec2_hf_segmenter_cfg import ModelConfig
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.loss.segment_contrastive_loss import (
    segment_contrastive_loss,
)

_HF_CACHE_DIR = "/work/asr4/zyang/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{_HF_CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_CACHE_DIR}/transformers"
os.environ["XDG_CACHE_HOME"] = _HF_CACHE_DIR


def _lengths_to_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    time_axis = torch.arange(max_length, device=lengths.device)[None, :]
    return time_axis < lengths[:, None]


def _resolve_hidden_state_index(index: int, num_hidden_states: int) -> int:
    resolved = index if index >= 0 else num_hidden_states + index
    if not 0 <= resolved < num_hidden_states:
        raise ValueError(f"Requested hidden-state index {index} resolves to invalid index {resolved}")
    return resolved


def _conv1d_output_lengths(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
) -> torch.Tensor:
    lengths = torch.div(
        lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1
    return torch.clamp(lengths, min=0)


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)

        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "wav2vec2_hf_segmenter requires the 'transformers' package in the RETURNN training environment."
            ) from exc

        if self.cfg.pretrained:
            hf_cfg = Wav2Vec2Config.from_pretrained(self.cfg.hf_model_name, cache_dir=_HF_CACHE_DIR)
        else:
            hf_cfg = Wav2Vec2Config()

        hf_cfg.apply_spec_augment = self.cfg.apply_spec_augment
        hf_cfg.final_dropout = self.cfg.final_dropout
        hf_cfg.attention_dropout = self.cfg.attention_dropout
        hf_cfg.hidden_dropout = self.cfg.hidden_dropout
        hf_cfg.feat_proj_dropout = self.cfg.feat_proj_dropout
        hf_cfg.activation_dropout = self.cfg.activation_dropout
        hf_cfg.layerdrop = self.cfg.layerdrop
        hf_cfg.mask_time_prob = self.cfg.mask_time_prob
        hf_cfg.mask_time_length = self.cfg.mask_time_length
        hf_cfg.mask_feature_prob = self.cfg.mask_feature_prob
        hf_cfg.mask_feature_length = self.cfg.mask_feature_length

        if self.cfg.pretrained:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                self.cfg.hf_model_name,
                config=hf_cfg,
                cache_dir=_HF_CACHE_DIR,
            )
        else:
            self.wav2vec2 = Wav2Vec2Model(hf_cfg)

        if self.cfg.gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()

        if self.cfg.freeze_feature_encoder:
            self.wav2vec2.feature_extractor._freeze_parameters()

        if self.cfg.freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        if self.cfg.segmenter_num_blocks < 1:
            raise ValueError("segmenter_num_blocks must be >= 1")
        if self.cfg.segmenter_kernel_size < 1:
            raise ValueError("segmenter_kernel_size must be >= 1")
        if self.cfg.segmenter_stride < 1:
            raise ValueError("segmenter_stride must be >= 1")
        if self.cfg.segmenter_dilation < 1:
            raise ValueError("segmenter_dilation must be >= 1")

        self.hidden_size = self.wav2vec2.config.hidden_size
        self.segmenter_kernel_size = self.cfg.segmenter_kernel_size
        self.segmenter_stride = self.cfg.segmenter_stride
        self.segmenter_dilation = self.cfg.segmenter_dilation
        self.segmenter_padding = (
            self.cfg.segmenter_padding
            if self.cfg.segmenter_padding is not None
            else (self.segmenter_dilation * (self.segmenter_kernel_size - 1)) // 2
        )

        blocks = []
        in_channels = self.hidden_size
        for _block_idx in range(self.cfg.segmenter_num_blocks):
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    self.cfg.segmenter_channels,
                    kernel_size=self.segmenter_kernel_size,
                    stride=self.segmenter_stride,
                    dilation=self.segmenter_dilation,
                    padding=self.segmenter_padding,
                    bias=self.cfg.segmenter_bias,
                )
            )
            blocks.append(nn.LeakyReLU(negative_slope=self.cfg.leaky_relu_negative_slope))
            if self.cfg.segmenter_dropout > 0.0:
                blocks.append(nn.Dropout(self.cfg.segmenter_dropout))
            in_channels = self.cfg.segmenter_channels
        self.segmenter = nn.Sequential(*blocks)

    def extract_hidden_states(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        squeezed_audio = torch.squeeze(raw_audio, dim=-1)
        raw_audio_len = raw_audio_len.to(device=squeezed_audio.device, dtype=torch.long)
        attention_mask = _lengths_to_attention_mask(raw_audio_len, squeezed_audio.shape[1]).to(dtype=torch.long)

        encoder_out = self.wav2vec2(
            input_values=squeezed_audio,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(raw_audio_len).to(dtype=torch.long)
        return encoder_out.hidden_states, output_lengths

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        all_hidden_states, output_lengths = self.extract_hidden_states(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        hidden_state_index = _resolve_hidden_state_index(self.cfg.return_layer, len(all_hidden_states))
        features = all_hidden_states[hidden_state_index]

        features = self.segmenter(features.transpose(1, 2)).transpose(1, 2)
        for _ in range(self.cfg.segmenter_num_blocks):
            output_lengths = _conv1d_output_lengths(
                output_lengths,
                kernel_size=self.segmenter_kernel_size,
                stride=self.segmenter_stride,
                dilation=self.segmenter_dilation,
                padding=self.segmenter_padding,
            )
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
