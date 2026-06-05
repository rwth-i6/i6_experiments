import os

import torch
from torch import nn

from .wav2vec2_hf_ctc_generative_cfg import ModelConfig
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.loss.fixed_ctc_loss import (
    torch_ctc_fixed_grad,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.loss.generative_loss import (
    generative_nce,
)


_HF_CACHE_DIR = "/work/asr4/zyang/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{_HF_CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_CACHE_DIR}/transformers"
os.environ["XDG_CACHE_HOME"] = _HF_CACHE_DIR


def _lengths_to_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _resolve_hidden_state_index(index: int, num_hidden_states: int) -> int:
    resolved = index if index >= 0 else num_hidden_states + index
    if not 0 <= resolved < num_hidden_states:
        raise ValueError(f"Requested hidden-state index {index} resolves to invalid index {resolved}")
    return resolved


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)

        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "wav2vec2_hf_ctc_generative requires the 'transformers' package in the RETURNN environment."
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
        if self.cfg.pad_token_id is not None:
            hf_cfg.pad_token_id = self.cfg.pad_token_id

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

        self.dropout = nn.Dropout(self.cfg.final_dropout)
        self.return_layers = self.cfg.aux_ctc_loss_layers or [-1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]
        if len(self.return_layers) != len(self.scales):
            raise ValueError("aux_ctc_loss_layers and aux_ctc_loss_scales must have the same length")

        self.blank_index = self.cfg.label_target_size if self.cfg.blank_index is None else self.cfg.blank_index
        self.output_dim = self.cfg.label_target_size + 1
        if not 0 <= self.blank_index < self.output_dim:
            raise ValueError(f"blank_index must be in [0, {self.output_dim}), got {self.blank_index}")

        self.hidden_size = self.wav2vec2.config.hidden_size
        if self.cfg.generator_kernel <= 0:
            raise ValueError("generator_kernel must be positive")
        if self.cfg.generator_stride <= 0:
            raise ValueError("generator_stride must be positive")
        if self.cfg.generator_dilation <= 0:
            raise ValueError("generator_dilation must be positive")
        self.generator_padding = (self.cfg.generator_dilation * (self.cfg.generator_kernel - 1)) // 2

        if self.cfg.input_time_batch_norm:
            self.input_time_batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for _ in self.return_layers])
            for bn in self.input_time_batch_norms:
                nn.init.constant_(bn.weight, self.cfg.input_time_batch_norm_affine_init)
                nn.init.zeros_(bn.bias)
        else:
            self.input_time_batch_norms = None

        if self.cfg.input_residual_linear:
            self.input_residual_linears = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in self.return_layers]
            )
        else:
            self.input_residual_linears = None

        self.generators = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_size,
                    self.output_dim,
                    kernel_size=self.cfg.generator_kernel,
                    stride=self.cfg.generator_stride,
                    dilation=self.cfg.generator_dilation,
                    padding=self.generator_padding,
                    bias=self.cfg.generator_bias,
                )
                for _ in self.return_layers
            ]
        )
        if self.cfg.freeze_output_layers:
            for generator in self.generators:
                for param in generator.parameters():
                    param.requires_grad = False

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

    def _get_generator_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        output_lengths = torch.div(
            input_lengths
            + 2 * self.generator_padding
            - self.cfg.generator_dilation * (self.cfg.generator_kernel - 1)
            - 1,
            self.cfg.generator_stride,
            rounding_mode="floor",
        ) + 1
        return torch.clamp(output_lengths, min=0).to(dtype=torch.long)

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        all_hidden_states, output_lengths = self.extract_hidden_states(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

        log_probs_list = []
        for idx, (layer_idx, generator) in enumerate(zip(self.return_layers, self.generators)):
            hidden_states = all_hidden_states[_resolve_hidden_state_index(layer_idx, len(all_hidden_states))]
            if self.input_time_batch_norms is not None:
                hidden_states = self.input_time_batch_norms[idx](hidden_states.transpose(1, 2)).transpose(1, 2)
            if self.input_residual_linears is not None:
                hidden_states = hidden_states + self.input_residual_linears[idx](hidden_states)
            hidden_states = self.dropout(hidden_states)
            logits = generator(hidden_states.transpose(1, 2)).transpose(1, 2)
            log_probs_list.append(torch.nn.functional.logsigmoid(logits).float())

        if self.cfg.generator_stride > 1 or self.cfg.generator_kernel > 1 or self.cfg.generator_dilation > 1:
            output_lengths = self._get_generator_output_lengths(output_lengths)
        return log_probs_list, output_lengths

    def get_log_probs_by_layer(self, log_probs_list, decode_layer_index=None):
        if decode_layer_index is None:
            return log_probs_list[-1]
        decode_layer_pos = self.return_layers.index(decode_layer_index)
        return log_probs_list[decode_layer_pos]


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    labels = data["labels"].to(torch.long)
    labels_len = data["labels:size1"].to(torch.long)

    with torch.enable_grad():
        log_probs_list, output_lengths = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        inv_norm_factor = torch.clamp(output_lengths.sum().to(dtype=torch.float32), min=1.0)

        for log_probs, layer_index, scale in zip(log_probs_list, model.return_layers, model.scales):
            if scale == 0.0:
                continue
            ctc_loss = torch_ctc_fixed_grad(
                log_probs.permute(1, 0, 2),
                labels,
                input_lengths=output_lengths.cpu(),
                target_lengths=labels_len.cpu(),
                blank=model.blank_index,
                reduction="sum",
                zero_infinity=True,
            )
            soft_target = -torch.autograd.grad(ctc_loss, log_probs, retain_graph=True)[0].detach()
            loss = generative_nce(
                log_probs,
                soft_target.to(log_probs.dtype),
                sampling_type=model.cfg.sampling_type,
                seq_len=output_lengths,
                sampling_ratio=model.cfg.sampling_ratio,
                share_samples=model.cfg.share_samples,
                ratio_corrector=model.cfg.ratio_corrector,
            )
            run_ctx.mark_as_loss(
                name=f"ctc_generative_nce_layer{layer_index}",
                loss=loss.sum(),
                scale=scale,
                inv_norm_factor=inv_norm_factor,
            )
