import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from i6_models.parts.rasr_fsa import RasrFsaBuilderV2

from .wav2vec2_hf_phmm_generative_cfg import ModelConfig
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.wav2vec2_hf_ctc_v2 import (
    _HF_CACHE_DIR,
    _lengths_to_attention_mask,
    mask_tensor,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.loss.generative_loss import generative_nce

class RasrFsaBuilderOrth(RasrFsaBuilderV2):
    def build_single(self, orth: str):
        return self.builder.build_by_orthography(orth)


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
                "wav2vec2_hf_phmm requires the 'transformers' package in the RETURNN training environment."
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

        self.dropout = nn.Dropout(self.cfg.final_dropout)
        self.requested_return_layers = self.cfg.aux_loss_layers or [-1]
        self.scales = self.cfg.aux_loss_scales or [1.0]
        if len(self.requested_return_layers) != len(self.scales):
            raise ValueError("aux_loss_layers and aux_loss_scales must have the same length")
        self.return_layers = list(self.requested_return_layers)

        self.hidden_size = self.wav2vec2.config.hidden_size
        self.generator_kernel = self.cfg.generator_kernel
        self.generator_stride = self.cfg.generator_stride
        self.generator_dilation = self.cfg.generator_dilation

        if self.cfg.input_time_batch_norm:
            self.input_time_batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_size) for _ in self.return_layers]
            )
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

        if self.cfg.generator_dilation < 1:
            raise ValueError("generator_dilation must be >= 1")

        self.generator_padding = (self.cfg.generator_dilation * (self.cfg.generator_kernel - 1)) // 2
        self.generators = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_size,
                    self.cfg.label_target_size,
                    kernel_size=self.generator_kernel,
                    stride=self.generator_stride,
                    dilation=self.generator_dilation,
                    padding=self.generator_padding,
                    bias=self.cfg.generator_bias,
                )
                for _ in self.return_layers
            ]
        )

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


    def _load_variance_state(self, path) -> None:
        path_str = self._get_path_str(path)
        if path_str.endswith(".pt"):
            variance_state = torch.load(path_str, map_location="cpu")
            if variance_state["return_layers"] != self.return_layers:
                raise ValueError(
                    f"Variance state return_layers mismatch: expected {self.return_layers}, got {variance_state['return_layers']}"
                )
            for idx, layer in enumerate(self.return_layers):
                mean = variance_state["means"][layer].to(dtype=self._get_variance_mean_buffer(idx).dtype)
                variance = variance_state["variances"][layer].to(dtype=self._get_variance_buffer(idx).dtype)
                self._get_variance_mean_buffer(idx).copy_(mean)
                self._get_variance_buffer(idx).copy_(variance)
                self._get_variance_is_initialized_buffer(idx).fill_(True)
            return

        raise ValueError("For pHMM variance loading, only a .pt variance_state file is supported.")

    def transform_hidden_states(self, all_hidden_states, output_lengths):
        hidden_states_mask = mask_tensor(all_hidden_states[0], output_lengths)
        transformed_hidden_states = []

        for layer_idx in self.return_layers:
            transformed_hidden_states.append(all_hidden_states[_resolve_hidden_state_index(layer_idx, len(all_hidden_states))])

        return transformed_hidden_states

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        all_hidden_states, output_lengths = self.extract_hidden_states(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        transformed_hidden_states = self.transform_hidden_states(
            all_hidden_states,
            output_lengths,
        )

        log_probs_list = []
        for idx, (hidden_states, generator) in enumerate(zip(transformed_hidden_states, self.generators)):
            if self.input_time_batch_norms is not None:
                hidden_states = self.input_time_batch_norms[idx](hidden_states.transpose(1, 2)).transpose(1, 2)
            if self.input_residual_linears is not None:
                hidden_states = hidden_states + self.input_residual_linears[idx](hidden_states)
            hidden_states = self.dropout(hidden_states)
            logits = generator(hidden_states.transpose(1, 2)).transpose(1, 2)
            # print("logits mean", logits.mean().item())
            # print("logits min", logits.min().item())
            # print("logits max", logits.max().item())
            log_probs = torch.nn.functional.logsigmoid(logits)
            log_probs_list.append(log_probs)

        if self.generator_stride > 1 or self.generator_kernel > 1 or self.generator_dilation > 1:
            output_lengths = torch.div(
                output_lengths + 2 * self.generator_padding - self.generator_dilation * (self.generator_kernel - 1) - 1,
                self.generator_stride,
                rounding_mode="floor",
            ) + 1

        return log_probs_list, output_lengths

    def get_log_probs_by_layer(self, log_probs_list, decode_layer_index=None):
        if decode_layer_index is None:
            return log_probs_list[-1]
        decode_layer_pos = self.return_layers.index(decode_layer_index)
        return log_probs_list[decode_layer_pos]


class PhmmTrainStep:
    def __init__(
        self,
        fsa_exporter_config_path: str,
        transition_scale: float = 1.0,
        zero_infinity: bool = True,
        label_smoothing_scale: float = 0.0,
    ):
        self.fsa_builder = RasrFsaBuilderOrth(fsa_exporter_config_path, transition_scale)
        self.zero_infinity = zero_infinity
        self.label_smoothing_scale = label_smoothing_scale

    def __call__(self, *, model: Model, data, run_ctx, **kwargs):
        from i6_native_ops.fbw2 import fbw2_loss

        raw_audio = data["raw_audio"]
        raw_audio_len = data["raw_audio:size1"].to(torch.long)

        labels_raw = data["labels"]
        labels_len = data["labels:size1"]
        labels = [
            bytes(labels_raw[i, :labels_len[i]].cpu().tolist()).decode("utf8") + " "
            for i in range(labels_raw.shape[0])
        ]

        seq_tags = data.get("seq_tag")
        if seq_tags is not None:
            if hasattr(seq_tags, "tolist"):
                seq_tags = seq_tags.tolist()
            seq_tags = [
                tag.decode("utf8") if isinstance(tag, (bytes, bytearray)) else str(tag)
                for tag in seq_tags
            ]

        logprobs_list, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        audio_features_len_for_loss = audio_features_len.to(dtype=torch.int32).contiguous()

        for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
            if self.label_smoothing_scale > 0:
                vocab_size = logprobs.shape[-1]
                probs = torch.exp(logprobs)
                smoothed_probs = (1 - self.label_smoothing_scale) * probs + self.label_smoothing_scale / vocab_size
                logprobs = torch.log(smoothed_probs)

            target_fsa = self.fsa_builder.build_batch(labels).to(logprobs.device)
            logprobs = logprobs.float()
            ml_loss = fbw2_loss(logprobs, target_fsa, audio_features_len_for_loss)
            inv_norm_factor = torch.sum(audio_features_len)

            inf_mask = torch.isinf(ml_loss)
            if torch.any(inf_mask):
                inf_indices = torch.nonzero(inf_mask, as_tuple=False).flatten().tolist()
                print(
                    f"phmm_loss_layer{layer_index}: detected {len(inf_indices)} inf losses in batch of size {len(labels)}"
                )
                for idx in inf_indices:
                    msg = (
                        f"  seq_idx={idx} audio_frames={int(audio_features_len[idx])} raw_audio_len={int(raw_audio_len[idx])}, labels_len={labels_len}"
                    )
                    if seq_tags is not None:
                        msg += f" seq_tag={seq_tags[idx]!r}"
                    msg += f" orth={labels[idx]!r}"
                    print(msg)

            if self.zero_infinity:
                valid_mask = ~inf_mask
                ml_loss = torch.where(valid_mask, ml_loss, torch.zeros_like(ml_loss))
                inv_norm_factor = torch.sum(audio_features_len[valid_mask])
            else:
                valid_mask = torch.ones_like(inf_mask, dtype=torch.bool)


            ml_loss = torch.sum(ml_loss)
            run_ctx.mark_as_loss(
                name=f"phmm_loss_layer{layer_index}",
                loss=ml_loss,
                scale=scale,
                inv_norm_factor=inv_norm_factor,
            )

class GenPhmmTrainStep:
    def __init__(
        self,
        fsa_exporter_config_path: str,
        transition_scale: float = 1.0,
        zero_infinity: bool = True,
        label_smoothing_scale: float = 0.0,
    ):
        self.fsa_builder = RasrFsaBuilderOrth(fsa_exporter_config_path, transition_scale)
        self.zero_infinity = zero_infinity
        self.label_smoothing_scale = label_smoothing_scale

    def __call__(self, *, model: Model, data, run_ctx, **kwargs):
        from i6_native_ops.fbw2 import fbw2_loss

        raw_audio = data["raw_audio"]
        raw_audio_len = data["raw_audio:size1"].to(torch.long)

        labels_raw = data["labels"]
        labels_len = data["labels:size1"]
        labels = [
            bytes(labels_raw[i, :labels_len[i]].cpu().tolist()).decode("utf8") + " "
            for i in range(labels_raw.shape[0])
        ]

        seq_tags = data.get("seq_tag")
        if seq_tags is not None:
            if hasattr(seq_tags, "tolist"):
                seq_tags = seq_tags.tolist()
            seq_tags = [
                tag.decode("utf8") if isinstance(tag, (bytes, bytearray)) else str(tag)
                for tag in seq_tags
            ]

        with torch.enable_grad():
            logprobs_list, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
            audio_features_len_for_loss = audio_features_len.to(dtype=torch.int32).contiguous()

            for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
                # label smoothing disabled
                #if self.label_smoothing_scale > 0:
                #    vocab_size = logprobs.shape[-1]
                #    probs = torch.exp(logprobs)
                #    smoothed_probs = (1 - self.label_smoothing_scale) * probs + self.label_smoothing_scale / vocab_size
                #    logprobs = torch.log(smoothed_probs)

                target_fsa = self.fsa_builder.build_batch(labels).to(logprobs.device)
                logprobs = logprobs.float()
                ml_loss = fbw2_loss(logprobs, target_fsa, audio_features_len_for_loss)
                inv_norm_factor = torch.sum(audio_features_len)

                inf_mask = torch.isinf(ml_loss)
                if torch.any(inf_mask):
                    inf_indices = torch.nonzero(inf_mask, as_tuple=False).flatten().tolist()
                    print(
                        f"phmm_loss_layer{layer_index}: detected {len(inf_indices)} inf losses in batch of size {len(labels)}"
                    )
                    for idx in inf_indices:
                        msg = (
                            f"  seq_idx={idx} audio_frames={int(audio_features_len[idx])} raw_audio_len={int(raw_audio_len[idx])}"
                        )
                        if seq_tags is not None:
                            msg += f" seq_tag={seq_tags[idx]!r}"
                        msg += f" orth={labels[idx]!r}"
                        print(msg)

                if self.zero_infinity:
                    valid_mask = ~inf_mask
                    ml_loss = torch.where(valid_mask, ml_loss, torch.zeros_like(ml_loss))
                    inv_norm_factor = torch.sum(audio_features_len[valid_mask])
                else:
                    valid_mask = torch.ones_like(inf_mask, dtype=torch.bool)

                ml_loss = torch.sum(ml_loss)
                soft_target = -torch.autograd.grad(ml_loss, logprobs)[0].detach()
                valid_seq_len = torch.where(
                    valid_mask,
                    audio_features_len,
                    torch.zeros_like(audio_features_len),
                )
                loss = generative_nce(
                    logprobs,
                    soft_target.to(logprobs.dtype),
                    sampling_type=model.cfg.sampling_type,
                    seq_len=valid_seq_len,
                    sampling_ratio=model.cfg.sampling_ratio,
                    share_samples=model.cfg.share_samples,
                    ratio_corrector=model.cfg.ratio_corrector,
                )
                loss = loss.sum()
                run_ctx.mark_as_loss(
                    name=f"phmm_loss_layer{layer_index}",
                    loss=loss,
                    scale=scale,
                    inv_norm_factor=inv_norm_factor,
                )


_phmm_train_step = None


def train_step(*, model: Model, data, run_ctx, **kwargs):
    global _phmm_train_step

    if model.cfg.viterbi_training:
        raw_audio = data["raw_audio"]
        raw_audio_len = data["raw_audio:size1"].to(torch.long)
        alignment = data["alignments"].to(torch.long)
        alignment_length = data["alignments:size1"].to(torch.long)
        logprobs_list, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        down_sample_factor = 2 * model.generator_stride
        down_sampled_alignment = alignment[:, ::down_sample_factor]
        shared_time_dim = min(logprobs_list[0].shape[1], down_sampled_alignment.shape[1])
        valid_seq_len = torch.minimum(alignment_length // down_sample_factor, audio_features_len)
        valid_seq_len = torch.clamp(valid_seq_len, max=shared_time_dim)
        alignment_for_loss = down_sampled_alignment[:, :shared_time_dim]
        soft_targets = F.one_hot(alignment_for_loss, num_classes=model.cfg.label_target_size).to(logprobs_list[0].dtype)
        for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
            logprobs = logprobs[:, :shared_time_dim]
            # print(f"layer{layer_index}")
            # values, indices = torch.topk(logprobs[0,:50],k=2,dim=-1)
            # print("values", values)
            # print("indices", indices)
            loss = generative_nce(
                logprobs,
                soft_targets.to(logprobs.dtype),
                sampling_type=model.cfg.sampling_type,
                seq_len=valid_seq_len,
                sampling_ratio=model.cfg.sampling_ratio,
                share_samples=model.cfg.share_samples,
                ratio_corrector=model.cfg.ratio_corrector,
            )
            loss = loss.sum()
            inv_norm_factor = torch.sum(valid_seq_len)
            run_ctx.mark_as_loss(
                name=f"phmm_loss_layer{layer_index}",
                loss=loss,
                scale=scale,
                inv_norm_factor=inv_norm_factor,
            )
    else:
        # using the full sum mode

        if _phmm_train_step is None:
            _phmm_train_step = GenPhmmTrainStep(
                fsa_exporter_config_path=kwargs["fsa_exporter_config_path"],
                transition_scale=kwargs.get("transition_scale", 1.0),
                zero_infinity=kwargs.get("zero_infinity", True),
                label_smoothing_scale=kwargs.get("label_smoothing_scale", 0.0),
            )
        _phmm_train_step(model=model, data=data, run_ctx=run_ctx, **kwargs)
    # if _phmm_train_step is None:
    #     _phmm_train_step = PhmmTrainStep(
    #         fsa_exporter_config_path=kwargs["fsa_exporter_config_path"],
    #         transition_scale=kwargs.get("transition_scale", 1.0),
    #         zero_infinity=kwargs.get("zero_infinity", True),
    #         label_smoothing_scale=kwargs.get("label_smoothing_scale", 0.0),
    #     )
    # _phmm_train_step(model=model, data=data, run_ctx=run_ctx, **kwargs)
