import os
import numpy as np
import torch
from dataclasses import dataclass
from torch import nn

from .wav2vec2_hf_variance_v1_cfg import ModelConfig


_HF_CACHE_DIR = "/work/asr4/zyang/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{_HF_CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_CACHE_DIR}/transformers"
os.environ["XDG_CACHE_HOME"] = _HF_CACHE_DIR


def _lengths_to_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


@dataclass
class VarianceConfig:
    feature_layer_index: int = -1


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)

        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "wav2vec2_hf_variance_v1 requires the 'transformers' package in the RETURNN environment."
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

    def extract_features(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, feature_layer_index=None):
        squeezed_audio = torch.squeeze(raw_audio, dim=-1)
        raw_audio_len = raw_audio_len.to(device=squeezed_audio.device, dtype=torch.long)
        attention_mask = _lengths_to_attention_mask(raw_audio_len, squeezed_audio.shape[1]).to(dtype=torch.long)

        encoder_out = self.wav2vec2(
            input_values=squeezed_audio,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        all_hidden_states = encoder_out.hidden_states
        layer_index = self.cfg.feature_layer_index if feature_layer_index is None else feature_layer_index
        features = all_hidden_states[layer_index]
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(raw_audio_len)
        return features, output_lengths.to(dtype=torch.long)


def variance_init_hook(run_ctx, **kwargs):
    cfg = VarianceConfig(**(kwargs.get("config") or {}))
    run_ctx.feature_layer_index = cfg.feature_layer_index
    run_ctx.sum = None
    run_ctx.sum_sq = None
    run_ctx.num_frames = 0


def variance_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    features, output_lengths = model.extract_features(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        feature_layer_index=run_ctx.feature_layer_index,
    )

    max_t = features.shape[1]
    mask = torch.arange(max_t, device=features.device)[None, :] < output_lengths[:, None]
    valid_features = features[mask]

    batch_sum = valid_features.sum(dim=0)
    batch_sum_sq = (valid_features ** 2).sum(dim=0)
    batch_num_frames = int(valid_features.shape[0])

    if run_ctx.sum is None:
        run_ctx.sum = batch_sum
        run_ctx.sum_sq = batch_sum_sq
    else:
        run_ctx.sum += batch_sum
        run_ctx.sum_sq += batch_sum_sq
    run_ctx.num_frames += batch_num_frames


def variance_finish_hook(run_ctx, **kwargs):
    if run_ctx.num_frames == 0:
        raise ValueError("No frames observed while computing wav2vec2 feature variance.")

    total_sum = run_ctx.sum.detach().cpu().numpy()
    total_sum_sq = run_ctx.sum_sq.detach().cpu().numpy()
    mean = total_sum / run_ctx.num_frames
    variance = total_sum_sq / run_ctx.num_frames - np.square(mean)
    variance = np.maximum(variance, 0.0)

    with open("variance.txt", "w") as f:
        np.savetxt(f, variance, delimiter=" ")
    print(
        f"Saved feature variance for layer {run_ctx.feature_layer_index} with {run_ctx.num_frames} frames to variance.txt."
    )
