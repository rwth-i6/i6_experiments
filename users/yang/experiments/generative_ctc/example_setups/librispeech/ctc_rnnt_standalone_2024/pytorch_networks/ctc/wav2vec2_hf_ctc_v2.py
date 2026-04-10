# a generative version possible

import os
import numpy as np
import torch
from torch import nn

from .wav2vec2_hf_ctc_v2_cfg import ModelConfig
from ...extra_code.incremental_pca import IncrementalPCA


_HF_CACHE_DIR = "/work/asr4/zyang/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{_HF_CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_CACHE_DIR}/transformers"
os.environ["XDG_CACHE_HOME"] = _HF_CACHE_DIR


def _lengths_to_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask

class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)

        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "wav2vec2_hf_ctc_v1 requires the 'transformers' package in the RETURNN training environment."
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
        assert len(self.return_layers) == len(self.scales), "aux_ctc_loss_layers/scales length mismatch"

        if self.cfg.pca_dim:
            self.pca_dim = self.cfg.pca_dim
            self.pca = nn.ModuleList([IncrementalPCA(n_components=self.pca_dim) for _ in self.return_layers])
            self.pca_component_buffer_names = []
            self.pca_mean_buffer_names = []
            self.pca_var_buffer_names = []
            self.pca_init_buffer_names = []
            hidden_size = self.wav2vec2.config.hidden_size
            for i, _layer in enumerate(self.return_layers):
                comp_name = f"pca_components_{i}"
                mean_name = f"pca_mean_{i}"
                var_name = f"pca_var_{i}"
                init_name = f"pca_initialized_{i}"
                self.register_buffer(comp_name, torch.zeros(self.pca_dim, hidden_size))
                self.register_buffer(mean_name, torch.zeros(hidden_size))
                self.register_buffer(var_name, torch.zeros(hidden_size))
                self.register_buffer(init_name, torch.zeros((), dtype=torch.bool))
                self.pca_component_buffer_names.append(comp_name)
                self.pca_mean_buffer_names.append(mean_name)
                self.pca_var_buffer_names.append(var_name)
                self.pca_init_buffer_names.append(init_name)
        else:
            self.pca = None
        if self.cfg.pca_state_path is not None:
            self._load_pca_state(self.cfg.pca_state_path)
        self.hidden_size = self.wav2vec2.config.hidden_size if not self.pca else self.pca_dim
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.cfg.label_target_size + 1)
                for _ in self.return_layers
            ]
        )
        if self.cfg.freeze_output_layers:
            for classifier in self.classifiers:
                for param in classifier.parameters():
                    param.requires_grad = False
        self.blank_index = (
            self.cfg.label_target_size if self.cfg.blank_index is None else self.cfg.blank_index
        )
        self.pca_components = None

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
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(raw_audio_len)
        return encoder_out.hidden_states, output_lengths.to(dtype=torch.long)

    def _sync_pca_buffers_from_module(self, pca: IncrementalPCA, idx: int):
        if not hasattr(pca, "components_") or pca.components_ is None:
            return
        self._get_pca_component_buffer(idx).copy_(pca.components_.detach())
        self._get_pca_mean_buffer(idx).copy_(pca.mean.detach())
        self._get_pca_var_buffer(idx).copy_(pca.var.detach())
        self._get_pca_is_initialized_buffer(idx).fill_(True)

    def _get_pca_component_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.pca_component_buffer_names[idx])

    def _get_pca_mean_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.pca_mean_buffer_names[idx])

    def _get_pca_var_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.pca_var_buffer_names[idx])

    def _get_pca_is_initialized_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.pca_init_buffer_names[idx])

    def _get_pca_state_path(self, path) -> str:
        return path.get_path() if hasattr(path, "get_path") else str(path)

    def _load_pca_state(self, path) -> None:
        if not self.pca:
            raise RuntimeError("Received pca_state_path, but pca_dim is not configured for this model.")
        pca_state = torch.load(self._get_pca_state_path(path), map_location="cpu")
        if pca_state["return_layers"] != self.return_layers:
            raise ValueError(
                f"PCA state return_layers mismatch: expected {self.return_layers}, got {pca_state['return_layers']}"
            )
        if pca_state["pca_dim"] != self.pca_dim:
            raise ValueError(f"PCA state pca_dim mismatch: expected {self.pca_dim}, got {pca_state['pca_dim']}")
        for buffer_name, value in pca_state["buffers"].items():
            getattr(self, buffer_name).copy_(value)

    def export_pca_state(self) -> dict:
        if not self.pca:
            raise RuntimeError("Cannot export PCA state when pca_dim is not configured.")
        buffer_names = (
            self.pca_component_buffer_names
            + self.pca_mean_buffer_names
            + self.pca_var_buffer_names
            + self.pca_init_buffer_names
        )
        return {
            "return_layers": list(self.return_layers),
            "pca_dim": self.pca_dim,
            "buffers": {buffer_name: getattr(self, buffer_name).detach().cpu() for buffer_name in buffer_names},
        }

    def transform_hidden_states(self, all_hidden_states, output_lengths, *, update_pca: bool):
        hidden_states_mask = mask_tensor(all_hidden_states[0], output_lengths)
        transformed_hidden_states = []

        if self.pca:
            for idx, (layer_idx, pca) in enumerate(zip(self.return_layers, self.pca)):
                hidden_states = all_hidden_states[layer_idx]
                if update_pca:
                    pca.partial_fit(hidden_states[hidden_states_mask].detach())
                    self._sync_pca_buffers_from_module(pca, idx)
                elif not bool(self._get_pca_is_initialized_buffer(idx)):
                    raise RuntimeError(
                        f"PCA components for layer {layer_idx} are not initialized. "
                        "Run training first or load a checkpoint with fitted PCA state."
                    )
                centered_hidden_states = hidden_states - self._get_pca_mean_buffer(idx)
                transformed_hidden_states.append(
                    centered_hidden_states @ self._get_pca_component_buffer(idx).T
                )
        else:
            for layer_idx in self.return_layers:
                transformed_hidden_states.append(all_hidden_states[layer_idx])

        return transformed_hidden_states

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        all_hidden_states, output_lengths = self.extract_hidden_states(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len
        )
        transformed_hidden_states = self.transform_hidden_states(
            all_hidden_states,
            output_lengths,
            update_pca=self.training and self.cfg.update_pca_during_training,
        )
        log_probs_list = []
        for hidden_states, classifier in zip(transformed_hidden_states, self.classifiers):
            hidden_states = self.dropout(hidden_states)
            logits = classifier(hidden_states)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs_list.append(log_probs)

        return log_probs_list, output_lengths

    def get_log_probs_by_layer(self, log_probs_list, decode_layer_index=None):
        if decode_layer_index is None:
            return log_probs_list[-1]
        decode_layer_pos = self.return_layers.index(decode_layer_index)
        return log_probs_list[decode_layer_pos]


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    labels = data["labels"]
    labels_len = data["labels:size1"].to(torch.long)

    num_labels = torch.sum(labels_len)
    log_probs_list, output_lengths = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    for i, (log_probs, scale) in enumerate(zip(log_probs_list, model.scales)):
        if scale == 0.0:
            continue
        ctc_loss = nn.functional.ctc_loss(
            torch.permute(log_probs, (1, 0, 2)),
            labels,
            input_lengths=output_lengths.cpu(),
            target_lengths=labels_len.cpu(),
            blank=model.blank_index,
            reduction=model.cfg.ctc_loss_reduction,
            zero_infinity=True,
        )
        run_ctx.mark_as_loss(
            name=f"ctc_loss_layer{model.return_layers[i]}",
            loss=ctc_loss,
            scale=scale,
            inv_norm_factor=num_labels,
        )


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    for layer, sum_probs in run_ctx.sum_probs.items():
        all_probs = sum_probs.detach().cpu().numpy()
        average_probs = all_probs / all_frames
        log_average_probs = np.log(average_probs)
        print(f"Prior sum for layer {layer} in std-space (should be close to 1.0):", np.sum(average_probs))
        filename = f"prior_{layer}.txt"
        with open(filename, "w") as f:
            np.savetxt(f, log_average_probs, delimiter=" ")
        print(f"Saved prior for layer {layer} in {filename} in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    log_probs_list, output_lengths = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(output_lengths)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = {}
    for layer, log_probs in zip(model.return_layers, log_probs_list):
        probs = torch.exp(log_probs)
        if layer not in run_ctx.sum_probs:
            run_ctx.sum_probs[layer] = torch.sum(probs, dim=(0, 1))
        else:
            run_ctx.sum_probs[layer] += torch.sum(probs, dim=(0, 1))


def variance_init_hook(run_ctx, **kwargs):
    run_ctx.sum = {}
    run_ctx.sum_sq = {}
    run_ctx.sum_frames = {}


def variance_finish_hook(run_ctx, **kwargs):
    for layer in sorted(run_ctx.sum.keys(), key=lambda x: (x == -1, x)):
        num_frames = run_ctx.sum_frames[layer]
        if num_frames == 0:
            raise ValueError(f"No frames observed while computing variance for layer {layer}.")
        total_sum = run_ctx.sum[layer].detach().cpu().numpy()
        total_sum_sq = run_ctx.sum_sq[layer].detach().cpu().numpy()
        mean = total_sum / num_frames
        variance = total_sum_sq / num_frames - np.square(mean)
        variance = np.maximum(variance, 0.0)
        filename = f"variance_{layer}.txt"
        with open(filename, "w") as f:
            np.savetxt(f, variance, delimiter=" ")
        print(f"Saved variance for layer {layer} with {num_frames} frames to {filename}.")


def variance_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    all_hidden_states, output_lengths = model.extract_hidden_states(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transformed_hidden_states = model.transform_hidden_states(
        all_hidden_states,
        output_lengths,
        update_pca=False,
    )

    max_t = int(output_lengths.max().item())
    frame_mask = torch.arange(max_t, device=output_lengths.device)[None, :] < output_lengths[:, None]

    for layer, hidden_states in zip(model.return_layers, transformed_hidden_states):
        hidden_states = hidden_states[:, :max_t]
        valid_hidden_states = hidden_states[frame_mask]
        batch_sum = valid_hidden_states.sum(dim=0)
        batch_sum_sq = (valid_hidden_states ** 2).sum(dim=0)
        batch_num_frames = int(valid_hidden_states.shape[0])

        if layer not in run_ctx.sum:
            run_ctx.sum[layer] = batch_sum
            run_ctx.sum_sq[layer] = batch_sum_sq
            run_ctx.sum_frames[layer] = batch_num_frames
        else:
            run_ctx.sum[layer] += batch_sum
            run_ctx.sum_sq[layer] += batch_sum_sq
            run_ctx.sum_frames[layer] += batch_num_frames


def pca_fit_init_hook(run_ctx, **kwargs):
    run_ctx.model = None
    run_ctx.num_batches = 0


def pca_fit_finish_hook(run_ctx, **kwargs):
    if run_ctx.model is None:
        raise ValueError("No batches were processed while fitting PCA.")
    torch.save(run_ctx.model.export_pca_state(), "pca_state.pt")
    print(f"Saved PCA state after {run_ctx.num_batches} batches to pca_state.pt.")


def pca_fit_step(*, model: Model, data, run_ctx, **kwargs):
    if not model.pca:
        raise RuntimeError("PCA fitting requires pca_dim to be configured.")
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    all_hidden_states, output_lengths = model.extract_hidden_states(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    model.transform_hidden_states(
        all_hidden_states,
        output_lengths,
        update_pca=True,
    )
    run_ctx.model = model
    run_ctx.num_batches += 1
