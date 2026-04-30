import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from i6_models.parts.rasr_fsa import RasrFsaBuilderV2

from .wav2vec2_hf_phmm_cfg import ModelConfig
from ...extra_code.incremental_pca import IncrementalPCA
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.wav2vec2_hf_ctc_v2 import (
    _HF_CACHE_DIR,
    _lengths_to_attention_mask,
    mask_tensor,
)

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

        self.variance_mean_buffer_names = []
        self.variance_buffer_names = []
        self.variance_init_buffer_names = []
        self.use_variance_buffers = self.cfg.variance_normalize or self.cfg.variance_path is not None
        if self.use_variance_buffers:
            for i, _layer in enumerate(self.return_layers):
                mean_name = f"feature_mean_{i}"
                variance_name = f"feature_variance_{i}"
                init_name = f"feature_variance_initialized_{i}"
                self.register_buffer(mean_name, torch.zeros(self.hidden_size))
                self.register_buffer(variance_name, torch.ones(self.hidden_size))
                self.register_buffer(init_name, torch.zeros((), dtype=torch.bool))
                self.variance_mean_buffer_names.append(mean_name)
                self.variance_buffer_names.append(variance_name)
                self.variance_init_buffer_names.append(init_name)
        if self.cfg.variance_path is not None:
            self._load_variance_state(self.cfg.variance_path)

        if self.cfg.generative_model:
            self.classifiers = None
            self.label_features = nn.ParameterList(
                [nn.Parameter(torch.randn(self.cfg.label_target_size, self.hidden_size)) for _ in self.return_layers]
            )
            if self.cfg.freeze_output_layers:
                for label_feature in self.label_features:
                    label_feature.requires_grad = False
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.cfg.label_target_size) for _ in self.return_layers]
            )
            if self.cfg.freeze_output_layers:
                for classifier in self.classifiers:
                    for param in classifier.parameters():
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

    def _get_path_str(self, path) -> str:
        return path.get_path() if hasattr(path, "get_path") else str(path)

    def _get_variance_mean_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.variance_mean_buffer_names[idx])

    def _get_variance_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.variance_buffer_names[idx])

    def _get_variance_is_initialized_buffer(self, idx: int) -> torch.Tensor:
        return getattr(self, self.variance_init_buffer_names[idx])

    def _load_pca_state(self, path) -> None:
        if not self.pca:
            raise RuntimeError("Received pca_state_path, but pca_dim is not configured for this model.")
        pca_state = torch.load(self._get_path_str(path), map_location="cpu")
        if pca_state["return_layers"] != self.return_layers:
            raise ValueError(
                f"PCA state return_layers mismatch: expected {self.return_layers}, got {pca_state['return_layers']}"
            )
        if pca_state["pca_dim"] != self.pca_dim:
            raise ValueError(f"PCA state pca_dim mismatch: expected {self.pca_dim}, got {pca_state['pca_dim']}")
        for buffer_name, value in pca_state["buffers"].items():
            getattr(self, buffer_name).copy_(value)

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

    def transform_hidden_states(self, all_hidden_states, output_lengths, *, update_pca: bool):
        hidden_states_mask = mask_tensor(all_hidden_states[0], output_lengths)
        transformed_hidden_states = []

        if self.pca:
            for idx, (layer_idx, pca) in enumerate(zip(self.return_layers, self.pca)):
                resolved_layer_idx = _resolve_hidden_state_index(layer_idx, len(all_hidden_states))
                hidden_states = all_hidden_states[resolved_layer_idx]
                if update_pca:
                    pca.partial_fit(hidden_states[hidden_states_mask].detach())
                    self._sync_pca_buffers_from_module(pca, idx)
                elif not bool(self._get_pca_is_initialized_buffer(idx)):
                    raise RuntimeError(
                        f"PCA components for layer {layer_idx} are not initialized. "
                        "Run training first or load a checkpoint with fitted PCA state."
                    )
                centered_hidden_states = hidden_states - self._get_pca_mean_buffer(idx)
                transformed_hidden_states.append(centered_hidden_states @ self._get_pca_component_buffer(idx).T)
        else:
            for layer_idx in self.return_layers:
                transformed_hidden_states.append(all_hidden_states[_resolve_hidden_state_index(layer_idx, len(all_hidden_states))])

        if self.cfg.l2_norm:
            transformed_hidden_states = [F.normalize(hidden_states, dim=-1) for hidden_states in transformed_hidden_states]

        return transformed_hidden_states

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        all_hidden_states, output_lengths = self.extract_hidden_states(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        transformed_hidden_states = self.transform_hidden_states(
            all_hidden_states,
            output_lengths,
            update_pca=self.training and self.cfg.update_pca_during_training,
        )

        log_probs_list = []
        if self.cfg.generative_model:
            for idx, (hidden_states, label_features) in enumerate(zip(transformed_hidden_states, self.label_features)):
                if self.cfg.variance_normalize:
                    if not bool(self._get_variance_is_initialized_buffer(idx)):
                        raise RuntimeError(
                            f"Feature variance for layer {self.return_layers[idx]} is not initialized. "
                            "Provide variance_path or run the variance computation job first."
                        )
                    mean = self._get_variance_mean_buffer(idx)
                    std = torch.sqrt(self._get_variance_buffer(idx).clamp_min(1e-12))
                    hidden_states = (hidden_states - mean) / std
                hidden_states = self.dropout(hidden_states)
                diffs = hidden_states.unsqueeze(-2) - label_features.unsqueeze(0).unsqueeze(0)
                squared_distances = torch.sum(diffs ** 2, dim=-1)
                log_probs = -squared_distances
                log_probs_list.append(log_probs)
        else:
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

            ml_loss = torch.sum(ml_loss)
            run_ctx.mark_as_loss(
                name=f"phmm_loss_layer{layer_index}",
                loss=ml_loss,
                scale=scale,
                inv_norm_factor=inv_norm_factor,
            )


_phmm_train_step = None


def train_step(*, model: Model, data, run_ctx, **kwargs):
    global _phmm_train_step
    if _phmm_train_step is None:
        _phmm_train_step = PhmmTrainStep(
            fsa_exporter_config_path=kwargs["fsa_exporter_config_path"],
            transition_scale=kwargs.get("transition_scale", 1.0),
            zero_infinity=kwargs.get("zero_infinity", True),
            label_smoothing_scale=kwargs.get("label_smoothing_scale", 0.0),
        )
    _phmm_train_step(model=model, data=data, run_ctx=run_ctx, **kwargs)


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

    log_probs_list, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = {}
    for layer, log_probs in zip(model.return_layers, log_probs_list):
        if model.cfg.generative_model:
            probs = torch.softmax(log_probs, dim=-1)
        else:
            probs = torch.exp(log_probs)
        mask = mask_tensor(probs, audio_features_len).unsqueeze(-1)
        probs = probs * mask
        if layer not in run_ctx.sum_probs:
            run_ctx.sum_probs[layer] = torch.sum(probs, dim=(0, 1))
        else:
            run_ctx.sum_probs[layer] += torch.sum(probs, dim=(0, 1))
