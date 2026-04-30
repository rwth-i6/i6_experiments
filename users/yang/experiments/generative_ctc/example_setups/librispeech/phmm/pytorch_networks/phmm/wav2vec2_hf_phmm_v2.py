import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from i6_models.parts.rasr_fsa import RasrFsaBuilderV2

from .wav2vec2_hf_phmm_v2_cfg import ModelConfig
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


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"Expected a positive value, got {value}")
    return float(np.log(np.expm1(value)))


def _inverse_softplus_tensor(value: torch.Tensor) -> torch.Tensor:
    if torch.any(value <= 0.0):
        raise ValueError("Expected strictly positive values for inverse softplus.")
    return torch.log(torch.expm1(value))


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
        self.class_stats_state = None
        if self.cfg.gaussian_class_stats_path is not None:
            self._load_class_stats_state(self.cfg.gaussian_class_stats_path)

        if self.cfg.generative_model and self.cfg.gaussian_mixture_model:
            raise ValueError("Only one of generative_model and gaussian_mixture_model can be enabled.")
        if self.cfg.num_gaussian_mixtures < 1:
            raise ValueError("num_gaussian_mixtures must be at least 1.")
        if self.cfg.gaussian_init_precision <= 0.0:
            raise ValueError("gaussian_init_precision must be positive.")
        if self.cfg.gaussian_precision_floor <= 0.0:
            raise ValueError("gaussian_precision_floor must be positive.")
        if self.cfg.gaussian_mean_init_scale <= 0.0:
            raise ValueError("gaussian_mean_init_scale must be positive.")
        if self.cfg.gaussian_precision_type not in {"diagonal", "full"}:
            raise ValueError(
                f"Unsupported gaussian_precision_type {self.cfg.gaussian_precision_type!r}. "
                "Expected 'diagonal' or 'full'."
            )

        if self.cfg.generative_model:
            self.classifiers = None
            self.gaussian_means = None
            self.gaussian_raw_precisions = None
            self.gaussian_mixture_logits = None
            self.label_features = nn.ParameterList(
                [nn.Parameter(torch.randn(self.cfg.label_target_size, self.hidden_size)) for _ in self.return_layers]
            )
            if self.cfg.freeze_output_layers:
                for label_feature in self.label_features:
                    label_feature.requires_grad = False
        elif self.cfg.gaussian_mixture_model:
            self.classifiers = None
            self.label_features = None
            self.num_gaussian_mixtures = self.cfg.num_gaussian_mixtures
            gaussian_mean_inits = []
            gaussian_raw_precision_inits = []
            for idx, _layer in enumerate(self.return_layers):
                if self.cfg.gaussian_init_from_class_stats:
                    mean_init, raw_precision_init = self._build_gaussian_initialization_from_class_stats(idx)
                elif self.cfg.gaussian_init_from_variance_stats:
                    mean_init, raw_precision_init = self._build_gaussian_initialization_from_variance_stats(idx)
                else:
                    mean_init, raw_precision_init = self._build_default_gaussian_initialization()
                gaussian_mean_inits.append(nn.Parameter(mean_init))
                gaussian_raw_precision_inits.append(nn.Parameter(raw_precision_init))

            self.gaussian_means = nn.ParameterList(gaussian_mean_inits)
            self.gaussian_raw_precisions = nn.ParameterList(gaussian_raw_precision_inits)
            self.gaussian_mixture_logits = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(self.cfg.label_target_size, self.num_gaussian_mixtures))
                    for _ in self.return_layers
                ]
            )
            if self.cfg.freeze_gaussian_precision:
                for parameter in self.gaussian_raw_precisions:
                    parameter.requires_grad = False
            if self.cfg.freeze_output_layers:
                    for parameter_list in (
                        self.gaussian_means,
                        self.gaussian_raw_precisions,
                        self.gaussian_mixture_logits,
                    ):
                        for parameter in parameter_list:
                            parameter.requires_grad = False
        else:
            self.label_features = None
            self.gaussian_means = None
            self.gaussian_raw_precisions = None
            self.gaussian_mixture_logits = None
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

    def _load_class_stats_state(self, path) -> None:
        class_stats_state = torch.load(self._get_path_str(path), map_location="cpu")
        available_layers = list(class_stats_state["return_layers"])
        missing_layers = [layer for layer in self.return_layers if layer not in available_layers]
        if missing_layers:
            raise ValueError(
                f"Class-stats return_layers missing requested layers: expected {self.return_layers}, "
                f"got {class_stats_state['return_layers']}"
            )
        loaded_label_target_size = int(class_stats_state["label_target_size"])
        expected_label_target_size = int(self.cfg.label_target_size)
        if loaded_label_target_size != expected_label_target_size:
            raise ValueError(
                f"Class-stats label_target_size mismatch: expected {expected_label_target_size}, "
                f"got {loaded_label_target_size}"
            )
        self.class_stats_state = class_stats_state

    def _build_default_gaussian_initialization(self):
        raw_precision_init = _inverse_softplus(self.cfg.gaussian_init_precision)
        mean_init = self.cfg.gaussian_mean_init_scale * torch.randn(
            self.cfg.label_target_size,
            self.num_gaussian_mixtures,
            self.hidden_size,
        )
        if self.cfg.gaussian_precision_type == "diagonal":
            precision_shape = (
                (self.cfg.label_target_size, self.hidden_size)
                if self.cfg.gaussian_class_dependent_variance
                else (self.hidden_size,)
            )
            raw_precision = torch.full(precision_shape, raw_precision_init)
        else:
            precision_shape = (
                (self.cfg.label_target_size, self.hidden_size, self.hidden_size)
                if self.cfg.gaussian_class_dependent_variance
                else (self.hidden_size, self.hidden_size)
            )
            raw_precision = torch.zeros(precision_shape)
            diag_indices = torch.arange(self.hidden_size)
            if self.cfg.gaussian_class_dependent_variance:
                raw_precision[:, diag_indices, diag_indices] = raw_precision_init
            else:
                raw_precision[diag_indices, diag_indices] = raw_precision_init
        return mean_init, raw_precision

    def _build_gaussian_initialization_from_variance_stats(self, idx: int):
        if self.cfg.variance_path is None:
            raise ValueError(
                "gaussian_init_from_variance_stats=True requires variance_path to be set or precomputed in the experiment config."
            )
        if not bool(self._get_variance_is_initialized_buffer(idx)):
            raise RuntimeError(
                f"Variance buffers for layer {self.return_layers[idx]} are not initialized. "
                "Provide variance_path or run the variance computation job first."
            )

        pooled_mean = self._get_variance_mean_buffer(idx).detach().clone()
        pooled_variance = torch.clamp(
            self._get_variance_buffer(idx).detach().clone(),
            min=self.cfg.gaussian_precision_floor,
        )
        pooled_std = torch.sqrt(pooled_variance)

        mean_init = pooled_mean.view(1, 1, -1).repeat(
            self.cfg.label_target_size,
            self.num_gaussian_mixtures,
            1,
        )
        if self.num_gaussian_mixtures > 1:
            mean_init = mean_init + (
                self.cfg.gaussian_mean_init_scale
                * pooled_std.view(1, 1, -1)
                * torch.randn_like(mean_init)
            )

        effective_precision = torch.clamp(
            1.0 / pooled_variance,
            min=self.cfg.gaussian_precision_floor + 1e-8,
        )
        raw_precision_diag = _inverse_softplus_tensor(
            torch.clamp(effective_precision - self.cfg.gaussian_precision_floor, min=1e-8)
        )

        if self.cfg.gaussian_precision_type == "diagonal":
            if self.cfg.gaussian_class_dependent_variance:
                raw_precision = raw_precision_diag.unsqueeze(0).repeat(self.cfg.label_target_size, 1)
            else:
                raw_precision = raw_precision_diag
        else:
            if self.cfg.gaussian_class_dependent_variance:
                raw_precision = torch.zeros(self.cfg.label_target_size, self.hidden_size, self.hidden_size)
                diag_indices = torch.arange(self.hidden_size)
                raw_precision[:, diag_indices, diag_indices] = raw_precision_diag.unsqueeze(0)
            else:
                raw_precision = torch.zeros(self.hidden_size, self.hidden_size)
                diag_indices = torch.arange(self.hidden_size)
                raw_precision[diag_indices, diag_indices] = raw_precision_diag

        return mean_init, raw_precision

    def _build_gaussian_initialization_from_class_stats(self, idx: int):
        if self.class_stats_state is None:
            raise ValueError(
                "gaussian_init_from_class_stats=True requires gaussian_class_stats_path to be set "
                "or precomputed in the experiment config."
            )

        layer = self.return_layers[idx]
        class_means = self.class_stats_state["means"][layer].detach().clone()
        pooled_mean = self.class_stats_state["pooled_means"][layer].detach().clone()

        mean_init = class_means.unsqueeze(1).repeat(1, self.num_gaussian_mixtures, 1)
        if self.num_gaussian_mixtures > 1:
            random_direction = torch.randn_like(mean_init)
            random_direction = random_direction / torch.clamp(
                torch.linalg.norm(random_direction, dim=-1, keepdim=True),
                min=1e-12,
            )
            class_mean_norm = torch.linalg.norm(class_means, dim=-1, keepdim=True).unsqueeze(1)
            perturbation = (
                self.cfg.gaussian_class_stats_mixture_perturb_scale
                * class_mean_norm
                * random_direction
            )
            mean_init = mean_init + perturbation

        if self.cfg.gaussian_init_class_means_only:
            _, raw_precision = self._build_default_gaussian_initialization()
            class_counts = self.class_stats_state["counts"][layer]
            unseen_mask = class_counts <= 0
            if torch.any(unseen_mask):
                mean_init[unseen_mask] = pooled_mean.view(1, -1).unsqueeze(1).repeat(
                    int(unseen_mask.sum()), self.num_gaussian_mixtures, 1
                )
            return mean_init, raw_precision

        class_variances = self.class_stats_state.get("variances", {}).get(layer, None)
        pooled_variance = self.class_stats_state.get("pooled_variances", {}).get(layer, None)
        if pooled_variance is None:
            raise ValueError(f"Missing pooled_variances entry for layer {layer} in class-stats state.")
        pooled_variance = torch.clamp(pooled_variance.detach().clone(), min=self.cfg.gaussian_precision_floor)
        if class_variances is not None:
            class_variances = torch.clamp(class_variances.detach().clone(), min=self.cfg.gaussian_precision_floor)

        if self.cfg.gaussian_class_dependent_variance and class_variances is not None:
            precision_source = class_variances
        else:
            precision_source = pooled_variance

        effective_precision = torch.clamp(
            1.0 / precision_source,
            min=self.cfg.gaussian_precision_floor + 1e-8,
        )
        raw_precision_diag = _inverse_softplus_tensor(
            torch.clamp(effective_precision - self.cfg.gaussian_precision_floor, min=1e-8)
        )

        if self.cfg.gaussian_precision_type == "diagonal":
            if self.cfg.gaussian_class_dependent_variance:
                if raw_precision_diag.ndim == 1:
                    raw_precision = raw_precision_diag.unsqueeze(0).repeat(self.cfg.label_target_size, 1)
                else:
                    raw_precision = raw_precision_diag
            else:
                raw_precision = raw_precision_diag if raw_precision_diag.ndim == 1 else raw_precision_diag.mean(dim=0)
        else:
            diag_indices = torch.arange(self.hidden_size)
            if self.cfg.gaussian_class_dependent_variance:
                raw_precision = torch.zeros(self.cfg.label_target_size, self.hidden_size, self.hidden_size)
                diag_values = raw_precision_diag
                if diag_values.ndim == 1:
                    diag_values = diag_values.unsqueeze(0).repeat(self.cfg.label_target_size, 1)
                raw_precision[:, diag_indices, diag_indices] = diag_values
            else:
                raw_precision = torch.zeros(self.hidden_size, self.hidden_size)
                diag_values = raw_precision_diag if raw_precision_diag.ndim == 1 else raw_precision_diag.mean(dim=0)
                raw_precision[diag_indices, diag_indices] = diag_values

        # Fall back to pooled mean for unseen classes.
        class_counts = self.class_stats_state["counts"][layer]
        unseen_mask = class_counts <= 0
        if torch.any(unseen_mask):
            mean_init[unseen_mask] = pooled_mean.view(1, -1).unsqueeze(1).repeat(int(unseen_mask.sum()), self.num_gaussian_mixtures, 1)

        return mean_init, raw_precision

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
        elif self.cfg.gaussian_mixture_model:
            for idx, (hidden_states, means, raw_precision, mixture_logits) in enumerate(
                zip(
                    transformed_hidden_states,
                    self.gaussian_means,
                    self.gaussian_raw_precisions,
                    self.gaussian_mixture_logits,
                )
            ):
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
                log_probs_list.append(
                    self._gaussian_log_probs(
                        hidden_states=hidden_states,
                        means=means,
                        raw_precision=raw_precision,
                        mixture_logits=mixture_logits,
                        ignore_var_det=self.cfg.freeze_gaussian_precision and not self.cfg.gaussian_class_dependent_variance # when the precision is frozen, no need to consider it during training
                    )
                )
        else:
            for hidden_states, classifier in zip(transformed_hidden_states, self.classifiers):
                hidden_states = self.dropout(hidden_states)
                logits = classifier(hidden_states)
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs_list.append(log_probs)


        return log_probs_list, output_lengths

    def _gaussian_log_probs(
        self,
        hidden_states: torch.Tensor,
        means: torch.Tensor,
        raw_precision: torch.Tensor,
        mixture_logits: torch.Tensor,
        ignore_var_det: bool = False,
    ) -> torch.Tensor:
        if self.cfg.gaussian_precision_type == "full":
            return self._full_gaussian_log_probs(
                hidden_states=hidden_states,
                means=means,
                raw_precision=raw_precision,
                mixture_logits=mixture_logits,
            )

        precision = F.softplus(raw_precision) + self.cfg.gaussian_precision_floor
        if self.cfg.gaussian_class_dependent_variance:
            precision = precision.unsqueeze(1)
        else:
            precision = precision.view(1, 1, -1)

        diff = hidden_states.unsqueeze(-2).unsqueeze(-2) - means.unsqueeze(0).unsqueeze(0)
        quadratic = torch.sum(diff.square() * precision.unsqueeze(0).unsqueeze(0), dim=-1)
        if ignore_var_det:
            log_norm = 0
        else:
            log_det_precision = torch.sum(torch.log(precision), dim=-1)
            log_det_precision = log_det_precision.squeeze(-1) if log_det_precision.ndim == 2 else log_det_precision
            log_norm = 0.5 * (log_det_precision - self.hidden_size * np.log(2.0 * np.pi))
            if not self.cfg.gaussian_class_dependent_variance:
                log_norm = log_norm.view(1)
            log_norm = log_norm.view(1,1,-1,1)
        component_log_probs = -0.5 * quadratic + log_norm
        if self.num_gaussian_mixtures == 1:
            return component_log_probs.squeeze(-1)
        log_weights = torch.log_softmax(mixture_logits, dim=-1)
        return torch.logsumexp(component_log_probs + log_weights.unsqueeze(0).unsqueeze(0), dim=-1)

    def _make_precision_cholesky(self, raw_precision: torch.Tensor) -> torch.Tensor:
        lower = torch.tril(raw_precision)
        diag = torch.diagonal(lower, dim1=-2, dim2=-1)
        positive_diag = F.softplus(diag) + self.cfg.gaussian_precision_floor
        return lower - torch.diag_embed(diag) + torch.diag_embed(positive_diag)

    def _full_gaussian_log_probs(
        self,
        hidden_states: torch.Tensor,
        means: torch.Tensor,
        raw_precision: torch.Tensor,
        mixture_logits: torch.Tensor,
    ) -> torch.Tensor:
        precision_cholesky = self._make_precision_cholesky(raw_precision)
        diff = hidden_states.unsqueeze(-2).unsqueeze(-2) - means.unsqueeze(0).unsqueeze(0)
        if self.cfg.gaussian_class_dependent_variance:
            projected_diff = torch.einsum("btcmd,cdh->btcmh", diff, precision_cholesky)
            log_det_precision = 2.0 * torch.sum(
                torch.log(torch.diagonal(precision_cholesky, dim1=-2, dim2=-1)),
                dim=-1,
            )
        else:
            projected_diff = torch.einsum("btcmd,dh->btcmh", diff, precision_cholesky)
            log_det_precision = 2.0 * torch.sum(
                torch.log(torch.diagonal(precision_cholesky, dim1=-2, dim2=-1))
            )

        quadratic = torch.sum(projected_diff.square(), dim=-1)
        log_norm = 0.5 * (log_det_precision - self.hidden_size * np.log(2.0 * np.pi))
        if self.cfg.gaussian_class_dependent_variance:
            component_log_probs = -0.5 * quadratic + log_norm.view(1, 1, -1, 1)
        else:
            component_log_probs = -0.5 * quadratic + log_norm
        if self.num_gaussian_mixtures == 1:
            return component_log_probs.squeeze(-1)
        log_weights = torch.log_softmax(mixture_logits, dim=-1)
        return torch.logsumexp(component_log_probs + log_weights.unsqueeze(0).unsqueeze(0), dim=-1)

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
            if model.cfg.gaussian_mixture_model and model.cfg.gaussian_use_posterior_training:
                logprobs = torch.log_softmax(logprobs, dim=-1)

            if self.label_smoothing_scale > 0:
                vocab_size = logprobs.shape[-1]
                probs = torch.exp(logprobs)
                smoothed_probs = (1 - self.label_smoothing_scale) * probs + self.label_smoothing_scale / vocab_size
                logprobs = torch.log(smoothed_probs)


                #logprobs.register_hook(_logprob_grad_hook)

            target_fsa = self.fsa_builder.build_batch(labels).to(logprobs.device)

            # for debugging
            # logprobs_detached = logprobs.detach().clone().requires_grad_(True)
            # lossA = fbw2_loss(logprobs_detached, target_fsa, audio_features_len_for_loss).sum()
            # lossA.backward()
            # gradA = logprobs_detached.grad.detach()
            # gradA_sum = gradA.sum(-1)
            # print(f"pseudo grad*********************************{layer_index}",gradA_sum[0,:10], gradA[0,2,:].sum())
            # #print(f"pseudo frame********************************{layer_index}", gradA[0,3,:])
            # print("gradA shape", gradA.shape)



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


def variance_init_hook(run_ctx, **kwargs):
    run_ctx.sum = {}
    run_ctx.sum_sq = {}
    run_ctx.sum_frames = {}


def variance_finish_hook(run_ctx, **kwargs):
    variance_state = {"return_layers": list(run_ctx.sum.keys()), "means": {}, "variances": {}}
    for layer in sorted(run_ctx.sum.keys(), key=lambda x: (x == -1, x)):
        num_frames = run_ctx.sum_frames[layer]
        if num_frames == 0:
            raise ValueError(f"No frames observed while computing variance for layer {layer}.")
        total_sum = run_ctx.sum[layer].detach().cpu().numpy()
        total_sum_sq = run_ctx.sum_sq[layer].detach().cpu().numpy()
        mean = total_sum / num_frames
        variance = total_sum_sq / num_frames - np.square(mean)
        variance = np.maximum(variance, 0.0)
        mean_filename = f"mean_{layer}.txt"
        with open(mean_filename, "w") as f:
            np.savetxt(f, mean, delimiter=" ")
        filename = f"variance_{layer}.txt"
        with open(filename, "w") as f:
            np.savetxt(f, variance, delimiter=" ")
        variance_state["means"][layer] = torch.from_numpy(mean)
        variance_state["variances"][layer] = torch.from_numpy(variance)
    torch.save(variance_state, "variance_state.pt")


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


def class_stats_init_hook(run_ctx, **kwargs):
    run_ctx.alignment_stream_name = kwargs.get("alignment_stream_name", "alignments")
    run_ctx.compute_variance = kwargs.get("compute_variance", True)
    run_ctx.downsample_factor = int(kwargs.get("downsample_factor", 2))
    run_ctx.sum = {}
    run_ctx.sum_sq = {}
    run_ctx.counts = {}
    run_ctx.pooled_sum = {}
    run_ctx.pooled_sum_sq = {}
    run_ctx.pooled_counts = {}


def class_stats_finish_hook(run_ctx, **kwargs):
    class_stats_state = {
        "return_layers": sorted(run_ctx.sum.keys(), key=lambda x: (x == -1, x)),
        "label_target_size": run_ctx.label_target_size,
        "downsample_factor": run_ctx.downsample_factor,
        "means": {},
        "counts": {},
        "pooled_means": {},
        "variances": {},
        "pooled_variances": {},
    }
    for layer in class_stats_state["return_layers"]:
        counts = run_ctx.counts[layer].detach().cpu()
        pooled_count = int(run_ctx.pooled_counts[layer])
        if pooled_count == 0:
            raise ValueError(f"No frames observed while computing class stats for layer {layer}.")

        pooled_sum = run_ctx.pooled_sum[layer].detach().cpu()
        pooled_mean = pooled_sum / pooled_count
        class_means = torch.empty_like(run_ctx.sum[layer].detach().cpu())
        nonzero_mask = counts > 0
        class_means[nonzero_mask] = run_ctx.sum[layer].detach().cpu()[nonzero_mask] / counts[nonzero_mask].unsqueeze(1)
        class_means[~nonzero_mask] = pooled_mean

        class_stats_state["means"][layer] = class_means
        class_stats_state["counts"][layer] = counts
        class_stats_state["pooled_means"][layer] = pooled_mean

        with open(f"pooled_mean_{layer}.txt", "w") as f:
            np.savetxt(f, pooled_mean.numpy(), delimiter=" ")
        with open(f"class_counts_{layer}.txt", "w") as f:
            np.savetxt(f, counts.numpy(), fmt="%d")

        pooled_sum_sq = run_ctx.pooled_sum_sq[layer].detach().cpu()
        pooled_variance = pooled_sum_sq / pooled_count - pooled_mean.square()
        pooled_variance = torch.clamp(pooled_variance, min=0.0)
        class_stats_state["pooled_variances"][layer] = pooled_variance
        with open(f"pooled_variance_{layer}.txt", "w") as f:
            np.savetxt(f, pooled_variance.numpy(), delimiter=" ")

        if run_ctx.compute_variance:
            class_variances = torch.empty_like(class_means)
            class_variances[nonzero_mask] = (
                run_ctx.sum_sq[layer].detach().cpu()[nonzero_mask] / counts[nonzero_mask].unsqueeze(1)
                - class_means[nonzero_mask].square()
            )
            class_variances[~nonzero_mask] = pooled_variance
            class_variances = torch.clamp(class_variances, min=0.0)
            class_stats_state["variances"][layer] = class_variances

    torch.save(class_stats_state, "class_stats_state.pt")


def class_stats_step(*, model: Model, data, run_ctx, **kwargs):
    alignment_key = run_ctx.alignment_stream_name
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    alignments = data[alignment_key].to(torch.long)
    alignment_lengths = data[f"{alignment_key}:size1"].to(torch.long)

    all_hidden_states, output_lengths = model.extract_hidden_states(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transformed_hidden_states = model.transform_hidden_states(
        all_hidden_states,
        output_lengths,
        update_pca=False,
    )

    run_ctx.label_target_size = model.cfg.label_target_size
    batch_size = raw_audio.shape[0]
    for layer_idx, (layer, hidden_states) in enumerate(zip(model.return_layers, transformed_hidden_states)):
        if model.cfg.variance_normalize:
            if not bool(model._get_variance_is_initialized_buffer(layer_idx)):
                raise RuntimeError(
                    f"Feature variance for layer {layer} is not initialized. "
                    "Provide variance_path or run the variance computation job first."
                )
            mean = model._get_variance_mean_buffer(layer_idx)
            std = torch.sqrt(model._get_variance_buffer(layer_idx).clamp_min(1e-12))
            hidden_states = (hidden_states - mean) / std

        if layer not in run_ctx.sum:
            run_ctx.sum[layer] = torch.zeros(
                model.cfg.label_target_size,
                hidden_states.shape[-1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            run_ctx.counts[layer] = torch.zeros(
                model.cfg.label_target_size,
                device=hidden_states.device,
                dtype=torch.long,
            )
            run_ctx.pooled_sum[layer] = torch.zeros(hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
            run_ctx.pooled_sum_sq[layer] = torch.zeros_like(run_ctx.pooled_sum[layer])
            run_ctx.pooled_counts[layer] = 0
            if run_ctx.compute_variance:
                run_ctx.sum_sq[layer] = torch.zeros_like(run_ctx.sum[layer])

        for b in range(batch_size):
            alignment_len = int(alignment_lengths[b].item())

            downsampled_alignment = alignments[b, :alignment_len:run_ctx.downsample_factor]
            valid_len = min(int(output_lengths[b].item()), int(downsampled_alignment.shape[0]))
            if valid_len <= 0:
                continue
            valid_labels = downsampled_alignment[:valid_len]
            valid_hidden_states = hidden_states[b, :valid_len]
            run_ctx.sum[layer].index_add_(0, valid_labels, valid_hidden_states)
            run_ctx.counts[layer].index_add_(
                0,
                valid_labels,
                torch.ones(valid_len, device=valid_labels.device, dtype=run_ctx.counts[layer].dtype),
            )
            run_ctx.pooled_sum[layer] += valid_hidden_states.sum(dim=0)
            run_ctx.pooled_sum_sq[layer] += valid_hidden_states.square().sum(dim=0)
            run_ctx.pooled_counts[layer] += valid_len
            if run_ctx.compute_variance:
                run_ctx.sum_sq[layer].index_add_(0, valid_labels, valid_hidden_states.square())


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
        if model.cfg.generative_model or model.cfg.gaussian_mixture_model:
            probs = torch.softmax(log_probs, dim=-1)
        else:
            probs = torch.exp(log_probs)
        mask = mask_tensor(probs, audio_features_len).unsqueeze(-1)
        probs = probs * mask
        if layer not in run_ctx.sum_probs:
            run_ctx.sum_probs[layer] = torch.sum(probs, dim=(0, 1))
        else:
            run_ctx.sum_probs[layer] += torch.sum(probs, dim=(0, 1))
