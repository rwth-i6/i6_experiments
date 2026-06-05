import os
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .wav2vec2_hf_pca_dump_cfg import ModelConfig
from . import wav2vec2_pca_label_stats as _label_stats
from ...extra_code.incremental_pca import IncrementalPCA

_HF_CACHE_DIR = "/work/asr3/zyang/share/joerg/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{_HF_CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_CACHE_DIR}/transformers"
os.environ["XDG_CACHE_HOME"] = _HF_CACHE_DIR

def _lengths_to_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device) 
    seq_mask = torch.less(r[None, :], seq_len[:, None])  
    return seq_mask


def _subsample_alignment(labels: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return labels
    return labels[::factor]

class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)

        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "wav2vec2_hf_pca_dump requires the 'transformers' package."
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
        getattr(self, self.pca_component_buffer_names[idx]).copy_(pca.components_.detach())
        getattr(self, self.pca_mean_buffer_names[idx]).copy_(pca.mean.detach())
        getattr(self, self.pca_var_buffer_names[idx]).copy_(pca.var.detach())
        getattr(self, self.pca_init_buffer_names[idx]).fill_(True)

    def _load_pca_state(self, path) -> None:
        if not self.pca:
            raise RuntimeError("Received pca_state_path, but pca_dim is not configured.")
        path_str = path.get_path() if hasattr(path, "get_path") else str(path)
        pca_state = torch.load(path_str, map_location="cpu")
        if pca_state["return_layers"] != self.return_layers:
            raise ValueError(f"PCA state return_layers mismatch: expected {self.return_layers}, got {pca_state['return_layers']}")
        for buffer_name, value in pca_state["buffers"].items():
            getattr(self, buffer_name).copy_(value)

    def export_pca_state(self) -> dict:
        if not self.pca:
            raise RuntimeError("Cannot export PCA state when pca_dim is not configured.")
        buffer_names = self.pca_component_buffer_names + self.pca_mean_buffer_names + self.pca_var_buffer_names + self.pca_init_buffer_names
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
                elif not bool(getattr(self, self.pca_init_buffer_names[idx])):
                    raise RuntimeError(f"PCA components for layer {layer_idx} are not initialized.")
                    
                centered_hidden_states = hidden_states - getattr(self, self.pca_mean_buffer_names[idx])
                transformed_hidden_states.append(
                    centered_hidden_states @ getattr(self, self.pca_component_buffer_names[idx]).T
                )
        else:
            for layer_idx in self.return_layers:
                transformed_hidden_states.append(all_hidden_states[layer_idx])

        return transformed_hidden_states


# ==========================================
# HOOKS FOR STAGE 1: FITTING PCA
# ==========================================
def pca_fit_init_hook(run_ctx, **kwargs):
    config = kwargs.get("config", {})
    run_ctx.model = None
    run_ctx.num_batches = 0
    run_ctx.num_frames = 0
    run_ctx.log_interval_batches = int(config.get("log_interval_batches", 10))

def pca_fit_finish_hook(run_ctx, **kwargs):
    if run_ctx.model is None:
        raise ValueError("No batches were processed while fitting PCA.")
    torch.save(run_ctx.model.export_pca_state(), "pca_state.pt")
    print(
        f"[PCA fit] done: processed {run_ctx.num_batches} batches and {run_ctx.num_frames} frames. "
        "Saved PCA state to pca_state.pt."
    )

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
        update_pca=True, # This is the critical part that actually trains the IncrementalPCA
    )
    run_ctx.model = model
    run_ctx.num_batches += 1
    run_ctx.num_frames += int(torch.sum(output_lengths).item())
    if run_ctx.log_interval_batches > 0 and run_ctx.num_batches % run_ctx.log_interval_batches == 0:
        print(
            f"[PCA fit] progress: batches={run_ctx.num_batches}, "
            f"frames={run_ctx.num_frames}, last_batch_size={int(raw_audio.shape[0])}"
        )


# ==========================================
# HOOKS FOR STAGE 2: FEATURE DUMPING
# ==========================================
def forward_init_hook(run_ctx, **kwargs):
    """
    Using the standard 'forward_init_hook' name so get_forward_config finds it automatically
    when we set decoder=network_module.
    """
    config = kwargs.get("config", {})
    output_filename = config.get("output_filename", "features_and_alignments.hdf5")
    run_ctx.target_layer = config.get("target_layer", -1)
    run_ctx.alignment_subsample_factor = int(config.get("alignment_subsample_factor", 2))
    
    # Open HDF5 file to store the paired sequences
    run_ctx.output_file = h5py.File(output_filename, "w")

def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.output_file.close()
    print("Finished dumping features to HDF5.")

def forward_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    
    # We mapped HDF to "alignments" in the dataset builder
    alignments = data.get("alignments")
    if alignments is None:
        raise ValueError("Alignments not found in data stream. Check hdf_stream_name in Sisyphus.")
    
    alignments_len = data["alignments:size1"].to(torch.long)
    tags = data["seq_tag"]

    # 1. Forward pass through Wav2Vec2
    all_hidden_states, output_lengths = model.extract_hidden_states(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    
    # 2. Apply pre-fitted PCA (update_pca=False)
    transformed_hidden_states = model.transform_hidden_states(
        all_hidden_states,
        output_lengths,
        update_pca=False,
    )

    layer_idx = model.return_layers.index(run_ctx.target_layer)
    layer_features = transformed_hidden_states[layer_idx]

    # 3. Dump batch to HDF5
    for batch_idx, tag in enumerate(tags):
        seq_len = int(output_lengths[batch_idx].item())
        aln_len = int(alignments_len[batch_idx].item())
        raw_alignments = alignments[batch_idx, :aln_len].detach().cpu().to(torch.long)
        subsampled_alignments = _subsample_alignment(raw_alignments, run_ctx.alignment_subsample_factor)
        effective_len = min(seq_len, int(subsampled_alignments.shape[0]))

        features_np = layer_features[batch_idx, :effective_len].detach().cpu().numpy()
        aln_np = subsampled_alignments[:effective_len].numpy()

        # h5py creates nested groups automatically if `tag` contains slashes (e.g. train-other-960/...)
        grp = run_ctx.output_file.create_group(tag)
        grp.create_dataset("features", data=features_np, compression="gzip")
        grp.create_dataset("alignments", data=aln_np, compression="gzip")


# ==========================================
# HOOKS FOR STAGE 3: PCA LABEL STATS
# ==========================================
def label_stats_init_hook(run_ctx, **kwargs):
    _label_stats.forward_init_hook(run_ctx, **kwargs)


def label_stats_finish_hook(run_ctx, **kwargs):
    _label_stats.forward_finish_hook(run_ctx, **kwargs)


def label_stats_step(*, model: Model, data, run_ctx, **kwargs):
    _label_stats.forward_step(model=model, data=data, run_ctx=run_ctx, **kwargs)
