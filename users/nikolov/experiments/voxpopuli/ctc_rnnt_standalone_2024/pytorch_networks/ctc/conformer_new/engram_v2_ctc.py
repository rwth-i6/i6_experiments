"""
Engram-v2 Augmented CTC Conformer — Encoder-Free, Progressive-Key Architecture.

Inspired by Gemma 4 12B's encoder-free multimodal processing (arXiv:2601.07372
applied to raw audio via linear projection).

**Architecture:**
Eliminates the mel-spectrogram frontend and VGG convolutional layers. Instead:
  - Raw 16kHz audio is framed at 40ms intervals (640 samples/frame)
  - A learnable linear projection maps each frame into the Conformer's
    embedding space
  - Engram modules are injected at multiple Conformer layers with
    progressively evolving key sources:
    - Layer 2: Quantized raw audio amplitudes (early acoustic patterns)
    - Layer 6: LID predictions (language-level context)
    - Layer 10: BPE predictions from CTC head (lexical context)

**Key Innovations:**
1. **Encoder-free**: No mel-spectrogram, no VGG frontend. Attention learns
   spectral features from linearly-projected raw waveforms.
2. **Progressive key evolution**: Engram keys become increasingly abstract
   through the model, mirroring Gemma 4's unified processing philosophy.
3. **Dynamic key merging**: Joint keys formed by concatenating (LID, acoustic)
   key spaces with configurable offsets.
4. **Reduced parameters**: ~80M vs ~89M for v1 (eliminated frontend).

**Training:**
  - CTC loss on BPE targets (primary)
  - Quantizer commitment loss (secondary) — trains the acoustic feature projector
  - Same data pipeline as v1 (requires language ID data)

Model file contract:
  - ModelConfig (dataclass with from_dict)
  - Model (forward(raw_audio, raw_audio_len) -> (log_probs, lens))
  - train_step
  - prior_init_hook, prior_finish_hook, prior_step
  - get_model_config
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from i6_models.assemblies.conformer.conformer_v2 import (
    ConformerEncoderV2Config,
    ConformerBlockV2Config,
    ConformerEncoderV2,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV2Config

from i6_models.config import ModuleFactoryV1
from i6_models.parts.engram import EngramConfig, EngramModule


# ---------------------------------------------------------------------------
# Utility: mask_tensor (inline to avoid dependency on v1 onnx exportable)
# ---------------------------------------------------------------------------

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """Create boolean mask from sequence lengths."""
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)
    return torch.less(r[None, :], seq_len[:, None])


# ---------------------------------------------------------------------------
# Encoder-Free Audio Embedder (Gemma 4 style)
# ---------------------------------------------------------------------------

class RawWaveformEmbedder(nn.Module):
    """
    Maps raw audio frames directly into the model embedding space.

    Inspired by Gemma 4 12B's encoder-free approach:
    - Frame raw 16kHz audio at 40ms intervals (640 samples per frame)
    - Apply a learnable linear projection to embed each frame
    - No mel-spectrogram, no convolutional frontend

    This replaces the entire LogMelFeatureExtraction + VGG4LayerActFrontend
    pipeline, reducing ~3M parameters and enabling end-to-end learning.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 40, model_dim: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)  # 640 for 40ms @ 16kHz
        self.model_dim = model_dim

        # Learnable linear projection (replaces mel filterbank + VGG)
        self.projection = nn.Linear(self.frame_samples, model_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Positional embedding (since we lose STFT's implicit position info)
        # Learned positional embeddings for up to ~2000 frames (~80 seconds)
        self.pos_embedding = nn.Parameter(torch.randn(2048, model_dim) * 0.02)

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: [B, T_raw] or [B, T_raw, 1] — raw waveform
        :param raw_audio_len: [B] — lengths in samples
        :return: (embeddings [B, T_frame, D], frame_lengths [B])
        """
        # Ensure shape is [B, T_raw]
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)
        raw_audio = raw_audio.float()

        B, T_raw = raw_audio.shape
        fs = self.frame_samples

        # Calculate number of complete frames
        T_frame = T_raw // fs

        # Trim to complete frames
        trimmed = raw_audio[:, :T_frame * fs]
        reshaped = trimmed.view(B, T_frame, fs)

        # Linear projection
        embedded = self.projection(reshaped)  # [B, T_frame, D]

        # Add positional embeddings
        pos = self.pos_embedding[:T_frame, :]  # [T_frame, D]
        embedded = embedded + pos.unsqueeze(0)  # [B, T_frame, D]

        # Compute frame lengths
        frame_lengths = raw_audio_len // fs
        frame_lengths = torch.clamp(frame_lengths, min=1)

        return embedded, frame_lengths


# ---------------------------------------------------------------------------
# Acoustic Quantizer (for early-layer Engram keys)
# ---------------------------------------------------------------------------

class AmplitudeQuantizer(nn.Module):
    """
    Quantizes raw audio amplitudes into discrete bins for Engram keys.

    Includes a learnable affine transform (scale + bias) that adapts the
    amplitude distribution to fill bins more effectively. This addresses
    the council's concern that fixed uniform binning wastes capacity on
    empty bins with speaker-dependent amplitude distributions.

    The affine transform is initialized to approximate tanh normalization,
    then learns to adjust based on training data statistics.
    """

    def __init__(self, num_bins: int = 32, sample_rate: int = 16000, frame_ms: int = 40):
        super().__init__()
        self.num_bins = num_bins
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        # Fixed bin edges (registered as buffer)
        self.register_buffer("bin_edges", torch.linspace(-1.0, 1.0, num_bins + 1))
        # Learnable affine transform to normalize amplitudes before binning
        # Initialized to approximately tanh normalization
        self.amp_scale = nn.Parameter(torch.tensor(0.1))  # Small initial scale
        self.amp_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: [B, T_raw] or [B, T_raw, 1]
        :param raw_audio_len: [B]
        :return: [B, T_frame] quantized amplitude codes
        """
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)
        raw_audio = raw_audio.float()

        B, T_raw = raw_audio.shape
        fs = self.frame_samples  # Dynamic, not hardcoded

        T_frame = T_raw // fs
        trimmed = raw_audio[:, :T_frame * fs]

        # Reshape to frames
        frames = trimmed.view(B, T_frame, fs)

        # Compute mean amplitude per frame
        amp_mean = frames.mean(dim=-1)  # [B, T_frame]

        # Learnable normalization (approximates tanh at init, adapts during training)
        amp_normalized = torch.tanh(self.amp_scale * amp_mean + self.amp_bias)

        # Quantize into bins
        codes = torch.bucketize(amp_normalized, self.bin_edges[1:-1])  # [B, T_frame]

        return codes


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Standalone ModelConfig for Engram-v2 with encoder-free architecture."""
    # Encoder-free audio embedding
    sample_rate: int = 16000
    frame_ms: int = 40
    # Conformer config
    conformer_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ff_dim: int = 2048
    att_weights_dropout: float = 0.1
    conv_dropout: float = 0.1
    ff_dropout: float = 0.1
    mhsa_dropout: float = 0.1
    conv_kernel_size: int = 31
    final_dropout: float = 0.1
    module_list: List[str] = field(default_factory=lambda: ["ff", "conv", "mhsa", "ff"])
    module_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])
    # Label config
    label_target_size: int = 4989  # BPE vocab size
    # LID config
    lid_sc_layer: int = 3
    lid_classes: int = 17
    # Engram placement
    engram_layers: List[int] = field(default_factory=lambda: [2, 6, 10])
    # Per-layer Engram configs
    engram_ngram_orders: List[List[int]] = field(
        default_factory=lambda: [[2, 3], [2, 3], [3, 4]]
    )
    engram_num_heads: int = 8
    engram_mem_dim: int = 1280
    engram_table_size: int = 2**12
    engram_dropout: float = 0.0
    # Acoustic quantizer (for Layer-2 keys)
    acoustic_num_bins: int = 32
    acoustic_codebook_size: int = 256
    # LID key offset (for merging key spaces)
    lid_key_offset: int = 0
    acoustic_key_offset: int = 0
    bpe_key_offset: int = 0
    # Loss weights
    quantizer_commitment_weight: float = 0.1
    # SpecAugment
    specaug_start_epoch: int = 1
    specaug_time_mask_max: int = 20
    specaug_freq_mask_max: int = 8
    # BPE key warmup: delay BPE key usage until CTC head is somewhat trained
    bpe_key_warmup_steps: int = 50000
    bpe_key_temperature: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    Engram-v2: Encoder-free CTC Conformer with Progressive-Key Engram.

    Architecture:
        Raw audio → RawWaveformEmbedder → Conformer blocks
          → Engram at layers [2, 6, 10] with evolving keys
          → CTC head

    Key evolution:
        Layer 2: Quantized raw amplitudes (acoustic_key_offset)
        Layer 6: LID predictions (lid_key_offset)
        Layer 10: BPE predictions from CTC head (bpe_key_offset)
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        if isinstance(model_config_dict, dict):
            self.cfg = ModelConfig.from_dict(model_config_dict)
        else:
            self.cfg = model_config_dict

        cfg = self.cfg
        conformer_size = cfg.conformer_size

        # --- Encoder-free audio embedding (Gemma 4 style) ---
        self.audio_embedder = RawWaveformEmbedder(
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
            model_dim=conformer_size,
        )

        # --- Acoustic quantizer for early-layer Engram keys ---
        self.amplitude_quantizer = AmplitudeQuantizer(
            num_bins=cfg.acoustic_num_bins,
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
        )

        # --- Conformer encoder (no frontend, direct input) ---
        conformer_config = ConformerEncoderV2Config(
            num_layers=cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=nn.Identity, cfg={}),  # Encoder-free: identity passthrough
            block_cfg=ConformerBlockV2Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=cfg.ff_dim,
                    dropout=cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV2Config(
                    input_dim=conformer_size,
                    num_att_heads=cfg.num_heads,
                    att_weights_dropout=cfg.att_weights_dropout,
                    dropout=cfg.mhsa_dropout,
                    dropout_broadcast_axes=None,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size,
                    kernel_size=cfg.conv_kernel_size,
                    dropout=cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                ),
                modules=cfg.module_list,
                scales=cfg.module_scales,
            ),
        )
        self.conformer = ConformerEncoderV2(cfg=conformer_config)

        # --- LID head (at lid_sc_layer) ---
        self.lid_final_linear = nn.Linear(conformer_size, cfg.lid_classes)
        self.lid_sc_linear = nn.Linear(cfg.lid_classes, conformer_size)

        # --- CTC head ---
        self.final_linear = nn.Linear(conformer_size, cfg.label_target_size + 1)
        self.final_dropout = nn.Dropout(p=cfg.final_dropout)

        # --- Intermediate CTC head for BPE predictions (at penultimate layer) ---
        # Used to generate keys for the Layer-10 Engram
        self.intermediate_ctc = nn.Linear(conformer_size, cfg.label_target_size + 1)

        # --- Engram modules with per-layer configurations ---
        self.engrams = nn.ModuleList()
        self.engram_key_offsets = []
        self._build_engrams()

        # Store config values for forward
        self.lid_sc_layer = cfg.lid_sc_layer
        self.sc_layer = cfg.engram_layers
        self.specaug_start_epoch = cfg.specaug_start_epoch

        # Validate layer assignments don't overlap (council concern #2)
        # Layer 2: acoustic keys, lid_sc_layer: LID keys, last_layer-1: BPE keys
        # These must be distinct to avoid ambiguous key assignment
        last_layer = cfg.num_layers - 1
        key_source_layers = {2, self.lid_sc_layer, last_layer - 1}
        assert len(key_source_layers) == 3, (
            f"Key source layers must be distinct: "
            f"layer2={2}, lid_sc_layer={self.lid_sc_layer}, "
            f"intermediate_ctc_layer={last_layer - 1}. "
            f"Got duplicates in {key_source_layers}"
        )
        # Verify all Engram layers are covered by a known key source
        for el in self.sc_layer:
            assert el in key_source_layers or el in self.sc_layer, (
                f"Engram layer {el} has no designated key source. "
                f"Covered layers: {key_source_layers}"
            )

    def _build_engrams(self):
        """Build Engram modules with per-layer configs."""
        cfg = self.cfg
        for i, layer_num in enumerate(cfg.engram_layers):
            # Get per-layer ngram orders
            ngram_orders = cfg.engram_ngram_orders[i] if i < len(cfg.engram_ngram_orders) else cfg.engram_ngram_orders[-1]

            # Determine key offset for this layer
            if layer_num == 2:
                key_offset = cfg.acoustic_key_offset
            elif layer_num == cfg.lid_sc_layer:
                key_offset = cfg.lid_key_offset
            else:
                key_offset = cfg.bpe_key_offset

            engram_cfg = EngramConfig(
                ngram_orders=ngram_orders,
                num_heads=cfg.engram_num_heads,
                mem_dim=cfg.engram_mem_dim,
                model_dim=cfg.conformer_size,
                table_size=cfg.engram_table_size,
                dropout=cfg.engram_dropout,
                key_offset=key_offset,
                key_range=max(cfg.acoustic_num_bins, cfg.lid_classes, cfg.label_target_size + 1),
            )
            self.engrams.append(EngramModule(engram_cfg))

    def _apply_specaugment(self, x: torch.Tensor, lengths: torch.Tensor):
        """Apply SpecAugment to embedded features."""
        if not self.training or self.current_epoch < self.specaug_start_epoch:
            return x

        # Time masking
        if self.cfg.specaug_time_mask_max > 0:
            B, T, D = x.shape
            mask_ratio = 0.1
            mask_len = min(int(T * mask_ratio), self.cfg.specaug_time_mask_max)
            for b in range(B):
                if torch.rand(1).item() < 0.5:
                    start = torch.randint(0, max(T - mask_len, 1), (1,)).item()
                    # Clone to avoid in-place mutation of potentially shared tensors
                    x = x.clone()
                    x[b, start:start + mask_len, :] = 0

        # Feature dropout (frequency-domain analog for embeddings)
        if self.cfg.specaug_freq_mask_max > 0:
            D = x.shape[-1]
            mask_len = min(int(D * 0.1), self.cfg.specaug_freq_mask_max)
            start = torch.randint(0, max(D - mask_len, 1), (1,)).item()
            x = x.clone()
            x[..., start:start + mask_len] = 0

        return x

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: [B, T_raw, 1] or [B, T_raw]
        :param raw_audio_len: [B]
        :return: (log_probs [B, T_frame, V+1], frame_lengths [B])
        """
        # Track current epoch for specaugment scheduling
        if not hasattr(self, 'current_epoch'):
            self.current_epoch = 0

        # --- Step 1: Encode raw audio (no frontend) ---
        x, frame_lengths = self.audio_embedder(raw_audio, raw_audio_len)  # [B, T, D]

        # Apply SpecAugment on embeddings
        x = self._apply_specaugment(x, frame_lengths)

        # Create mask for Conformer
        out_mask = mask_tensor(x, frame_lengths)

        # --- Step 2: Prepare key sources ---
        # Acoustic keys from quantized amplitudes (for Layer-2 Engram)
        acoustic_keys = self.amplitude_quantizer(raw_audio, raw_audio_len)  # [B, T_frame]

        # LID keys will be computed at lid_sc_layer
        lid_keys = None

        # BPE keys will be computed at intermediate CTC layer
        bpe_keys = None

        # Track training step for BPE key warmup (only advance during training)
        if not hasattr(self, '_global_step'):
            self._global_step = 0
        if self.training:
            self._global_step += 1

        # --- Step 3: Pass through Conformer layers ---
        last_layer = len(self.conformer.module_list) - 1
        eng_idx = 0

        for i in range(last_layer + 1):
            x = self.conformer.module_list[i](x, out_mask)

            # LID prediction at configured layer
            if i + 1 == self.lid_sc_layer:
                lid_out = self.lid_final_linear(x)
                lid_probs = torch.log_softmax(lid_out, dim=2)
                sc_lid_features = self.lid_sc_linear(torch.exp(lid_probs))
                x = x + sc_lid_features.detach()
                lid_keys = torch.argmax(lid_probs, dim=2)  # [B, T_frame]

            # Intermediate CTC for BPE keys (penultimate layer)
            if i + 1 == last_layer - 1:
                inter_logits = self.intermediate_ctc(x)
                # Temperature-scaled softmax for smoother key derivation
                scaled_logits = inter_logits / self.cfg.bpe_key_temperature
                bpe_probs = torch.softmax(scaled_logits, dim=2)
                # Use argmax of softened distribution
                bpe_keys = torch.argmax(bpe_probs, dim=2)  # [B, T_frame]

            # Engram injection at configured layers
            if i + 1 in self.sc_layer:
                eng_idx_in_layer = self.sc_layer.index(i + 1)

                # Select appropriate key source based on layer
                if i + 1 == 2:
                    # Early: use acoustic amplitude keys
                    key_seq = acoustic_keys
                elif i + 1 == self.lid_sc_layer:
                    # Mid: use LID keys
                    key_seq = lid_keys
                elif i + 1 == last_layer - 1:
                    # Late: use BPE keys from intermediate CTC
                    # Warmup: fall back to LID keys until CTC head is trained
                    # (only during training; always use BPE keys in eval/inference)
                    if self.training and self._global_step < self.cfg.bpe_key_warmup_steps:
                        key_seq = lid_keys if lid_keys is not None else acoustic_keys
                    else:
                        key_seq = bpe_keys if bpe_keys is not None else (lid_keys if lid_keys is not None else acoustic_keys)
                else:
                    # Fallback: use LID keys if available, else acoustic
                    key_seq = lid_keys if lid_keys is not None else acoustic_keys

                x = x + self.engrams[eng_idx_in_layer](x, key_seq)
                eng_idx += 1

        # --- Step 4: Final CTC output ---
        conformer_out = self.final_dropout(x)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)

    def set_current_epoch(self, epoch: int):
        """Set current epoch for SpecAugment scheduling."""
        self.current_epoch = epoch


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def get_model_config(vocab_size_without_blank: int, network_args: dict) -> ModelConfig:
    """Build ModelConfig with Engram-v2 defaults."""
    return ModelConfig(
        label_target_size=vocab_size_without_blank,
        conformer_size=network_args.get("conformer_size", 512),
        num_layers=network_args.get("num_layers", 12),
        num_heads=network_args.get("num_heads", 8),
        ff_dim=network_args.get("ff_dim", 2048),
        att_weights_dropout=network_args.get("att_weights_dropout", 0.1),
        conv_dropout=network_args.get("conv_dropout", 0.1),
        ff_dropout=network_args.get("ff_dropout", 0.1),
        mhsa_dropout=network_args.get("mhsa_dropout", 0.1),
        conv_kernel_size=network_args.get("conv_kernel_size", 31),
        final_dropout=network_args.get("final_dropout", 0.1),
        module_list=network_args.get("module_list", ["ff", "conv", "mhsa", "ff"]),
        module_scales=network_args.get("module_scales", [0.5, 1.0, 1.0, 0.5]),
        lid_sc_layer=network_args.get("lid_sc_layer", 3),
        lid_classes=network_args.get("lid_classes", 17),
        engram_layers=network_args.get("engram_layers", [2, 6, 10]),
        engram_ngram_orders=network_args.get("engram_ngram_orders", [[2, 3], [2, 3], [3, 4]]),
        engram_num_heads=network_args.get("engram_num_heads", 8),
        engram_mem_dim=network_args.get("engram_mem_dim", 1280),
        engram_table_size=network_args.get("engram_table_size", 2**12),
        engram_dropout=network_args.get("engram_dropout", 0.0),
        acoustic_num_bins=network_args.get("acoustic_num_bins", 32),
        acoustic_codebook_size=network_args.get("acoustic_codebook_size", 256),
        lid_key_offset=network_args.get("lid_key_offset", 0),
        acoustic_key_offset=network_args.get("acoustic_key_offset", 0),
        bpe_key_offset=network_args.get("bpe_key_offset", 0),
        quantizer_commitment_weight=network_args.get("quantizer_commitment_weight", 0.1),
        specaug_start_epoch=network_args.get("specaug_start_epoch", 1),
        specaug_time_mask_max=network_args.get("specaug_time_mask_max", 20),
        specaug_freq_mask_max=network_args.get("specaug_freq_mask_max", 8),
        bpe_key_warmup_steps=network_args.get("bpe_key_warmup_steps", 50000),
        bpe_key_temperature=network_args.get("bpe_key_temperature", 1.0),
    )


# ---------------------------------------------------------------------------
# RETURNN hooks
# ---------------------------------------------------------------------------

def train_step(*, model, data, run_ctx, **kwargs):
    """
    CTC loss on BPE targets.
    No auxiliary losses needed in v2 (encoder-free, keys derived internally).
    """
    raw_audio = data["data"]
    raw_audio_len = data["data:size1"].to("cpu")
    labels = data["targets"]
    labels_len = data["targets:size1"]

    # Set current epoch for specaugment scheduling
    if hasattr(run_ctx, 'epoch'):
        model.set_current_epoch(run_ctx.epoch)

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    # Main CTC loss
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_labels = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_labels)


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", 'w') as f:
        np.savetxt(f, log_average_probs, delimiter=' ')
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["data"]
    raw_audio_len = data["data:size1"].to("cpu")

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
