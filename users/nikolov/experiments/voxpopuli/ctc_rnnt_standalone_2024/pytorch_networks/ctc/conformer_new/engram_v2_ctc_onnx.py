"""
ONNX-exportable recognition wrapper for the Engram-v2 CTC Conformer.

Architecture is identical to engram_v2_ctc.Model so that checkpoints
load correctly (same state_dict keys).

Uses the same encoder-free RawWaveformEmbedder and progressive-key Engram
injection. forward() returns (log_probs, frame_lengths) — exactly 2 outputs —
for compatibility with the flashlight_ctc_v1_onnx_v2 decoder.

State_dict keys match engram_v2_ctc.Model exactly.
"""

import numpy as np
import torch
from torch import nn
from typing import Tuple, List
from dataclasses import dataclass, field

from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v2 import (
    ConformerEncoderV2Config,
    ConformerBlockV2Config,
    ConformerEncoderV2,
)

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV2Config
from i6_models.parts.engram import EngramConfig, EngramModule


# ---------------------------------------------------------------------------
# ModelConfig — must match engram_v2_ctc.ModelConfig exactly
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    sample_rate: int = 16000
    frame_ms: int = 40
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
    label_target_size: int = 4989
    lid_sc_layer: int = 3
    lid_classes: int = 17
    engram_layers: List[int] = field(default_factory=lambda: [2, 6, 10])
    engram_ngram_orders: List[List[int]] = field(
        default_factory=lambda: [[2, 3], [2, 3], [3, 4]]
    )
    engram_num_heads: int = 8
    engram_mem_dim: int = 1280
    engram_table_size: int = 2**12
    engram_dropout: float = 0.0
    acoustic_num_bins: int = 32
    acoustic_codebook_size: int = 256
    lid_key_offset: int = 0
    acoustic_key_offset: int = 0
    bpe_key_offset: int = 0
    quantizer_commitment_weight: float = 0.1
    specaug_start_epoch: int = 1
    specaug_time_mask_max: int = 20
    specaug_freq_mask_max: int = 8
    bpe_key_warmup_steps: int = 50000
    bpe_key_temperature: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ---------------------------------------------------------------------------
# Encoder-Free Audio Embedder (must match training model)
# ---------------------------------------------------------------------------

class RawWaveformEmbedder(nn.Module):
    """
    ONNX-exportable raw waveform embedder (identical to training model).

    Frames raw 16kHz audio at 40ms intervals, projects via learned linear layer.
    No mel-spectrogram, no VGG frontend — pure encoder-free approach.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 40, model_dim: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.model_dim = model_dim

        self.projection = nn.Linear(self.frame_samples, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(2048, model_dim) * 0.02)

    def forward(self, raw_audio: torch.Tensor, length: torch.Tensor):
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)
        raw_audio = raw_audio.float()

        B, T_raw = raw_audio.shape
        fs = self.frame_samples
        T_frame = T_raw // fs

        trimmed = raw_audio[:, :T_frame * fs]
        reshaped = trimmed.view(B, T_frame, fs)

        embedded = self.projection(reshaped)
        pos = self.pos_embedding[:T_frame, :]
        embedded = embedded + pos.unsqueeze(0)

        frame_lengths = length // fs
        frame_lengths = torch.clamp(frame_lengths, min=1)

        return embedded, frame_lengths


# ---------------------------------------------------------------------------
# Amplitude Quantizer (must match training model)
# ---------------------------------------------------------------------------

class AmplitudeQuantizer(nn.Module):
    """Quantizes raw audio amplitudes into discrete bins for Engram keys.

    Includes a learnable affine transform (scale + bias) to adapt amplitude
    distribution. Mirrors the training model exactly for state_dict compatibility.
    """

    def __init__(self, num_bins: int = 32, sample_rate: int = 16000, frame_ms: int = 40):
        super().__init__()
        self.num_bins = num_bins
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.register_buffer("bin_edges", torch.linspace(-1.0, 1.0, num_bins + 1))
        self.amp_scale = nn.Parameter(torch.tensor(0.1))
        self.amp_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)
        raw_audio = raw_audio.float()

        B, T_raw = raw_audio.shape
        fs = self.frame_samples

        T_frame = T_raw // fs
        trimmed = raw_audio[:, :T_frame * fs]
        frames = trimmed.view(B, T_frame, fs)

        amp_mean = frames.mean(dim=-1)
        amp_normalized = torch.tanh(self.amp_scale * amp_mean + self.amp_bias)
        codes = torch.bucketize(amp_normalized, self.bin_edges[1:-1])

        return codes


# ---------------------------------------------------------------------------
# Mask tensor helper
# ---------------------------------------------------------------------------

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)
    return torch.less(r[None, :], seq_len[:, None])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(torch.nn.Module):
    """
    ONNX-exportable Engram-v2 CTC Conformer.

    State_dict keys match engram_v2_ctc.Model exactly.
    forward() returns (log_probs, frame_lengths) for ONNX export.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        cfg = self.cfg
        conformer_size = cfg.conformer_size

        # --- Encoder-free audio embedding ---
        self.audio_embedder = RawWaveformEmbedder(
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
            model_dim=conformer_size,
        )

        # --- Acoustic quantizer ---
        self.amplitude_quantizer = AmplitudeQuantizer(
            num_bins=cfg.acoustic_num_bins,
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
        )

        # --- Conformer encoder (identity frontend) ---
        conformer_config = ConformerEncoderV2Config(
            num_layers=cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=nn.Identity, cfg={}),
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

        # --- LID head ---
        self.lid_final_linear = nn.Linear(conformer_size, cfg.lid_classes)
        self.lid_sc_linear = nn.Linear(cfg.lid_classes, conformer_size)

        # --- CTC head ---
        self.final_linear = nn.Linear(conformer_size, cfg.label_target_size + 1)
        self.final_dropout = nn.Dropout(p=cfg.final_dropout)

        # --- Intermediate CTC head ---
        self.intermediate_ctc = nn.Linear(conformer_size, cfg.label_target_size + 1)

        # --- Engram modules ---
        self.engrams = nn.ModuleList()
        self._build_engrams()

        self.lid_sc_layer = cfg.lid_sc_layer
        self.sc_layer = cfg.engram_layers
        self.specaug_start_epoch = cfg.specaug_start_epoch

        # Validate layer assignments don't overlap
        last_layer = cfg.num_layers - 1
        key_source_layers = {2, self.lid_sc_layer, last_layer - 1}
        assert len(key_source_layers) == 3, (
            f"Key source layers must be distinct: "
            f"layer2={2}, lid_sc_layer={self.lid_sc_layer}, "
            f"intermediate_ctc_layer={last_layer - 1}. "
            f"Got duplicates in {key_source_layers}"
        )

    def _build_engrams(self):
        """Build Engram modules with per-layer configs (must match training)."""
        cfg = self.cfg
        for i, layer_num in enumerate(cfg.engram_layers):
            ngram_orders = cfg.engram_ngram_orders[i] if i < len(cfg.engram_ngram_orders) else cfg.engram_ngram_orders[-1]

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

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T, 1]
        :param raw_audio_len: [B]
        :return: (log_probs [B, T, V+1], frame_lengths [B])
        """
        # --- Encode raw audio (no frontend) ---
        x, frame_lengths = self.audio_embedder(raw_audio, raw_audio_len)

        out_mask = mask_tensor(x, frame_lengths)

        # --- Prepare key sources ---
        acoustic_keys = self.amplitude_quantizer(raw_audio, raw_audio_len)
        lid_keys = None
        bpe_keys = None

        last_layer = len(self.conformer.module_list) - 1
        eng_idx = 0

        for i in range(last_layer + 1):
            x = self.conformer.module_list[i](x, out_mask)

            if i + 1 == self.lid_sc_layer:
                lid_out = self.lid_final_linear(x)
                lid_probs = torch.log_softmax(lid_out, dim=2)
                sc_lid_features = self.lid_sc_linear(torch.exp(lid_probs))
                x = x + sc_lid_features.detach()
                lid_keys = torch.argmax(lid_probs, dim=2)

            if i + 1 == last_layer - 1:
                inter_logits = self.intermediate_ctc(x)
                scaled_logits = inter_logits / self.cfg.bpe_key_temperature
                bpe_probs = torch.softmax(scaled_logits, dim=2)
                bpe_keys = torch.argmax(bpe_probs, dim=2)

            if i + 1 in self.sc_layer:
                eng_idx_in_layer = self.sc_layer.index(i + 1)

                if i + 1 == 2:
                    key_seq = acoustic_keys
                elif i + 1 == self.lid_sc_layer:
                    key_seq = lid_keys
                elif i + 1 == last_layer - 1:
                    key_seq = bpe_keys
                else:
                    key_seq = lid_keys if lid_keys is not None else acoustic_keys

                x = x + self.engrams[eng_idx_in_layer](x, key_seq)
                eng_idx += 1

        conformer_out = self.final_dropout(x)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


# ---------------------------------------------------------------------------
# RETURNN hooks
# ---------------------------------------------------------------------------

def train_step(**kwargs):
    raise NotImplementedError("engram_v2_ctc_onnx is for recognition only.")


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["data"]
    raw_audio_len = data["data:size1"].to("cpu")

    logprobs, frame_lengths = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(frame_lengths)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def get_model_config(vocab_size_without_blank: int, network_args: dict) -> ModelConfig:
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
