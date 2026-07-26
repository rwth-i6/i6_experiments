"""
ONNX-exportable recognition wrapper for the GPT-2 Ngram CTC model.

Architecture is identical to gpt2_ngram_ctc.Model so that checkpoints
load correctly (same state_dict keys).

forward() returns (log_probs, audio_features_len) — exactly 2 outputs —
for compatibility with the flashlight_ctc_v1_onnx_v2 decoder.

Both Engram injection (at layers 2 and 6) and the LID head (at layer 6)
are preserved during inference, matching training-time computation.

Key differences from training model:
    - No training-specific branches (no specaugment, no returnn.torch)
    - train_step raises NotImplementedError (recognition only)
    - Operations chosen for ONNX traceability (-1e9 instead of -inf)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from i6_models.parts.engram import EngramConfig, EngramModule


# ============================================================================
# GPT-2 Configuration
# ============================================================================

@dataclass
class GPT2Config:
    """GPT-2 transformer block configuration."""
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    ff_dim: int = 3072
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position: int = 4096


# ============================================================================
# GPT-2 Components
# ============================================================================

class GPT2Attention(nn.Module):
    """Causal multi-head self-attention (GPT-2 style)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask_4d = causal_mask.unsqueeze(0).unsqueeze(0)
        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(1)
            mask_4d = mask_4d & key_mask

        att = att.masked_fill(~mask_4d, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.c_proj(out))
        return out


class GPT2MLP(nn.Module):
    """Feed-forward network (GPT-2 style, with GELU)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ff_dim)
        self.c_proj = nn.Linear(config.ff_dim, config.n_embd)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 transformer block (pre-LayerNorm)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = GPT2Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Must match gpt2_ngram_ctc.ModelConfig exactly for checkpoint compatibility."""
    gpt2_config: GPT2Config
    sample_rate: int
    frame_ms: int
    frame_size: int
    label_target_size: int
    final_dropout: float
    engram_layers: List[int] = field(default_factory=lambda: [2, 6])
    engram_lid_layer: int = 6
    engram_audio_bins: int = 32
    engram_lid_classes: int = 17
    engram_ngram_orders: List[int] = field(default_factory=lambda: [2, 3])
    engram_num_heads: int = 8
    engram_mem_dim: int = 1280
    engram_table_size: int = 2**12
    engram_dropout: float = 0.0
    specaug_start_epoch: int = 1

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["gpt2_config"] = GPT2Config(**d["gpt2_config"])
        return cls(**d)


# ============================================================================
# Model
# ============================================================================

class Model(torch.nn.Module):
    """
    ONNX-exportable GPT-2 Ngram CTC model.

    State_dict keys match gpt2_ngram_ctc.Model exactly.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict) if isinstance(model_config_dict, dict) else model_config_dict
        gpt2_cfg = self.cfg.gpt2_config
        frame_size = self.cfg.frame_size
        model_dim = gpt2_cfg.n_embd

        # Audio projection
        self.audio_proj = nn.Linear(frame_size, model_dim)

        # Positional embeddings
        self.position_embed = nn.Embedding(gpt2_cfg.max_position, model_dim)
        self.drop = nn.Dropout(gpt2_cfg.embd_pdrop)

        # GPT-2 blocks
        self.blocks = nn.ModuleList([GPT2Block(gpt2_cfg) for _ in range(gpt2_cfg.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(model_dim, eps=gpt2_cfg.layer_norm_eps)

        # CTC head
        self.ctc_head = nn.Linear(model_dim, self.cfg.label_target_size + 1)

        # LID head
        self.lid_head = nn.Linear(model_dim, self.cfg.engram_lid_classes)
        self.lid_sc_linear = nn.Linear(self.cfg.engram_lid_classes, model_dim)
        self.lid_scale = nn.Parameter(torch.tensor(0.1))

        # Engram modules
        # Sort engram_layers to ensure correct index mapping between
        # construction order and forward-pass insertion order.
        sorted_engram_layers = sorted(self.cfg.engram_layers)
        engram_configs = []
        for layer in sorted_engram_layers:
            if layer == self.cfg.engram_lid_layer:
                ec = EngramConfig(
                    ngram_orders=self.cfg.engram_ngram_orders,
                    num_heads=self.cfg.engram_num_heads,
                    mem_dim=self.cfg.engram_mem_dim,
                    model_dim=model_dim,
                    table_size=self.cfg.engram_table_size,
                    conv_kernel=4,
                    dropout=self.cfg.engram_dropout,
                    key_offset=self.cfg.engram_audio_bins,
                    key_range=self.cfg.engram_audio_bins + self.cfg.engram_lid_classes - 1,
                )
            else:
                ec = EngramConfig(
                    ngram_orders=self.cfg.engram_ngram_orders,
                    num_heads=self.cfg.engram_num_heads,
                    mem_dim=self.cfg.engram_mem_dim,
                    model_dim=model_dim,
                    table_size=self.cfg.engram_table_size,
                    conv_kernel=4,
                    dropout=self.cfg.engram_dropout,
                    key_offset=0,
                    key_range=self.cfg.engram_audio_bins - 1,
                )
            engram_configs.append(ec)
        self.engrams = nn.ModuleList([EngramModule(ec) for ec in engram_configs])

        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specaug_start_epoch
        self.engram_layers_set = set(self.cfg.engram_layers)
        self.lid_layer = self.cfg.engram_lid_layer
        self.audio_bins = self.cfg.engram_audio_bins
        self.frame_size = frame_size

        # Weight initialization
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if "c_proj" in name and "attn" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * gpt2_cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _derive_audio_keys(self, frames: torch.Tensor) -> torch.Tensor:
        """Derive discrete audio keys from raw audio frames (RMS energy quantization)."""
        frame_rms = torch.sqrt(torch.mean(frames.float() ** 2, dim=-1) + 1e-8)
        min_rms = frame_rms.amin(dim=1, keepdim=True)
        max_rms = frame_rms.amax(dim=1, keepdim=True)
        norm_rms = (frame_rms - min_rms) / (max_rms - min_rms + 1e-8)
        audio_keys = (norm_rms * self.audio_bins).floor().long().clamp(0, self.audio_bins - 1)
        return audio_keys

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T, 1]
        :param raw_audio_len: [B]
        :return: (log_probs [B, T_frame, V+1], audio_features_len [B])
        """
        B = raw_audio.shape[0]
        T = raw_audio.shape[1]

        squeezed = raw_audio.squeeze(-1).float()
        frame_size = self.frame_size
        T_frame = T // frame_size
        if T_frame == 0:
            T_frame = 1
            pad_len = frame_size - T
            squeezed = F.pad(squeezed, (0, pad_len))
            frames = squeezed.reshape(B, T_frame, frame_size)
        else:
            frames = squeezed[:, :T_frame * frame_size].reshape(B, T_frame, frame_size)

        audio_features_len = (raw_audio_len // frame_size).clamp(min=1, max=T_frame)

        # Safety: ensure we don't exceed max_position
        assert T_frame <= self.cfg.gpt2_config.max_position, (
            f"T_frame ({T_frame}) exceeds max_position ({gpt2_cfg.max_position}); "
            f"increase max_position in GPT2Config"
        )

        x = self.audio_proj(frames)
        positions = torch.arange(T_frame, device=x.device)
        x = x + self.position_embed(positions)
        x = self.drop(x)

        audio_keys = self._derive_audio_keys(frames)

        mask = torch.arange(T_frame, device=x.device).unsqueeze(0) < audio_features_len.to(x.device).unsqueeze(1)

        lid_keys = None
        eng_idx = 0

        for i, block in enumerate(self.blocks):
            layer_num = i + 1
            x = block(x, mask)

            if layer_num == self.lid_layer:
                lid_logits = self.lid_head(x)
                lid_probs = F.softmax(lid_logits, dim=-1)
                lid_features = self.lid_sc_linear(lid_probs)
                x = x + self.lid_scale * lid_features
                lid_keys = torch.argmax(lid_logits, dim=-1)

            if layer_num in self.engram_layers_set:
                if layer_num == self.lid_layer:
                    keys = lid_keys
                else:
                    keys = audio_keys
                x = x + self.engrams[eng_idx](x, keys)
                eng_idx += 1

        x = self.ln_f(x)
        x = self.final_dropout(x)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, audio_features_len


# ============================================================================
# RETURNN hooks
# ============================================================================

def train_step(**kwargs):
    raise NotImplementedError("gpt2_ngram_ctc_onnx is for recognition only.")


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


# ============================================================================
# Config builder
# ============================================================================

def get_model_config(vocab_size_without_blank: int, network_args: dict) -> ModelConfig:
    """Build ModelConfig with GPT-2 Ngram CTC defaults (matches training model)."""
    gpt2_cfg = GPT2Config(
        n_layer=network_args.get("n_layer", 12),
        n_head=network_args.get("n_head", 12),
        n_embd=network_args.get("n_embd", 768),
        ff_dim=network_args.get("ff_dim", 3072),
        resid_pdrop=network_args.get("resid_pdrop", 0.1),
        embd_pdrop=network_args.get("embd_pdrop", 0.1),
        attn_pdrop=network_args.get("attn_pdrop", 0.1),
        layer_norm_eps=network_args.get("layer_norm_eps", 1e-5),
        max_position=network_args.get("max_position", 4096),
    )

    sample_rate = network_args.get("sample_rate", 16000)
    frame_ms = network_args.get("frame_ms", 40)
    frame_size = int(sample_rate * frame_ms / 1000)

    return ModelConfig(
        gpt2_config=gpt2_cfg,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        frame_size=frame_size,
        label_target_size=vocab_size_without_blank,
        final_dropout=network_args.get("final_dropout", 0.1),
        engram_layers=network_args.get("engram_layers", [2, 6]),
        engram_lid_layer=network_args.get("engram_lid_layer", 6),
        engram_audio_bins=network_args.get("engram_audio_bins", 32),
        engram_lid_classes=network_args.get("engram_lid_classes", 17),
        engram_ngram_orders=network_args.get("engram_ngram_orders", [2, 3]),
        engram_num_heads=network_args.get("engram_num_heads", 8),
        engram_mem_dim=network_args.get("engram_mem_dim", 1280),
        engram_table_size=network_args.get("engram_table_size", 2**12),
        engram_dropout=network_args.get("engram_dropout", 0.0),
        specaug_start_epoch=network_args.get("specaug_start_epoch", 1),
    )
