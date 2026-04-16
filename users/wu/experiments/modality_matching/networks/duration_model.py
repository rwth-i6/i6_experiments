from dataclasses import dataclass
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import returnn.frontend as rf

from i6_experiments.users.jxu.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.common.setups.serialization import Import, NonhashedCode
from i6_models.config import ModelConfiguration

@dataclass
class DurationModelConfig(ModelConfiguration):
    vocab_size: int
    d_model: int
    num_blocks: int
    dropout: float
    token_pad_id: int


# -------------------------
# Alignment → tokens & durations
# -------------------------

def _strip_edge_blanks(aln_1d: torch.Tensor, blank_id: int) -> torch.Tensor:
    """
    Trim leading/trailing blanks from a single alignment (1D LongTensor).
    If all blank, returns empty.
    """
    nonblank = (aln_1d != blank_id)
    if not nonblank.any():
        return aln_1d.new_empty((0,), dtype=aln_1d.dtype)
    first = int(torch.argmax(nonblank.int()))
    return aln_1d[first:]


def tokens_and_durations_from_alignment(
    aln_1d: torch.Tensor,
    *,
    blank_id: int,
    trim_edges: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RNNT-like alignment (blanks + single occurrences of labels).
    Duration of token i = frames from its emission (inclusive) up to the next emission;
    the last token spans to end. Returns (tokens[N], durations[N]).
    """
    aln = _strip_edge_blanks(aln_1d, blank_id) if trim_edges else aln_1d
    if aln.numel() == 0:
        return aln.new_empty((0,), dtype=torch.long), aln.new_empty((0,), dtype=torch.long)

    emit_pos = torch.nonzero(aln != blank_id, as_tuple=False).flatten()  # [N]
    if emit_pos.numel() == 0:
        return aln.new_empty((0,), dtype=torch.long), aln.new_empty((0,), dtype=torch.long)

    tokens = aln[emit_pos]  # [N]
    durs   = torch.empty_like(tokens)

    if tokens.numel() == 1:
        durs[0] = aln.numel() - emit_pos[0]
    else:
        durs[:-1] = emit_pos[1:] - emit_pos[:-1]
        durs[-1]  = aln.numel() - emit_pos[-1]

    durs.clamp_(min=1)
    return tokens, durs


def pad_list_2d(lst: List[torch.Tensor], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of 1D tensors to [B, Nmax]. Returns (padded[B,Nmax], lengths[B]).
    """
    if len(lst) == 0:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    device = lst[0].device
    lens = torch.tensor([t.numel() for t in lst], device=device, dtype=torch.long)
    Nmax = int(lens.max().item())
    B = len(lst)
    out = torch.full((B, Nmax), pad_value, dtype=lst[0].dtype, device=device)
    for i, t in enumerate(lst):
        if t.numel() > 0:
            out[i, :t.numel()] = t
    return out, lens


# -------------------------
# MAESTRO-style lightweight conv blocks (kernel 3x1)
# -------------------------

class LConvBlock(nn.Module):
    """
    Depthwise-separable 1D conv block:
      LayerNorm → DepthwiseConv1d(k=3) → PointwiseConv1d(1x1) → GELU → Dropout → Residual
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw   = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True)
        self.pw   = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        y = self.norm(x)
        y = y.transpose(1, 2)          # [B, D, N]
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)          # [B, N, D]
        y = self.act(y)
        y = self.drop(y)
        return x + y                    # residual


class DurationPredictorConv(nn.Module):
    """
    MAESTRO-like duration model: 4 × 3x1 lightweight conv blocks → Linear → log-duration.
    If you already have a text encoder, you can bypass the embedding by calling `forward_embeddings`.
    """
    def __init__(
        self,
        step: int,
        cfg: DurationModelConfig,
        **kwargs
    ):
        super().__init__()
        self.token_pad_id = cfg.token_pad_id
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.token_pad_id)

        self.blocks = nn.ModuleList([LConvBlock(cfg.d_model, dropout=cfg.dropout) for _ in range(cfg.num_blocks)])
        self.proj   = nn.Linear(cfg.d_model, 1)

    def forward_embeddings(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:        [B, N, D] text embeddings (from your text encoder)
        lengths:  [B]       number of valid tokens per sample
        returns:  [B, N]    log-duration predictions
        """
        # Zero-out padded positions to prevent conv bleed-through from random pad vectors.
        N = x.size(1)
        pad_mask = (torch.arange(N, device=lengths.device)[None, :] >= lengths[:, None]).unsqueeze(-1)  # [B,N,1]
        x = x.masked_fill(pad_mask, 0.0)

        y = x
        for blk in self.blocks:
            y = blk(y)
        y = self.proj(y).squeeze(-1)  # [B, N]
        return y

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        tokens:   [B, N] Long (phoneme ids)
        lengths:  [B]
        """
        x = self.embed(tokens)  # [B, N, D]
        return self.forward_embeddings(x, lengths)


def train_step(
    *,
    model: DurationPredictorConv,
    extern_data,
    blank_id: int = 0,
    token_pad_id: int = 0,     # token pad id inside the model
    trim_edge_blanks: bool = True,
    jitter_frames: int = 0,    # e.g., 0 or 1
    min_duration: int = 1,
    log_eps: float = 1e-6,
    **kwargs
):
    """
    Uses only the RNNT-style alignment to derive (tokens, durations) and optimize log-duration MSE.
    """
    device = next(model.parameters()).device
    alignments: torch.Tensor = extern_data["data"].raw_tensor.to(device=device, dtype=torch.long)   # [B, T]
    aln_lens = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=device)

    # Build per-utt tokens & durations directly from the alignment
    tokens_list: List[torch.Tensor] = []
    durs_list:   List[torch.Tensor] = []
    B, _ = alignments.shape
    for b in range(B):
        aln_b = alignments[b, :aln_lens[b]]
        tok_b, dur_b = tokens_and_durations_from_alignment(
            aln_b, blank_id=blank_id, trim_edges=trim_edge_blanks
        )
        if tok_b.numel() == 0:  # degenerate: keep one padded token with min duration
            tok_b = torch.tensor([token_pad_id], dtype=torch.long, device=device)
            dur_b = torch.tensor([min_duration], dtype=torch.long, device=device)
        else:
            dur_b[-1] = torch.clamp(dur_b[-1], max=round(1.5 * 25))  # 1.5s would be too long
        tokens_list.append(tok_b)
        durs_list.append(dur_b)

    tokens_pad, tok_lens = pad_list_2d(tokens_list, pad_value=token_pad_id)   # [B, Nmax], [B]
    durs_pad,   _        = pad_list_2d(durs_list,   pad_value=min_duration)   # [B, Nmax]
    tokens_pad = tokens_pad.to(device)
    durs_pad   = durs_pad.to(device)
    tok_lens   = tok_lens.to(device)

    # Optional tiny jitter on GT durations (only valid tokens)
    if jitter_frames > 0:
        jitter = torch.randint(-jitter_frames, jitter_frames + 1, size=durs_pad.shape, device=device)
        valid = (torch.arange(tokens_pad.size(1), device=device)[None, :] < tok_lens[:, None])
        durs_pad = torch.clamp(durs_pad + (jitter * valid), min=min_duration)

    # Targets: log-durations
    log_d_tgt = (durs_pad.to(torch.float32) + log_eps).log()  # [B, Nmax]

    # Forward + masked MSE
    log_d_pred = model(tokens_pad, tok_lens)                  # [B, Nmax]
    Nmax = tokens_pad.size(1)
    mask = (torch.arange(Nmax, device=device)[None, :] < tok_lens[:, None])   # [B, Nmax]
    mse  = ((log_d_pred - log_d_tgt) ** 2) * mask.float()
    loss = mse.sum() / mask.float().sum().clamp_min(1.0)

    # mse on log-duration
    rf.get_run_ctx().mark_as_loss(name="log_dur_mse", loss=loss)

    pred_cont  = log_d_pred.exp().clamp_min(1.0)          # [B,N], continuous
    pred_round = pred_cont.round()                         # integer frames
    ae_round   = (pred_round - durs_pad.float()).abs()     # [B,N]
    mask_f     = mask.float()
    mae_frames_round = (ae_round * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    # mae on duration
    rf.get_run_ctx().mark_as_loss(name="round_dur_mae", loss=mae_frames_round, as_error=True)


def get_default_config_v1(vocab_size: int, d_model=512, num_blocks=4, dropout=0.1, token_pad_id=0) -> DurationModelConfig:
    return DurationModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_blocks=num_blocks,
        dropout=dropout,
        token_pad_id=token_pad_id,
    )


def get_serializer(
    model_config: DurationModelConfig,
):
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{DurationPredictorConv.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
        ],
    ) 
