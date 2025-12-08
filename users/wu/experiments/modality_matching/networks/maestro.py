from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_rel_pos_ctc import (
    LogMelFeatureExtractionV1,
    LogMelFeatureExtractionV1Config,
    ConformerRelPosEncoderV1,
    ConformerRelPosEncoderV1Config,
    specaugment_v1_by_length,
    lengths_to_padding_mask,
)
from .duration_model import (
    DurationModelConfig,
    DurationPredictorConv,
    tokens_and_durations_from_alignment,  # RNNT-style (no label loop)
    pad_list_2d,
)
from .best_rq_conformer import BestRQConformerModel, BestRQConformerConfig

import returnn.frontend as rf
from returnn.tensor import batch_dim

from i6_experiments.users.berger.systems.dataclasses import ConfigVariant
from i6_experiments.users.jxu.pytorch.serializers.basic import (
    get_basic_pt_network_serializer_v2,
)
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.common.setups.serialization import Import, NonhashedCode


# -------------------------
# Small utilities
# -------------------------

def remove_blanks_only(aln_1d: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Remove blanks, keep everything else (no repetition collapsing)."""
    return aln_1d[aln_1d != blank_id]

def _first_nonblank_and_trim(aln_1d: torch.Tensor, blank_id: int) -> Tuple[torch.Tensor, int]:
    """Trim leading blanks; return trimmed alignment and the first nonblank index (or len if none)."""
    nb = (aln_1d != blank_id)
    if not nb.any():
        return aln_1d.new_empty((0,), dtype=aln_1d.dtype), int(aln_1d.numel())
    first = int(torch.argmax(nb.int()))
    return aln_1d[first:], first

def pad_list_3d(lst: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad list of [Ti, D] to [B, Tmax, D] and return (padded, lengths[B])."""
    if len(lst) == 0:
        device = torch.device("cuda", 0)
        return torch.empty(0, 0, 0, device=device), torch.empty(0, dtype=torch.long, device=device)
    device = lst[0].device
    lens = torch.tensor([t.size(0) for t in lst], device=device, dtype=torch.long)
    T = int(lens.max().item())
    D = lst[0].size(1)
    out = torch.full((len(lst), T, D), pad_value, device=device, dtype=lst[0].dtype)
    for i, t in enumerate(lst):
        if t.numel() > 0:
            out[i, :t.size(0)] = t
    return out, lens

def expand_by_durations(emb: torch.Tensor, durs: torch.Tensor) -> torch.Tensor:
    """Repeat each token embedding emb[i] by durs[i] frames; returns [sum(durs), D]."""
    if emb.size(0) == 0:
        return emb.new_zeros((0, emb.size(1)))
    reps = torch.repeat_interleave(torch.arange(emb.size(0), device=emb.device), durs.clamp_min(1))
    return emb[reps]

def _generate_span_mask(T: int, p: float, span_len: int, device: torch.device) -> torch.Tensor:
    """Generates a boolean mask of shape [T] with spans of size `span_len`."""
    if p <= 0.0 or T == 0:
        return torch.zeros(T, dtype=torch.bool, device=device)
    
    # Calculate probability of a span starting at any given frame
    # Adjust p slightly up to account for overlaps, but p/span_len is a good approximation
    prob_start = p / span_len
    
    # sample start indices
    starts = (torch.rand(T, device=device) < prob_start).float()
    
    # dilation trick for efficiency
    mask = F.max_pool1d(
        starts.view(1, 1, T), 
        kernel_size=span_len, 
        stride=1, 
        padding=span_len // 2
    )
    
    return (mask.squeeze(0).squeeze(0) > 0.5)[:T]


# -------------------------
# Text encoder: 3× Conv (k=5) → PosEnc → Transformer
# -------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        return x + self.pe[:, : x.size(1), :]

class TextEncoderMaestro(nn.Module):
    """
    3 conv layers (kernel=5, channels=d_model) → sinusoidal pos enc → Transformer encoder.
    Padding is handled via key_padding_mask.
    """
    def __init__(self, num_layers: int, vocab_size: int, d_model: int, pad_id: int, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # 3× Conv1d on sequence (time) axis; keep width at d_model
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=True),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=True),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=True),
        ])
        self.conv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.conv_drop = nn.Dropout(dropout)

        self.posenc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        tokens:  [B, N]  (ints)
        lengths: [B]
        returns: [B, N, D]
        """
        tokens = tokens.to(dtype=torch.long)
        # completely empty batch/sequence: return correctly-shaped empty
        if tokens.numel() == 0:
            B = int(tokens.size(0))
            N = int(tokens.size(1))
            return tokens.new_zeros((B, N, self.embed.embedding_dim))

        x = self.embed(tokens)  # [B, N, D]
        # zero-out padding before convs to avoid bleed
        N = x.size(1)
        pad_mask = (torch.arange(N, device=x.device)[None, :] >= lengths[:, None])  # [B,N] True=pad
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        # Conv stack
        y = x.transpose(1, 2)  # [B, D, N]
        for i, conv in enumerate(self.convs):
            y = conv(y)
            y = y.transpose(1, 2)        # [B, N, D]
            y = self.conv_norms[i](y)
            y = F.gelu(y)
            y = self.conv_drop(y)
            y = y.transpose(1, 2)        # [B, D, N]
        y = y.transpose(1, 2)            # [B, N, D]

        # Positional encoding then Transformer
        y = self.posenc(y)               # [B, N, D]
        enc = self.encoder(y, src_key_padding_mask=pad_mask)  # [B, N, D]
        return enc


# -------------------------
# Refiner (2×: 8-head MHSA + 17×1 depthwise conv)
# -------------------------

class RefinerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, conv_kernel: int = 17):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.dw   = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2,
                               groups=d_model, bias=True)
        self.pw   = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act  = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]; key_padding_mask: True for PAD positions
        N = x.size(1)
        # avoid attending on padded positions in mhsa
        kpm = (torch.arange(N, device=x.device)[None, :] >= lengths[:, None])  # [B, N] True=pad
        y = self.ln1(x)
        y, _ = self.mha(y, y, y, key_padding_mask=kpm)  # [B, N, D]
        x = x + self.drop1(y)

        y = self.ln2(x).transpose(1, 2)  # [B, D, N]
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)            # [B, N, D]
        y = self.act(y)
        x = x + self.drop2(y)
        return x

class Refiner(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, conv_kernel: int = 17, layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(d_model, num_heads, dropout, conv_kernel) for _ in range(layers)])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, lengths)
        return x


# -------------------------
# Maestro config & model
# -------------------------

@dataclass
class MaestroConfig:
    # Private speech encoder also allowing best_rq loss
    specaug_args: dict
    best_rq_cfg: BestRQConformerConfig

    # Text
    text_num_layers: int  # 6 for standard MAESTRO, 3 for our initial model
    text_vocab_size: int
    text_pad_id: int
    text_d_model: int
    text_nheads: int
    text_dropout: float

    # Duration (frozen)
    duration_cfg: DurationModelConfig  # full model incl. embedding; will be frozen

    # Shared encoder (+ CTC head)
    shared_cfg: ConformerRelPosEncoderV1Config
    target_size: int  # vocab incl blank

    # Matching/refiner
    refiner_heads: int = 8
    refiner_dropout: float = 0.1
    refiner_kernel: int = 17
    refiner_layers: int = 2

    # Misc training knobs
    blank_id: int = 0
    max_dur_frames: int = 120         # cap expanded frames per token
    text_frame_drop_p: float = 0.0    # optional tiny frame drop on expanded text
    loss_scales: dict = None          # {"ctc_paired":1.0,"ctc_text":1.0,"match":1.0,"best_rq":1.0}


class MaestroModel(nn.Module):
    """
    MAESTRO:
      - Private speech encoder (frame rate) -> adapter -> Shared -> CTC
      - Text encoder (conv+Transformer) + Refiner (token-rate) + frozen duration model to expand tokens (text-only) -> adapter -> Shared -> CTC
      - Modality match (MSE) computed in train_step by refining text tokens, expanding with alignment durations, and matching at frame-rate in shared-in space.
      - Audio-only: BestRQ SSL branch tied to the same private speech encoder.
    """
    def __init__(self, step: int, cfg: MaestroConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        if self.cfg.loss_scales is None:
            self.cfg.loss_scales = {"ctc_paired": 1.0, "ctc_text": 1.0, "match": 1.0, "best_rq": 1.0}

        self.best_rq = BestRQConformerModel(step=step, cfg=cfg.best_rq_cfg, **kwargs)
        self.sp_out_dim = cfg.best_rq_cfg.conformer_cfg.block_cfg.ff_cfg.input_dim

        # Text encoder (3 conv + Transformer)
        self.text_enc = TextEncoderMaestro(
            num_layers=cfg.text_num_layers,
            vocab_size=cfg.text_vocab_size,
            d_model=cfg.text_d_model,
            pad_id=cfg.text_pad_id,
            nheads=cfg.text_nheads,
            dropout=cfg.text_dropout,
        )

        # Frozen duration model
        self.duration = DurationPredictorConv(step=step, cfg=cfg.duration_cfg)
        for p in self.duration.parameters():
            p.requires_grad = False
        self.duration.eval()

        # Adapters
        shared_in = cfg.shared_cfg.block_cfg.ff_cfg.input_dim
        self.speech_adapter = nn.Linear(self.sp_out_dim, shared_in) if self.sp_out_dim != shared_in else nn.Identity()
        self.text_adapter   = nn.Linear(cfg.text_d_model, shared_in) if cfg.text_d_model != shared_in else nn.Identity()

        # Refiner (token-rate, on text embeddings)
        self.refiner = Refiner(d_model=shared_in, num_heads=cfg.refiner_heads,
                               dropout=cfg.refiner_dropout, conv_kernel=cfg.refiner_kernel, layers=cfg.refiner_layers)

        # Shared encoder + CTC head
        self.shared = ConformerRelPosEncoderV1(cfg=cfg.shared_cfg)
        self.final  = nn.Linear(shared_in, cfg.target_size)

    # --------- low-level forward ---------
    def _best_rq_encoder_forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward the bests_rq(speech private) encoder without getting best-rq targets for supervised trainin """
        logmel, seq_mask = self.best_rq.forward_logmel(audio_features=audio, audio_features_len=audio_len)

        # Add specaugment for supervised training
        x = specaugment_v1_by_length(logmel, **self.cfg.specaug_args) if self.training else logmel

        conformer_out_list, seq_mask = self.best_rq.conformer(x, seq_mask)  # just output final layer
        # lengths = number of non-pad frames after frontend/subsampling
        in_len = seq_mask.sum(dim=1).to(torch.long)
        return conformer_out_list[0], in_len

    def shared_ctc_from_speech_features(self, sp: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp  = self.speech_adapter(sp)                                        # [B,T,Dshared]
        y, _ = self.shared(inp, lengths_to_padding_mask(T))                   # [B,T,Dshared]
        logp = F.log_softmax(self.final(y[0]), dim=-1).transpose(0, 1)           # [T,B,V]
        return logp, T

    def _upsample_and_refine_text_frames(
        self,
        tokens: torch.Tensor,        # [B,N]
        tok_lens: torch.Tensor,      # [B]
        durs: torch.Tensor,          # [B,N] (int) durations per token (from duration model or alignment)
        frame_drop_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Text → token embeddings → expand by durations (frame-rate) → project to shared_in → Refiner (frame-rate).
        Returns (refined_frames[B,Tmax,Dshared], frame_lengths[B]).
        """
        shared_in = self.cfg.shared_cfg.block_cfg.ff_cfg.input_dim
        # 1) token-rate embeddings
        text_tok = self.text_enc(tokens, tok_lens)  # [B,N,Dtext]

        # 2) expand to frames & project to shared_in
        frame_list: List[torch.Tensor] = []
        for b in range(tokens.size(0)):
            n = int(tok_lens[b].item())
            if n == 0:
                frame_list.append(text_tok.new_zeros((0, self.shared.cfg.block_cfg.ff_cfg.input_dim)))
                continue
            exp_b = expand_by_durations(text_tok[b, :n], durs[b, :n])  # [Tb, Dtext]
            exp_b = self.text_adapter(exp_b)                          # [Tb, Dshared]
            if self.training and frame_drop_p > 0.0 and exp_b.size(0) > 0:
                span_mask = _generate_span_mask(
                    exp_b.size(0), 
                    p=frame_drop_p, 
                    span_len=15,  # 600ms seems a good span 
                    device=exp_b.device
                ).unsqueeze(-1) # [T, 1]
                exp_b = exp_b.masked_fill(span_mask, 0.0)

            frame_list.append(exp_b)

        # if absolutely no samples made it through (rare but possible), return empty on the correct device
        if len(frame_list) == 0:
            x_pad = torch.empty(0, 0, shared_in, device=torch.device("cuda", 0), dtype=text_tok.dtype)
            x_lens = torch.empty(0, dtype=torch.long, device=torch.device("cuda", 0))
            return x_pad, x_lens

        x_pad, x_lens = pad_list_3d(frame_list, pad_value=0.0)       # [B,Tmax,Dshared], [B]

        # 3) frame-rate Refiner in shared_in space
        x_ref = self.refiner(x_pad, x_lens)                          # [B,Tmax,Dshared]
        return x_ref, x_lens

    def shared_ctc_from_text(self, tokens: torch.Tensor, tok_lens: torch.Tensor,
                             clamp_max: int, frame_drop_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # durations from frozen model (token-rate)
        with torch.no_grad():
            log_d = self.duration(tokens, tok_lens)           # [B,N]
            durs  = log_d.exp().round().clamp_min(1).clamp_max(clamp_max).long()

        # upsample to frame-rate and refine in shared_in space
        x_ref, in_lens = self._upsample_and_refine_text_frames(tokens, tok_lens, durs, frame_drop_p)

        # shared encoder + CTC
        y, _ = self.shared(x_ref, lengths_to_padding_mask(in_lens))  # [B,T,Dshared]
        logp = F.log_softmax(self.final(y[0]), dim=-1).transpose(0, 1)  # [T,B,V]
        return logp, in_lens

# -------------------------
# Train step
# -------------------------
def train_step(
    *,
    model: MaestroModel,
    extern_data,
    **kwargs
):
    """
    extern_data keys (mixed inside a batch; empty sequences length=0):
      - "paired_audio": B, T_a   (waveform)
      - "paired_align": B, T_aln (RNNT-style alignment with blanks; no label loop)
      - "audio":        B, T_u   (unlabeled waveform; audio-only SSL)
      - "text":         B, N     (token ids, no blanks)
    """
    device = next(model.parameters()).device
    cfg = model.cfg
    blank_id = cfg.blank_id
    pad_id   = cfg.text_pad_id

    # lengths (keep on CPU here; raw tensors are usually CPU)
    len_paired_audio = extern_data["paired_audio"].dims[1].dyn_size_ext.raw_tensor.to(device=device)
    len_paired_align = extern_data["paired_align"].dims[1].dyn_size_ext.raw_tensor.to(device=device)
    len_audio        = extern_data["audio"].dims[1].dyn_size_ext.raw_tensor.to(device=device)
    len_text         = extern_data["text"].dims[1].dyn_size_ext.raw_tensor.to(device=device)

    # indices per branch (CPU)
    idx_paired = torch.where((len_paired_audio > 0) & (len_paired_align > 0))[0].to(device=device)
    idx_text   = torch.where(len_text > 0)[0].to(device=device)
    idx_audio  = torch.where(len_audio > 0)[0].to(device=device)

    # -------------------------
    # Paired path
    # -------------------------
    if idx_paired.numel() > 0:
        # slice on CPU, then move to GPU only what you need
        audio_b = extern_data["paired_audio"].raw_tensor.index_select(0, idx_paired).to(device=device)
        T_audio = len_paired_audio.index_select(0, idx_paired).to(device=device)

        align_b = extern_data["paired_align"].raw_tensor.index_select(0, idx_paired).to(device=device, dtype=torch.long)  # CPU int
        T_align = len_paired_align.index_select(0, idx_paired).to(device=device)

        # Private speech features (frame-rate) via BestRQ encoder
        sp_frames, sp_T = model._best_rq_encoder_forward(audio_b, T_audio)  # [B,T,Dsp], [B]

        # CTC on shared from these features
        logp_sp, in_len_sp = model.shared_ctc_from_speech_features(sp_frames, sp_T)

        # Targets by removing blanks only
        target_list: List[torch.Tensor] = []
        for b in range(idx_paired.numel()):
            target_list.append(remove_blanks_only(align_b[b, :T_align[b]], blank_id=blank_id))
        target_pad, target_len = pad_list_2d(target_list, pad_value=pad_id)                 # [B,S], [B]

        loss_ctc_sp = F.ctc_loss(
            log_probs=logp_sp, targets=target_pad,
            input_lengths=in_len_sp, target_lengths=target_len,
            blank=blank_id, reduction="none", zero_infinity=True
        )
        rf.get_run_ctx().mark_as_loss(
            name="ctc_paired",
            loss=loss_ctc_sp,
            scale=cfg.loss_scales.get("ctc_paired", 1.0),
            use_normalized_loss=True,
            custom_inv_norm_factor=rf.convert_to_tensor(target_len.cpu(), dims=[batch_dim]),
        )

    tokens_list: List[torch.Tensor] = []
    durs_list:   List[torch.Tensor] = []
    first_idx_list: List[int] = []
    trim_len_list: List[int] = []

    for b in range(idx_paired.numel()):
        aln_b = align_b[b, :T_align[b]]
        aln_trim, first = _first_nonblank_and_trim(aln_b, blank_id=blank_id)
        tok_b, dur_b = tokens_and_durations_from_alignment(aln_trim, blank_id=blank_id, trim_edges=False)
        if tok_b.numel() == 0:
            tok_b = torch.tensor([pad_id], dtype=torch.long, device=device)
            dur_b = torch.tensor([1], dtype=torch.long, device=device)
        dur_b[-1] = torch.clamp(dur_b[-1], max=cfg.max_dur_frames)
        tokens_list.append(tok_b)
        durs_list.append(dur_b)
        first_idx_list.append(first)
        trim_len_list.append(int(aln_trim.numel()))

    tokens_pad, tok_lens = pad_list_2d(tokens_list, pad_value=pad_id)   # [B,N], [B]
    durs_pad, _          = pad_list_2d(durs_list,   pad_value=1)        # [B,N]
    durs_pad = durs_pad.to(device=device, dtype=torch.long)

    # Text: upsample by alignment durations and refine (frame-rate) in shared_in space (no frame drop)
    text_frames_pad, text_T = model._upsample_and_refine_text_frames(
        tokens=tokens_pad, tok_lens=tok_lens, durs=durs_pad, frame_drop_p=0.0
    )  # [B,Tmax,Dshared], [B]

    # Speech: slice the same trimmed window and project to shared_in
    sp_frame_list: List[torch.Tensor] = []
    for b in range(idx_paired.numel()):
        first = first_idx_list[b]
        tlen  = int(text_T[b].item())               # ensure same length after clamping
        sp_b = sp_frames[b, first:first + tlen]    # [Ttrim?, Dsp]
        sp_b = model.speech_adapter(sp_b)          # [Ttrim?, Dshared]
        sp_frame_list.append(sp_b)
    sp_frames_pad, _ = pad_list_3d(sp_frame_list, pad_value=0.0)        # [B,Tmax,Dshared]

    # Masked MSE at frame-rate in shared_in
    valid_frames_total = int(text_T.sum().item()) if text_T.numel() > 0 else 0
    if (tokens_pad.size(0) == 0) or (valid_frames_total == 0):
        pass
    else:
        Tmax = text_frames_pad.size(1)
        mask = (torch.arange(Tmax, device=device)[None, :] < text_T[:, None]).float()  # [B,T]
        match_target = sp_frames_pad.detach()  # no grad for audio encoder, only text encoder should imitate audio encoder update 
        match_pred   = text_frames_pad
        # add norm to avoid model collapse
        match_target_norm = F.layer_norm(match_target, match_target.shape[-1:])
        match_pred_norm   = F.layer_norm(match_pred,   match_pred.shape[-1:])

        # mean over F axis to have reasonable magnitude for loss
        mse = ((match_target_norm - match_pred_norm) ** 2).mean(dim=2) * mask  # [B,T]
        loss_match = mse.sum(dim=1)  # reduce to [B]

        rf.get_run_ctx().mark_as_loss(
            name="modality_match",
            loss=loss_match,
            scale=cfg.loss_scales.get("match", 1.0),
            use_normalized_loss=True,
            custom_inv_norm_factor=rf.convert_to_tensor(text_T.cpu(), dims=[batch_dim]),
        )

    # -------------------------
    # Text-only path: refine tokens → expand via frozen duration model → shared → CTC
    # -------------------------
    if idx_text.numel() > 0:
        text_only     = extern_data["text"].raw_tensor.index_select(0, idx_text).to(device=device, dtype=torch.long)
        text_only_len = len_text.index_select(0, idx_text).to(device=device)

        logp_text, in_len_text = model.shared_ctc_from_text(
            tokens=text_only, tok_lens=text_only_len,
            clamp_max=cfg.max_dur_frames, frame_drop_p=cfg.text_frame_drop_p
        )

        loss_ctc_text = F.ctc_loss(
            log_probs=logp_text, targets=text_only,
            input_lengths=in_len_text, target_lengths=text_only_len,
            blank=blank_id, reduction="none", zero_infinity=True
        )
        rf.get_run_ctx().mark_as_loss(
            name="ctc_text",
            loss=loss_ctc_text,
            scale=cfg.loss_scales.get("ctc_text", 1.0),
            use_normalized_loss=True,
            custom_inv_norm_factor=rf.convert_to_tensor(text_only_len.cpu(), dims=[batch_dim]),
        )

    # -------------------------
    # Audio-only SSL: BestRQ branch
    # -------------------------
    if idx_audio.numel() > 0:
        audio_only = extern_data["audio"].raw_tensor.index_select(0, idx_audio).to(device=device)
        audio_only_len = len_audio.index_select(0, idx_audio).to(device=device)

        # aux loss should not be needed, but keep the possibility
        outs, targets_list = model.best_rq(audio_features=audio_only, audio_features_len=audio_only_len)  # lists
        for i, (logits, targets) in enumerate(zip(outs, targets_list)):
            # logits and targets already flattened -> normalized here instead of in rf
            ce = F.cross_entropy(logits, targets.to(device=logits.device, dtype=torch.long), reduction="sum")
            normalized_ce = ce / targets.numel() 
            rf.get_run_ctx().mark_as_loss(
                name=f"best_rq_{i}",
                loss=normalized_ce,
                scale=cfg.loss_scales.get("best_rq", 1.0)
            )


def get_train_serializer(
    model_config: ConformerCTCConfig,
) -> Collection:
    return get_basic_pt_network_serializer_v2(
        module_import_path=f"{__name__}.{MaestroModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
        ],
    )


class MaestroCTCExport(MaestroModel):
    """
    Recognition-only subclass for ONNX export.

    Forward applies log_softmax and returns (log_probs, enc_lens).
    Unused submodules (text encoder, duration model, refiner, SSL heads) are not touched,
    so they won’t be in the exported ONNX graph.
    """
    def __init__(self, step: int, cfg: MaestroConfig, **kwargs):
        super().__init__(step=step, cfg=cfg, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        audio_features: torch.Tensor,                      # [B, T] or [B, 1, T] waveform (float32)
        audio_features_len: Optional[torch.Tensor] = None  # [B] lengths expected by logmel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Feature extraction at log-mel rate (no SpecAugment, no SSL masking)
        logmel, seq_mask = self.best_rq.forward_logmel(audio_features=audio_features, audio_features_len=audio_features_len)  # [B, Tm, F], [B,Tm]

        conformer_out_list, seq_mask = self.best_rq.conformer(logmel, seq_mask)  # just output final layer
        # lengths = number of non-pad frames after frontend/subsampling
        enc_lens = seq_mask.sum(dim=1).to(torch.long)

        # 3) Project to shared_in and run the shared encoder
        x = self.speech_adapter(conformer_out_list[0])                                        # [B, Tenc, Dshared]
        x, _ = self.shared(x, seq_mask)          # [B, Tenc, Dshared]

        # 4) Final CTC head + log-softmax
        logits = self.final(x[0])                                         # [B, Tenc, V]
        log_probs = F.log_softmax(logits, dim=-1)                         # [B, Tenc, V]
        return log_probs, enc_lens


def export(*, model: torch.nn.Module, model_filename: str):
    dummy_data = torch.randn(1, 30*160, 1, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32)*30*160

    model.best_rq.export_mode = True
    torch.onnx.export(
        model=model.eval(),
        args=(dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data:size1"],
        output_names=["log_probs"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data:size1": {0: "batch"},
            "log_probs": {0: "batch", 1: "time"},
        },
    )

def get_recog_serializer(
    model_config: MaestroConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer_v2(
        module_import_path=f"{__name__}.{MaestroCTCExport.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
        ],
    )

def get_prior_serializer(
    model_config: MaestroConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    pytorch_package = "i6_experiments.users.berger.pytorch"

    return get_basic_pt_network_serializer_v2(
        module_import_path=f"{__name__}.{MaestroCTCExport.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            NonhashedCode("import returnn.frontend as rf\n"),
            Import(f"{pytorch_package}.forward.basic.forward_step"),
            Import(f"{pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"),
        ],
    )

def get_serializer(model_config: MaestroConfig, variant: ConfigVariant) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args) -> MaestroConfig:
    """
    Build a MaestroConfig using *separate* sub-dicts inside `network_args`:
      - best_rq_args:          args for BestRQ private speech encoder (passed through to its get_default_config_v1)
      - duration_model_args:   args for the frozen duration model
      - text_encoder_args:     args for the text embedding extractor
      - shared_encoder_args:   args for the shared Conformer (depth etc.)
      - refiner_args:          args for the frame-rate refiner
      - specaug_args:          SpecAugment args for supervised audio path
      - ctc_args:              target_size / blank_id for CTC
      - loss_scales:           per-loss weights
    """
    # lazy imports to avoid top-level deps
    from .best_rq_conformer import get_default_config_v1 as get_bestrq_default_config_v1
    from .duration_model import get_default_config_v1 as get_duration_default_config_v1

    # ---- sub-arg dictionaries (with sane defaults) ----
    best_rq_args        = network_args.get("best_rq_args", {"num_outputs": 8192})  # default code book size
    duration_model_args = network_args.get("duration_model_args", {})
    text_encoder_args   = network_args.get("text_encoder_args", {})
    shared_encoder_args = network_args.get("shared_encoder_args", {})
    refiner_args        = network_args.get("refiner_args", {})
    # same as CTC baseline, maybe try weaker due to the modality matching?
    specaug_args = {"time_min_num_masks": 2,
                    "time_max_mask_per_n_frames": 25,
                    "time_mask_max_size": 20,
                    "freq_min_num_masks": 2,
                    "freq_mask_max_size": 5,
                    "freq_max_num_masks": 8} if "specaug_args" not in network_args else network_args["specaug_args"]
    ctc_args            = network_args.get("ctc_args", {})
    loss_scales         = network_args.get("loss_scales", {"ctc_paired": 1.0, "ctc_text": 1.0, "match": 1.0, "best_rq": 1.0})

    # ---------- Private speech encoder (BestRQ) ----------
    # Pass only best_rq_args to BestRQ's builder.
    best_rq_cfg = get_bestrq_default_config_v1(num_inputs, best_rq_args["num_outputs"], best_rq_args)

    # ---------- CTC / vocab ----------
    blank_id    = ctc_args.get("blank_id", 0)
    target_size = ctc_args.get("target_size", num_outputs)

    # Text vocab: allow explicit override, otherwise infer from CTC target_size / blank
    text_vocab_size = text_encoder_args.get("vocab_size", target_size)

    # ---------- Text encoder (conv + Transformer) ----------
    text_num_layers = text_encoder_args.get("num_layers", 3)  # half of 6-layer paper default
    text_d_model    = text_encoder_args.get("d_model", 512)
    text_nheads     = text_encoder_args.get("nheads", max(1, text_d_model // 64))  # e.g., 8 for 512
    text_dropout    = text_encoder_args.get("dropout", 0.1)
    text_pad_id     = text_encoder_args.get("pad_id", 0)

    # optional frame-drop on expanded text frames (used in A-MLM path)
    text_frame_drop_p = text_encoder_args.get("frame_drop_p", 0.3)

    # ---------- Duration model (frozen at runtime) ----------
    duration_cfg   = get_duration_default_config_v1(vocab_size=text_vocab_size, **duration_model_args)

    # ---------- Shared encoder ----------
    # Must use the SAME Conformer block config as BestRQ (private encoder), but allow different num_layers.
    shared_num_layers = shared_encoder_args.get(
        "num_layers",
        max(1, best_rq_cfg.conformer_cfg.num_layers * 2),  # 1:2 for small model, 1:3 for larger models
    )
    shared_cfg = ConformerRelPosEncoderV1Config(
        num_layers=shared_num_layers,
        frontend=None,  # shared runs on projected frame features; no extra frontend
        block_cfg=best_rq_cfg.conformer_cfg.block_cfg,  # reuse the exact block config
    )

    # ---------- Refiner (frame-rate, operates in shared_in space) ----------
    refiner_heads   = refiner_args.get("heads", 8)
    refiner_kernel  = refiner_args.get("kernel", 17)
    refiner_dropout = refiner_args.get("dropout", 0.1)
    refiner_layers  = refiner_args.get("layers", 2)

    # ---------- Misc knobs ----------
    max_dur_frames = network_args.get("max_dur_frames", 120)

    return MaestroConfig(
        specaug_args=specaug_args,
        best_rq_cfg=best_rq_cfg,

        text_num_layers=text_num_layers,
        text_vocab_size=text_vocab_size,
        text_pad_id=text_pad_id,
        text_d_model=text_d_model,
        text_nheads=text_nheads,
        text_dropout=text_dropout,

        duration_cfg=duration_cfg,

        shared_cfg=shared_cfg,
        target_size=target_size,

        # refiner
        refiner_heads=refiner_heads,
        refiner_dropout=refiner_dropout,
        refiner_kernel=refiner_kernel,
        refiner_layers=refiner_layers,

        # misc
        blank_id=blank_id,
        max_dur_frames=max_dur_frames,
        text_frame_drop_p=text_frame_drop_p,
        loss_scales=loss_scales,
    )

