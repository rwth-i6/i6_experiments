"""Mamba-2 (SSD) layer for ChunkedConformerEncoderV2.

Implements the Mamba-2 selective state-space block via the State Space Duality
(SSD) chunked algorithm: the scan reduces to block-diagonal matmuls inside
each chunk + a small cross-chunk state recurrence. Pure PyTorch, no custom
CUDA / Triton kernels needed.

Pure SSD scan (~80 LoC) ported from the official
``mamba_ssm.modules.ssd_minimal.ssd_minimal_discrete``
(https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py).

The encoder layer wraps the Mamba-2 block exactly like the previous Mamba-1
variant: pre-LayerNorm + un-chunk -> SSD scan over full time -> re-chunk via
``rf.window`` + residual. Drop-in for ``ChunkedConformerEncoderV2.
encoder_layer_per_layer``.

Streaming-mode state passing is a TODO; offline training and offline recog
work today (each forward initializes the SSM state to zeros).

Reference: Dao & Gu, "Transformers are SSMs" (https://arxiv.org/abs/2405.21060).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from returnn.tensor import Dim, Tensor
import returnn.frontend as rf

from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import _BatchChunkingSettings


# ---------------------------------------------------------------------------
# SSD chunked scan (PyTorch)
# ---------------------------------------------------------------------------


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable computation of cumulative segment sums. From Mamba-2 paper Appendix B."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def _ssd_chunked(
    X: torch.Tensor,  # [B, L, H, P]   value
    A: torch.Tensor,  # [B, L, H]      discretized A (a_t * dt_t, signed)
    B: torch.Tensor,  # [B, L, H, N]   state-input (per-head shared groups assumed)
    C: torch.Tensor,  # [B, L, H, N]   output-state
    block_len: int,
    initial_states: Optional[torch.Tensor] = None,
    super_batch_chunk: Optional[int] = None,
    return_final_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SSD chunked algorithm. Returns Y of shape [B, L, H, P].

    L must be a multiple of block_len.

    ``super_batch_chunk``: if set, process the super-batch dim ``B`` in slices of this size,
    each slice wrapped in :func:`torch.utils.checkpoint.checkpoint`
    so the per-einsum intermediates (``L_mat``, ``states``, ``Y_diag``, ``Y_off``)
    don't have to be kept alive for the backward pass --
    they're recomputed once per slice when backward visits it.
    Peak GPU memory then scales with one slice instead of the full super-batch.
    Trade-off: ~2x compute for the SSD scan portion (the rest of the layer is unchanged),
    but the SSD is a small fraction of total step time at typical hyperparameters,
    so end-to-end throughput hit is much smaller than 2x.
    Useful for the bidir Mamba-2 path,
    where the bwd direction merges ``(batch, num_chunks_per_seq)`` into a super-batch
    and the SSD intermediates would otherwise eat 1-2 GB each
    -- and crucially, retain them all simultaneously for the backward pass.
    """
    if super_batch_chunk is not None and X.size(0) > super_batch_chunk:
        assert not return_final_states, "return_final_states not supported with super_batch_chunk"
        outs = []
        sb = X.size(0)
        # Only use checkpoint when we're inside an autograd graph; otherwise
        # (eval / inference) plain recursion is enough and skips the checkpoint overhead.
        use_ckpt = torch.is_grad_enabled() and any(t.requires_grad for t in (X, A, B, C))
        for s in range(0, sb, super_batch_chunk):
            e = min(s + super_batch_chunk, sb)
            init = None if initial_states is None else initial_states[s:e]

            def _run(X_sl, A_sl, B_sl, C_sl, init=init):
                return _ssd_chunked(
                    X_sl, A_sl, B_sl, C_sl,
                    block_len=block_len,
                    initial_states=init,
                    super_batch_chunk=None,  # recursion guard
                )

            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    _run, X[s:e], A[s:e], B[s:e], C[s:e], use_reentrant=False
                )
            else:
                out = _run(X[s:e], A[s:e], B[s:e], C[s:e])
            outs.append(out)
        return torch.cat(outs, dim=0)
    L = X.size(1)
    assert L % block_len == 0, f"L={L} not divisible by block_len={block_len}"

    # Reshape to [B, n_chunks, block_len, ...].
    X = rearrange(X, "b (c l) h p -> b c l h p", l=block_len)
    A = rearrange(A, "b (c l) h -> b c l h", l=block_len)
    B = rearrange(B, "b (c l) h n -> b c l h n", l=block_len)
    C = rearrange(C, "b (c l) h n -> b c l h n", l=block_len)

    # A reshape for segsum: [B, H, C, L].
    A_perm = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A_perm, dim=-1)

    # 1) Diagonal blocks: within-chunk parallel attention-like matmul.
    L_mat = torch.exp(_segsum(A_perm))  # [B, H, C, L, L]
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L_mat, X)

    # 2) Inter-chunk: scan over chunk-end states.
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # [B, H, C, L]
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)  # [B, C, H, P, N]

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)  # [B, C+1, H, P, N]
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[..., -1], (1, 0))))  # [B, H, C+1, C+1]
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]  # [B, C, H, P, N]

    # 3) Output: states-to-output via remaining decay.
    state_decay_out = torch.exp(A_cumsum)  # [B, H, C, L]
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    if return_final_states:
        # new_states[:, 1:] = per-block END states (state after each block).
        return Y, new_states[:, 1:]
    return Y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_BTF(t: Tensor, spatial_dim: Dim, feat_dim: Dim) -> torch.Tensor:
    """
    Permute the raw torch tensor of t to ``(batch, spatial, feat)`` order.

    The batch dim is identified as the unique non-spatial, non-feature dim;
    if t has multiple batch-like dims, merge them with ``rf.merge_dims`` first.
    """
    batch_dims = t.remaining_dims((spatial_dim, feat_dim))
    assert len(batch_dims) == 1, f"_to_BTF expected exactly one batch-like dim in {t}, got {batch_dims}"
    return t.copy_compatible_to_dims_raw([batch_dims[0], spatial_dim, feat_dim]).contiguous()


def _from_BTF(raw: torch.Tensor, like: Tensor, spatial_dim: Dim, feat_dim: Dim, *, name: str) -> Tensor:
    """
    Wrap a raw torch tensor ``[B, T, F]`` as an rf.Tensor,
    re-using the batch dim from ``like``.

    ``feat_dim`` is the *output* feature dim (what the wrapped tensor will carry).
    ``like`` is only consulted to find the batch dim,
    so it may have any feature dim (typically the block's *input* feature dim,
    e.g. ``d_in_proj`` while ``feat_dim`` is the SSM block's output ``d_inner``).
    """
    batch_dims = like.remaining_dims((spatial_dim, like.feature_dim))
    assert len(batch_dims) == 1, f"_from_BTF expected exactly one batch-like dim in {like}, got {batch_dims}"
    return Tensor(
        name,
        dims=[batch_dims[0], spatial_dim, feat_dim],
        dtype=str(raw.dtype).split(".")[-1].replace("torch.", ""),
        raw_tensor=raw,
        feature_dim=feat_dim,
    )


def _pad_to_block_len(x: torch.Tensor, block_len: int, *, time_dim: int = 1) -> tuple[torch.Tensor, int]:
    """Right-pad with zeros along ``time_dim`` so its size is a multiple of block_len.

    Returns (padded, original_length).
    """
    T = x.size(time_dim)
    pad = (-T) % block_len
    if pad == 0:
        return x, T
    pad_shape = list(x.shape)
    pad_shape[time_dim] = pad
    zeros = x.new_zeros(pad_shape)
    return torch.cat([x, zeros], dim=time_dim), T


# ---------------------------------------------------------------------------
# Mamba-2 block
# ---------------------------------------------------------------------------


class Mamba2Block(rf.Module):
    """Mamba-2 SSD block.

    Linear in_proj -> split into (z, xBC, dt) -> causal depthwise 1d conv on
    xBC -> split (x, B, C) -> reshape x as (heads, head_dim) and B/C as
    (groups, d_state) -> dt = softplus(dt + dt_bias), A = -exp(A_log) per
    head -> A_disc = A * dt -> SSD scan -> y * silu(z) -> Linear out_proj.

    No norm or residual; wrap externally.
    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        block_len: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        """
        :param in_dim: input/output model dim.
        :param d_state: SSM state size per head. Mamba-2 uses 64-128 (paper)
            vs Mamba-1's 16 -- larger state is the key win of SSD.
        :param d_conv: depthwise 1d-conv kernel size.
        :param expand: inner expansion factor (d_inner = in_dim * expand).
        :param head_dim: width of each SSM head. Determines n_heads = d_inner / head_dim.
        :param n_groups: number of B/C groups (1 = shared across heads).
        :param block_len: SSD chunk length for the parallel scan. 64 is the
            common default; larger = less inter-chunk overhead, smaller =
            less memory.
        :param dt_min, dt_max, dt_init_floor: dt initialization range.
        """
        super().__init__()
        self.in_dim = in_dim
        d_inner = in_dim.dimension * expand
        assert d_inner % head_dim == 0, f"d_inner={d_inner} not multiple of head_dim={head_dim}"
        n_heads = d_inner // head_dim
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.n_groups = n_groups
        self.d_conv = d_conv
        self.block_len = block_len

        # Out dims used by RF wrappers.
        self.d_inner_dim = Dim(d_inner, name="m2-inner")
        # in_proj output: z (d_inner) + xBC (d_inner + 2 * n_groups * d_state) + dt (n_heads).
        d_in_proj = d_inner + d_inner + 2 * n_groups * d_state + n_heads
        self.d_in_proj_dim = Dim(d_in_proj, name="m2-in-proj")
        d_xBC = d_inner + 2 * n_groups * d_state
        self.d_xBC_dim = Dim(d_xBC, name="m2-xBC")

        self.in_proj = rf.Linear(in_dim, self.d_in_proj_dim, with_bias=False)

        # Depthwise causal 1d conv on the xBC part. Width = d_inner + 2 * groups * d_state.
        self.conv1d = rf.Conv1d(
            in_dim=self.d_xBC_dim,
            out_dim=self.d_xBC_dim,
            filter_size=d_conv,
            groups=d_xBC,
            padding="valid",
            with_bias=True,
        )

        # Per-head learned A_log and dt bias.
        self.n_heads_dim = Dim(n_heads, name="m2-heads")
        self.A_log = rf.Parameter([self.n_heads_dim])
        self.dt_bias = rf.Parameter([self.n_heads_dim])
        # D: skip-connection per head (broadcast over head_dim).
        self.D = rf.Parameter([self.n_heads_dim])

        # Output norm (RMSNorm in official; LayerNorm here for simplicity) + projection.
        self.norm_pre_out = rf.LayerNorm(self.d_inner_dim)
        self.out_proj = rf.Linear(self.d_inner_dim, in_dim, with_bias=False)

        self._init_params(dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor)

    def _init_params(self, *, dt_min: float, dt_max: float, dt_init_floor: float):
        with torch.no_grad():
            # A_log: per-head, init so A = -exp(A_log) covers [-1, -8] roughly.
            n_heads = self.n_heads
            A_init = torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32))
            self.A_log.raw_tensor.copy_(A_init)
            # D = 1.
            self.D.raw_tensor.fill_(1.0)
            # dt bias so softplus(bias) ~ log-uniform[dt_min, dt_max].
            dt = torch.exp(torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp_min_(
                dt_init_floor
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.raw_tensor.copy_(inv_dt)

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        """x: [B, T, in_dim] -> y: [B, T, in_dim]."""
        # 1) in_proj -> split into (z, xBC, dt).
        proj = self.in_proj(x)
        proj_t = _to_BTF(proj, spatial_dim, self.d_in_proj_dim)  # [B, T, d_in_proj]
        b, T, _ = proj_t.shape

        d_inner = self.d_inner
        n_groups = self.n_groups
        d_state = self.d_state
        n_heads = self.n_heads
        head_dim = self.head_dim
        d_xBC = d_inner + 2 * n_groups * d_state

        z_t = proj_t[:, :, :d_inner]
        xBC_t = proj_t[:, :, d_inner : d_inner + d_xBC]
        dt_t = proj_t[:, :, d_inner + d_xBC :]  # [B, T, n_heads]

        # 2) Causal depthwise 1d conv on xBC. Left-pad by (d_conv - 1).
        xBC_padded = F.pad(xBC_t.transpose(1, 2), (self.d_conv - 1, 0))  # [B, d_xBC, T+pad]
        # Apply conv via torch directly (depthwise). Reuses self.conv1d weights through RF for hashing/serialization.
        conv_w = self.conv1d.filter.raw_tensor  # [d_xBC, 1, d_conv] (groups = d_xBC)
        conv_b = self.conv1d.bias.raw_tensor if self.conv1d.with_bias else None
        xBC_conv = F.conv1d(xBC_padded, conv_w, bias=conv_b, groups=d_xBC)  # [B, d_xBC, T]
        xBC_conv = F.silu(xBC_conv).transpose(1, 2)  # [B, T, d_xBC]

        # 3) Split xBC into (x, B_state, C_state) and reshape to head-dim form.
        x_in = xBC_conv[:, :, :d_inner].reshape(b, T, n_heads, head_dim)
        B_state = xBC_conv[:, :, d_inner : d_inner + n_groups * d_state].reshape(b, T, n_groups, d_state)
        C_state = xBC_conv[:, :, d_inner + n_groups * d_state :].reshape(b, T, n_groups, d_state)
        # If n_groups < n_heads, broadcast groups to heads (Mamba-2 standard).
        if n_groups != n_heads:
            B_state = B_state.repeat_interleave(n_heads // n_groups, dim=2)
            C_state = C_state.repeat_interleave(n_heads // n_groups, dim=2)

        # 4) dt -> softplus(+bias); A_disc = dt * A per head.
        dt = F.softplus(dt_t + self.dt_bias.raw_tensor)  # [B, T, n_heads]
        A = -torch.exp(self.A_log.raw_tensor)  # [n_heads]
        A_disc = dt * A  # [B, T, n_heads]

        # 5) SSD scan. Pad T to a multiple of block_len.
        x_pad, T_orig = _pad_to_block_len(x_in, self.block_len, time_dim=1)
        A_pad, _ = _pad_to_block_len(A_disc, self.block_len, time_dim=1)
        B_pad, _ = _pad_to_block_len(B_state, self.block_len, time_dim=1)
        C_pad, _ = _pad_to_block_len(C_state, self.block_len, time_dim=1)
        y_pad = _ssd_chunked(x_pad, A_pad, B_pad, C_pad, block_len=self.block_len)
        y_t = y_pad[:, :T_orig]  # [B, T, n_heads, head_dim]

        # 6) + D * x (per-head skip).
        D_view = self.D.raw_tensor.view(1, 1, n_heads, 1)
        y_t = y_t + x_in * D_view  # [B, T, n_heads, head_dim]
        y_t = y_t.reshape(b, T, d_inner)  # [B, T, d_inner]

        # 7) Norm + gate(z) + out_proj.
        # silu(z) gate.
        y_t = y_t * F.silu(z_t)
        y = _from_BTF(y_t, proj, spatial_dim, self.d_inner_dim, name="m2_y")
        y = self.norm_pre_out(y)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Encoder layer wrapper
# ---------------------------------------------------------------------------


class Mamba2ChunkedLayerV2(rf.Module):
    """Pure-Mamba-2 chunked-encoder layer (LayerNorm + Mamba2Block + residual).

    Runs the Mamba-2 SSD scan over the un-chunked time axis (state carries
    across all frames). Drop-in for ``ChunkedConformerEncoderV2.
    encoder_layer_per_layer``.
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        block_len: int = 64,
        dropout: float = 0.0,
        # Optional Conformer-flavored wrappers (additive; defaults preserve current behavior).
        # No ``with_local_conv`` because Mamba-2 already has an internal depthwise causal conv.
        with_macaron_ff: bool = False,
        macaron_ff_dim_factor: int = 4,
        with_post_ln: bool = False,
        version: int = 1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        assert version == 2, f"Mamba2ChunkedLayerV2: only version=2 (lookahead-correct) supported, got {version}"
        self.norm = rf.LayerNorm(out_dim)
        self.mamba = Mamba2Block(
            out_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            n_groups=n_groups,
            block_len=block_len,
        )
        # Optional Conformer-flavored wrappers.
        # Always declare attributes (default ``None``); see README "Optional submodules".
        self.ffn1_layer_norm = None
        self.ffn1 = None
        self.ffn2_layer_norm = None
        self.ffn2 = None
        self.final_layer_norm = None
        if with_macaron_ff:
            from returnn.frontend.encoder.conformer import ConformerPositionwiseFeedForward

            ff_dim = Dim(macaron_ff_dim_factor * out_dim.dimension, name="mamba-ff")
            self.ffn1_layer_norm = rf.LayerNorm(out_dim)
            self.ffn1 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
            self.ffn2_layer_norm = rf.LayerNorm(out_dim)
            self.ffn2 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
        if with_post_ln:
            self.final_layer_norm = rf.LayerNorm(out_dim)

    def _mamba_block(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        chunking: Optional[_BatchChunkingSettings],
    ) -> Tensor:
        """Run the bare Mamba-2 block (chunked-aware), returning the per-position output.

        Extracted from the original ``__call__`` so the optional Conformer wrappers
        can call into it cleanly.
        """
        x_norm = self.norm(x)
        if self.dropout:
            x_norm = rf.dropout(x_norm, self.dropout, axis=self.out_dim)
        if chunking is None:
            return self.mamba(x_norm, spatial_dim=spatial_dim)
        # Lookahead-correct chunked path: unbounded-past centers (state carries across chunks) +
        # strict-R per-chunk lookahead via state-seeded continuation. The old path reconstructed the
        # lookahead by re-windowing the global center scan, which leaked/compounded the lookahead
        # across layers -- removed. See chunked_recurrent_lookahead.mamba2_chunked_forward.
        from .chunked_recurrent_lookahead import mamba2_chunked_forward

        return mamba2_chunked_forward(
            self.mamba,
            x_norm,
            spatial_dim=spatial_dim,
            center_dim=chunking.end_chunk_size_dim,
            chunked_time_dim=chunking.chunked_time_dim,
        )

    def __call__(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        chunking: Optional[_BatchChunkingSettings] = None,
    ) -> Tensor:
        out = x
        # 1/2 FFN before (macaron half-step).
        if self.ffn1 is not None:
            ffn1_in = self.ffn1_layer_norm(out)
            ffn1_out = self.ffn1(ffn1_in)
            out = out + 0.5 * rf.dropout(ffn1_out, self.dropout, axis=self.out_dim)
        # Mamba-2 block.
        out = out + self._mamba_block(out, spatial_dim=spatial_dim, chunking=chunking)
        # 1/2 FFN after (macaron half-step).
        if self.ffn2 is not None:
            ffn2_in = self.ffn2_layer_norm(out)
            ffn2_out = self.ffn2(ffn2_in)
            out = out + 0.5 * rf.dropout(ffn2_out, self.dropout, axis=self.out_dim)
        # Post-LN.
        if self.final_layer_norm is not None:
            out = self.final_layer_norm(out)
        return out


class BidirMamba2ChunkedLayer(rf.Module):
    """
    Bidirectional Mamba-2 chunked-encoder layer (ConMamba-style hybrid sharing).

    Shared across directions:
    input projection (``in_proj``),
    output norm (``norm_pre_out``),
    output projection (``out_proj``).

    Per direction:
    depthwise causal conv (``conv1d_fwd`` / ``conv1d_bwd``),
    SSM state parameters ``A_log`` / ``D`` / ``dt_bias``.

    Forward direction:
    chunked flat-merge causal scan (cross-chunk continuity across center frames),
    matching the existing causal layer.

    Backward direction:
    per-chunk reverse scan over ``[center + lookahead]``,
    state resets per chunk.

    Outputs averaged at center positions;
    lookahead positions keep the forward output
    (which carries the next chunk's first ``lookahead`` center frames
    via the rf.window overlap pattern).
    The ``silu(z)`` gate (with ``z`` from the shared in_proj output) is applied
    after combining the two directions,
    then the shared norm + out_proj close the block.

    See ConMamba / Speech Slytherin (Jiang & Kim, Interspeech 2024,
    https://arxiv.org/abs/2407.09732)
    for the bi-Mamba layout this layer is modeled after.
    The chunked-aware fwd/bwd time-topology split is our adaptation
    for streaming-friendly chunked encoders.

    Optional Conformer-flavored wrappers (macaron FFN, post-LN) as flags.
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        block_len: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        with_macaron_ff: bool = False,
        macaron_ff_dim_factor: int = 4,
        with_post_ln: bool = False,
        ssd_super_batch_chunk: Optional[int] = None,
        version: int = 1,
    ):
        super().__init__()
        # ``ssd_super_batch_chunk``: if set, the bwd-direction SSD scan processes its
        # super-batch (``batch * num_chunks_per_seq``) in slices of this size,
        # trading a small amount of throughput for a big drop in peak GPU memory.
        # See :func:`_ssd_chunked`'s ``super_batch_chunk`` docstring;
        # used by the bidir Mamba-2 path where the bwd direction's per-chunk reset
        # would otherwise eat 1-2 GB per einsum intermediate.
        self.ssd_super_batch_chunk = ssd_super_batch_chunk
        self.out_dim = out_dim
        d_inner = out_dim.dimension * expand
        assert d_inner % head_dim == 0, f"d_inner={d_inner} not multiple of head_dim={head_dim}"
        n_heads = d_inner // head_dim
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.n_groups = n_groups
        self.d_conv = d_conv
        self.block_len = block_len
        self.dropout = dropout
        assert version == 2, f"BidirMamba2ChunkedLayer: only version=2 (lookahead-correct) supported, got {version}"

        # Dims used by the projections + per-head params.
        self.d_inner_dim = Dim(d_inner, name="m2-bidir-inner")
        d_in_proj = d_inner + d_inner + 2 * n_groups * d_state + n_heads
        self.d_in_proj_dim = Dim(d_in_proj, name="m2-bidir-in-proj")
        d_xBC = d_inner + 2 * n_groups * d_state
        self.d_xBC_dim = Dim(d_xBC, name="m2-bidir-xBC")
        self.d_xBC = d_xBC
        self.n_heads_dim = Dim(n_heads, name="m2-bidir-heads")

        # Pre-LN (shared) -- wraps the whole bidir block (Conformer pre-norm style).
        self.norm = rf.LayerNorm(out_dim)
        # Shared input projection.
        self.in_proj = rf.Linear(out_dim, self.d_in_proj_dim, with_bias=False)
        # Per-direction depthwise causal 1d conv on the xBC stream.
        self.conv1d_fwd = rf.Conv1d(
            in_dim=self.d_xBC_dim,
            out_dim=self.d_xBC_dim,
            filter_size=d_conv,
            groups=d_xBC,
            padding="valid",
            with_bias=True,
        )
        self.conv1d_bwd = rf.Conv1d(
            in_dim=self.d_xBC_dim,
            out_dim=self.d_xBC_dim,
            filter_size=d_conv,
            groups=d_xBC,
            padding="valid",
            with_bias=True,
        )
        # Per-direction SSM state params.
        self.A_log_fwd = rf.Parameter([self.n_heads_dim])
        self.A_log_bwd = rf.Parameter([self.n_heads_dim])
        self.dt_bias_fwd = rf.Parameter([self.n_heads_dim])
        self.dt_bias_bwd = rf.Parameter([self.n_heads_dim])
        self.D_fwd = rf.Parameter([self.n_heads_dim])
        self.D_bwd = rf.Parameter([self.n_heads_dim])
        # Shared output norm + projection.
        self.norm_pre_out = rf.LayerNorm(self.d_inner_dim)
        self.out_proj = rf.Linear(self.d_inner_dim, out_dim, with_bias=False)
        # Init SSM params (per-direction independent random init for dt_bias).
        self._init_ssm_params(dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor)

        # Optional Conformer wrappers (declare as None up front; assign on demand).
        self.ffn1_layer_norm = None
        self.ffn1 = None
        self.ffn2_layer_norm = None
        self.ffn2 = None
        self.final_layer_norm = None
        if with_macaron_ff:
            from returnn.frontend.encoder.conformer import ConformerPositionwiseFeedForward

            ff_dim = Dim(macaron_ff_dim_factor * out_dim.dimension, name="bidir-mamba-ff")
            self.ffn1_layer_norm = rf.LayerNorm(out_dim)
            self.ffn1 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
            self.ffn2_layer_norm = rf.LayerNorm(out_dim)
            self.ffn2 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
        if with_post_ln:
            self.final_layer_norm = rf.LayerNorm(out_dim)

    def _init_ssm_params(self, *, dt_min: float, dt_max: float, dt_init_floor: float):
        # Legacy ``raw_tensor.copy_`` / ``fill_`` init pattern, mirroring :class:`Mamba2Block`.
        # TODO: modernize to ``.initial = value`` per README "RF Parameter init: use .initial = value".
        with torch.no_grad():
            n_heads = self.n_heads
            A_init = torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32))
            for A_log in (self.A_log_fwd, self.A_log_bwd):
                A_log.raw_tensor.copy_(A_init)
            for D in (self.D_fwd, self.D_bwd):
                D.raw_tensor.fill_(1.0)
            for dt_bias in (self.dt_bias_fwd, self.dt_bias_bwd):
                dt = torch.exp(
                    torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
                ).clamp_min_(dt_init_floor)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                dt_bias.raw_tensor.copy_(inv_dt)

    def _ssd_one_direction(
        self,
        xBC: Tensor,
        dt: Tensor,
        *,
        time_dim: Dim,
        conv1d: rf.Conv1d,
        A_log: rf.Parameter,
        dt_bias: rf.Parameter,
        D: rf.Parameter,
    ) -> Tensor:
        """
        Run conv1d + SSD scan + D-skip for one direction.

        Inputs and output are rf.Tensors.
        Raw-torch is used only inside this method,
        between the rf->torch boundary at the input
        and the torch->rf boundary at the output.
        The output has feature dim ``self.d_inner_dim``
        and time axis ``time_dim``,
        BEFORE the silu(z) gate and the shared norm + out_proj.
        """
        d_inner = self.d_inner
        n_groups = self.n_groups
        d_state = self.d_state
        n_heads = self.n_heads
        head_dim = self.head_dim
        d_xBC = self.d_xBC
        xBC_t = _to_BTF(xBC, time_dim, self.d_xBC_dim)
        dt_t = _to_BTF(dt, time_dim, self.n_heads_dim)
        b, T, _ = xBC_t.shape

        # Causal depthwise 1d conv.
        conv_w = conv1d.filter.raw_tensor
        conv_b = conv1d.bias.raw_tensor if conv1d.with_bias else None
        xBC_padded = F.pad(xBC_t.transpose(1, 2), (self.d_conv - 1, 0))
        xBC_conv = F.conv1d(xBC_padded, conv_w, bias=conv_b, groups=d_xBC)
        xBC_conv = F.silu(xBC_conv).transpose(1, 2)

        # Split into x, B, C.
        x_in = xBC_conv[:, :, :d_inner].reshape(b, T, n_heads, head_dim)
        B_state = xBC_conv[:, :, d_inner : d_inner + n_groups * d_state].reshape(b, T, n_groups, d_state)
        C_state = xBC_conv[:, :, d_inner + n_groups * d_state :].reshape(b, T, n_groups, d_state)
        if n_groups != n_heads:
            B_state = B_state.repeat_interleave(n_heads // n_groups, dim=2)
            C_state = C_state.repeat_interleave(n_heads // n_groups, dim=2)

        # dt + bias / A discretize.
        dt = F.softplus(dt_t + dt_bias.raw_tensor)
        A = -torch.exp(A_log.raw_tensor)
        A_disc = dt * A

        # SSD scan + D skip.
        x_pad, T_orig = _pad_to_block_len(x_in, self.block_len, time_dim=1)
        A_pad, _ = _pad_to_block_len(A_disc, self.block_len, time_dim=1)
        B_pad, _ = _pad_to_block_len(B_state, self.block_len, time_dim=1)
        C_pad, _ = _pad_to_block_len(C_state, self.block_len, time_dim=1)
        y_pad = _ssd_chunked(
            x_pad, A_pad, B_pad, C_pad,
            block_len=self.block_len,
            super_batch_chunk=self.ssd_super_batch_chunk,
        )
        y_t = y_pad[:, :T_orig]
        D_view = D.raw_tensor.view(1, 1, n_heads, 1)
        y_t = y_t + x_in * D_view
        y_t = y_t.reshape(b, T, d_inner)
        return _from_BTF(y_t, xBC, time_dim, self.d_inner_dim, name="m2_bidir_dir_y")

    def __call__(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        chunking: Optional[_BatchChunkingSettings] = None,
    ) -> Tensor:
        out = x
        # 1/2 FFN before (macaron half-step).
        if self.ffn1 is not None:
            ffn1_in = self.ffn1_layer_norm(out)
            ffn1_out = self.ffn1(ffn1_in)
            out = out + 0.5 * rf.dropout(ffn1_out, self.dropout, axis=self.out_dim)

        # Pre-LN + shared in_proj (Conformer pre-norm).
        x_ln = self.norm(out)
        if self.dropout:
            x_ln = rf.dropout(x_ln, self.dropout, axis=self.out_dim)
        proj = self.in_proj(x_ln)

        # Split proj into z / xBC / dt at the rf-tensor level.
        # All three keep the input's batch + time layout; only the feature dim is partitioned.
        z, xBC, dt = rf.split(
            proj,
            axis=self.d_in_proj_dim,
            out_dims=[self.d_inner_dim, self.d_xBC_dim, self.n_heads_dim],
        )

        if chunking is None:
            # Non-chunked: both directions over the full sequence.
            y_fwd = self._ssd_one_direction(
                xBC,
                dt,
                time_dim=spatial_dim,
                conv1d=self.conv1d_fwd,
                A_log=self.A_log_fwd,
                dt_bias=self.dt_bias_fwd,
                D=self.D_fwd,
            )
            xBC_rev = rf.reverse_sequence(xBC, axis=spatial_dim, handle_dynamic_dims=False)
            dt_rev = rf.reverse_sequence(dt, axis=spatial_dim, handle_dynamic_dims=False)
            y_bwd_rev = self._ssd_one_direction(
                xBC_rev,
                dt_rev,
                time_dim=spatial_dim,
                conv1d=self.conv1d_bwd,
                A_log=self.A_log_bwd,
                dt_bias=self.dt_bias_bwd,
                D=self.D_bwd,
            )
            y_bwd = rf.reverse_sequence(y_bwd_rev, axis=spatial_dim, handle_dynamic_dims=False)
            y = 0.5 * (y_fwd + y_bwd) * rf.silu(z)
        else:
            # Chunked: fwd uses rf.slice + rf.merge_dims for the cross-chunk flat-merge
            # (same wiring as :class:`Mamba2ChunkedLayerV2`).
            # Bwd does per-chunk reverse over [center + lookahead] via rf.reverse_sequence
            # + rf.merge_dims into a super-batch (kind=Batch so the inner BTF helpers find it).
            chunked_time = chunking.chunked_time_dim
            # Forward: unbounded-past centers + strict-R per-chunk lookahead (no rf.window overlap leak).
            from .chunked_recurrent_lookahead import mamba2_bidir_fwd_chunked

            y_fwd = mamba2_bidir_fwd_chunked(
                self,
                xBC,
                dt,
                spatial_dim=spatial_dim,
                center_dim=chunking.end_chunk_size_dim,
                chunked_time_dim=chunked_time,
            )

            xBC_rev = rf.reverse_sequence(xBC, axis=spatial_dim, handle_dynamic_dims=False)
            dt_rev = rf.reverse_sequence(dt, axis=spatial_dim, handle_dynamic_dims=False)
            xBC_batch_dims = xBC_rev.remaining_dims((spatial_dim, xBC_rev.feature_dim))
            dt_batch_dims = dt_rev.remaining_dims((spatial_dim, dt_rev.feature_dim))
            xBC_merged, super_batch_dim = rf.merge_dims(xBC_rev, dims=xBC_batch_dims)
            dt_merged, _ = rf.merge_dims(dt_rev, dims=dt_batch_dims, out_dim=super_batch_dim)
            y_bwd_merged = self._ssd_one_direction(
                xBC_merged,
                dt_merged,
                time_dim=spatial_dim,
                conv1d=self.conv1d_bwd,
                A_log=self.A_log_bwd,
                dt_bias=self.dt_bias_bwd,
                D=self.D_bwd,
            )
            y_bwd_split = rf.split_dims(y_bwd_merged, axis=super_batch_dim, dims=xBC_batch_dims)
            y_bwd = rf.reverse_sequence(y_bwd_split, axis=spatial_dim, handle_dynamic_dims=False)

            # Combine: average at center positions, keep fwd at lookahead positions
            # (fwd carries next chunk's content via the rf.window overlap).
            center_size = chunking.end_chunk_size_dim.dimension
            positions = rf.range_over_dim(spatial_dim)
            mask = positions < center_size
            y_combined = rf.where(mask, 0.5 * (y_fwd + y_bwd), y_fwd)
            # Apply silu(z) gate (z is in the same chunked layout).
            y = y_combined * rf.silu(z)

        # Shared norm + out_proj.
        y = self.norm_pre_out(y)
        y_out = self.out_proj(y)
        out = out + y_out

        # 1/2 FFN after (macaron half-step).
        if self.ffn2 is not None:
            ffn2_in = self.ffn2_layer_norm(out)
            ffn2_out = self.ffn2(ffn2_in)
            out = out + 0.5 * rf.dropout(ffn2_out, self.dropout, axis=self.out_dim)
        # Post-LN.
        if self.final_layer_norm is not None:
            out = self.final_layer_norm(out)
        return out
