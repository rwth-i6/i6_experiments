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
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from returnn.tensor import Dim, Tensor, batch_dim as _batch_dim
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
) -> torch.Tensor:
    """SSD chunked algorithm. Returns Y of shape [B, L, H, P].

    L must be a multiple of block_len.
    """
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
    return Y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_batch_dim(t: Tensor) -> Dim:
    for d in t.dims:
        if d is _batch_dim:
            return d
    for d in t.dims:
        if d.is_batch_dim():
            return d
    raise RuntimeError(f"no batch dim in {t.dims}")


def _to_BTF(t: Tensor, spatial_dim: Dim, feat_dim: Dim) -> torch.Tensor:
    b = _find_batch_dim(t)
    s = (
        spatial_dim
        if spatial_dim in t.dims
        else next(d for d in t.dims if d.dimension == spatial_dim.dimension and d.name == spatial_dim.name)
    )
    order = list(t.dims)
    return t.raw_tensor.permute(order.index(b), order.index(s), order.index(feat_dim)).contiguous()


def _from_BTF(raw: torch.Tensor, like: Tensor, spatial_dim: Dim, feat_dim: Dim, *, name: str) -> Tensor:
    b = _find_batch_dim(like)
    return Tensor(
        name,
        dims=[b, spatial_dim, feat_dim],
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
        **_ignored_kwargs: Any,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
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

    def __call__(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        chunking: Optional[_BatchChunkingSettings] = None,
        **_kwargs: Any,
    ) -> Tensor:
        x_norm = self.norm(x)
        if self.dropout:
            x_norm = rf.dropout(x_norm, self.dropout, axis=self.out_dim)

        if chunking is None:
            return x + self.mamba(x_norm, spatial_dim=spatial_dim)

        x_stride, _ = rf.slice(x_norm, axis=spatial_dim, size=chunking.end_chunk_size_dim)
        x_flat, full_time_dim = rf.merge_dims(x_stride, dims=(chunking.chunked_time_dim, chunking.end_chunk_size_dim))
        # TODO: streaming -- thread Mamba-2 state via streaming_cache_v2 across segments.
        y_flat = self.mamba(x_flat, spatial_dim=full_time_dim)
        y_windowed, new_chunked_time_dim = rf.window(
            y_flat,
            spatial_dim=full_time_dim,
            window_dim=spatial_dim,
            window_left=0,
            stride=chunking.end_chunk_size_dim.dimension,
            pad_value=0.0,
        )
        y_chunked = rf.replace_dim_v2(y_windowed, in_dim=new_chunked_time_dim, out_dim=chunking.chunked_time_dim)
        return x + y_chunked
