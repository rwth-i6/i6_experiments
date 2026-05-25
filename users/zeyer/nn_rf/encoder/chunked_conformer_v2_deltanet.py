"""DeltaNet layer for ChunkedConformerEncoderV2.

Implements the parallel-delta-rule linear-attention scan from Yang et al. 2024,
"Parallelizing Linear Transformers with the Delta Rule"
(https://arxiv.org/abs/2406.06484), in pure PyTorch via chunkwise computation.

DeltaNet recurrence (per head, with state matrix S of shape [d_v, d_k]):

    S_t = (I - β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T
        = S_{t-1} - β_t (S_{t-1} k_t) k_t^T + β_t v_t k_t^T
    y_t = S_t q_t

where:
    q_t, k_t in R^{d_k}     (L2-normalized -- the paper uses normalized keys)
    v_t in R^{d_v}
    β_t in R                (learned gate, sigmoid)

The chunkwise parallel form within a block of length L:

    Let K, V be stacked over the block (L x d_k, L x d_v) and let β be the
    diagonal gate matrix. Then the block's contribution to the state is

        delta_block = T^{-1} (β * V)

    where T = I - tril(β * K K^T, diag=-1) is the lower-triangular delta
    factor. Within-chunk outputs use this; cross-chunk state recurrence
    applies the same delta factor with the chunk-end state.

Pure-PyTorch reference port. ~150 LoC for the scan. Adapted from the
flash-linear-attention reference impl
(https://github.com/sustcsonglin/flash-linear-attention).

Streaming-mode state-passing across forward calls is a TODO; offline training
and offline recog work today (each forward initializes state to zeros).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from returnn.tensor import Dim, Tensor, batch_dim as _batch_dim
import returnn.frontend as rf

from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import _BatchChunkingSettings


# ---------------------------------------------------------------------------
# DeltaNet chunked scan (PyTorch)
# ---------------------------------------------------------------------------


def _delta_rule_chunked(
    Q: torch.Tensor,   # [B, L, H, Dk]
    K: torch.Tensor,   # [B, L, H, Dk]
    V: torch.Tensor,   # [B, L, H, Dv]
    beta: torch.Tensor,  # [B, L, H]   gate in (0, 1)
    block_len: int,
    initial_state: Optional[torch.Tensor] = None,  # [B, H, Dv, Dk] or None
) -> torch.Tensor:
    """Chunked parallel DeltaNet scan.

    Returns Y of shape [B, L, H, Dv]. L must be a multiple of block_len.

    Algorithm (per chunk):
      1. Within-chunk: solve a small triangular system to get the per-token
         contributions, expressed as matmuls.
      2. Cross-chunk: propagate the state through the chunk-end via the
         block's delta operator.
    """
    L = Q.size(1)
    assert L % block_len == 0
    B, H = Q.size(0), Q.size(2)
    Dk, Dv = Q.size(3), V.size(3)

    Q = rearrange(Q, "b (c l) h d -> b c h l d", l=block_len)        # [B, C, H, L, Dk]
    K = rearrange(K, "b (c l) h d -> b c h l d", l=block_len)
    V = rearrange(V, "b (c l) h d -> b c h l d", l=block_len)
    beta = rearrange(beta, "b (c l) h -> b c h l", l=block_len)      # [B, C, H, L]

    # Gate-weighted V and K: tilde_V = beta * V, tilde_K = beta * K.
    tilde_V = V * beta.unsqueeze(-1)                                  # [B, C, H, L, Dv]

    # Within-chunk attention-like matmul: A = K Q^T (per chunk).
    # KK^T inside the chunk (lower-triangular).
    KK = torch.einsum("bchld,bchsd->bchls", K, K)                     # [B, C, H, L, L]
    KK = KK * beta.unsqueeze(-1)                                      # scale rows by beta
    # T_inv = (I - tril(KK, -1))^{-1} computed via Neumann-like accumulation.
    # For small L (block_len ~32-64) we just invert.
    eye = torch.eye(block_len, device=Q.device, dtype=Q.dtype).expand_as(KK)
    L_mat = eye - torch.tril(KK, diagonal=-1)                          # [B, C, H, L, L]
    # Solve L_mat * U = tilde_V for U.  U = L_mat^{-1} tilde_V.
    # L_mat is lower triangular with 1s on the diag, so triangular_solve is exact.
    U = torch.linalg.solve_triangular(L_mat, tilde_V, upper=False, unitriangular=True)
    # U[b,c,h,l,:] = decomposed value at position l within the chunk.

    # Within-chunk output: Y_diag = (Q K^T) * tril(diag=0) ... * (something with U).
    # Following the FLA reference: within-chunk output = causal-masked
    # softmax-free attention with V replaced by U.
    QK = torch.einsum("bchld,bchsd->bchls", Q, K)                     # [B, C, H, L, L]
    causal_mask = torch.tril(torch.ones(block_len, block_len, device=Q.device, dtype=torch.bool))
    QK = QK.masked_fill(~causal_mask, 0.0)
    Y_diag = torch.einsum("bchls,bchsd->bchld", QK, U)                # [B, C, H, L, Dv]

    # Cross-chunk: maintain state S, [B, H, Dv, Dk]. Init zeros (or initial_state).
    # The state after a chunk: S' = block_op(S, K, U).
    # Per-chunk block_op: S' = S - sum_l (S k_l) (k_l^T) * beta_l + sum_l v'_l k_l^T (with v' = U).
    #                    = S (I - tilde_K tilde_K^T_sum) + tilde_K^T U  -- block matrix form
    # Equivalently: S' = (S - S K^T diag(beta) K) + U^T K   (using diag-beta absorbed in tilde_K).
    # Concretely:
    #   state_decay = (I - sum_l beta_l k_l k_l^T)  is hard to materialize for d_k > L; use
    #   the running form: for each l from 0..L-1: S <- (I - beta_l k_l k_l^T) S + beta_l v_l k_l^T.
    # That's the sequential form. For pure-PyTorch chunked, we can do block matmul:
    #   K_chunk: [L, Dk]. The block update applies a sequence of rank-1 modifications. We
    #   express it via U (already computed):
    #       S_after = S_before - K^T (beta * (Q ... )) + K^T U ...
    # See FLA implementation for the closed form. Here we keep it simple and correct via
    # the per-time loop INSIDE each chunk (still vectorized across batch / heads).
    # For block_len ~32-64 this loop is small (32-64 iterations) and runs entirely on GPU.
    if initial_state is None:
        S = Q.new_zeros((B, H, Dv, Dk))
    else:
        S = initial_state
    n_chunks = Q.size(1)
    Y_off_list = []
    for c in range(n_chunks):
        # Cross-chunk contribution to this chunk's output:
        # y_l += q_l S (state at chunk start), for each l in this chunk.
        Y_off_c = torch.einsum("bhld,bhvd->bhlv", Q[:, c], S)          # [B, H, L, Dv]
        Y_off_list.append(Y_off_c)

        # Update S: apply each step in the chunk sequentially (cheap for small L).
        # S <- (I - beta_l k_l k_l^T) S + beta_l v_l k_l^T
        K_c = K[:, c]                                                  # [B, H, L, Dk]
        V_c = V[:, c]
        beta_c = beta[:, c]                                            # [B, H, L]
        for li in range(block_len):
            k_l = K_c[:, :, li, :]                                     # [B, H, Dk]
            v_l = V_c[:, :, li, :]                                     # [B, H, Dv]
            b_l = beta_c[:, :, li]                                     # [B, H]
            # delta = beta_l * (S @ k_l)              [B, H, Dv]
            delta = b_l.unsqueeze(-1) * torch.einsum("bhvd,bhd->bhv", S, k_l)
            # outer subtraction: S -= delta . k_l^T
            S = S - delta.unsqueeze(-1) * k_l.unsqueeze(-2)
            # outer addition: S += beta_l * v_l . k_l^T
            add = (b_l.unsqueeze(-1) * v_l).unsqueeze(-1) * k_l.unsqueeze(-2)
            S = S + add

    Y_off = torch.stack(Y_off_list, dim=1)                              # [B, C, H, L, Dv]
    Y = Y_diag + Y_off                                                  # [B, C, H, L, Dv]
    Y = rearrange(Y, "b c h l d -> b (c l) h d")
    return Y


# ---------------------------------------------------------------------------
# Helpers (shared shape ops; same as mamba2 module).
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
    s = spatial_dim if spatial_dim in t.dims else next(
        d for d in t.dims if d.dimension == spatial_dim.dimension and d.name == spatial_dim.name
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
    T = x.size(time_dim)
    pad = (-T) % block_len
    if pad == 0:
        return x, T
    pad_shape = list(x.shape)
    pad_shape[time_dim] = pad
    return torch.cat([x, x.new_zeros(pad_shape)], dim=time_dim), T


# ---------------------------------------------------------------------------
# DeltaNet block
# ---------------------------------------------------------------------------


class DeltaNetBlock(rf.Module):
    """DeltaNet linear-attention block with delta-rule state update.

    Forward = Linear(x -> q, k, v, beta) -> L2-normalize q, k -> chunked
    delta-rule scan -> head merge -> Linear(out_proj).
    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        d_state: int = 128,
        expand: int = 1,
        head_dim: int = 64,
        n_heads: Optional[int] = None,
        block_len: int = 32,
    ):
        """
        :param in_dim: input/output model dim.
        :param d_state: head key/value dim (Dk = Dv = d_state).
        :param expand: inner expansion factor (d_inner = in_dim * expand).
        :param head_dim: per-head feature dim. Determines n_heads = d_inner / head_dim
            (if ``n_heads`` not given).
        :param n_heads: number of heads (overrides head_dim if given).
        :param block_len: chunk length for the chunkwise scan.
        """
        super().__init__()
        self.in_dim = in_dim
        d_inner = in_dim.dimension * expand
        if n_heads is None:
            assert d_inner % head_dim == 0
            n_heads = d_inner // head_dim
        else:
            head_dim = d_inner // n_heads
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.block_len = block_len

        # Projections: q, k each [n_heads * d_state]; v [n_heads * head_dim]; beta [n_heads].
        self.d_qk_dim = Dim(n_heads * d_state, name="dn-qk")
        self.d_v_dim = Dim(n_heads * head_dim, name="dn-v")
        self.d_beta_dim = Dim(n_heads, name="dn-beta")
        self.q_proj = rf.Linear(in_dim, self.d_qk_dim, with_bias=False)
        self.k_proj = rf.Linear(in_dim, self.d_qk_dim, with_bias=False)
        self.v_proj = rf.Linear(in_dim, self.d_v_dim, with_bias=False)
        self.beta_proj = rf.Linear(in_dim, self.d_beta_dim, with_bias=True)

        # Output projection. Note: input to out_proj is d_v (n_heads * head_dim).
        self.out_proj = rf.Linear(self.d_v_dim, in_dim, with_bias=False)

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        """x: [B, T, in_dim] -> y: [B, T, in_dim]."""
        # Extract torch tensors.
        q = _to_BTF(self.q_proj(x), spatial_dim, self.d_qk_dim)
        k = _to_BTF(self.k_proj(x), spatial_dim, self.d_qk_dim)
        v = _to_BTF(self.v_proj(x), spatial_dim, self.d_v_dim)
        beta = _to_BTF(self.beta_proj(x), spatial_dim, self.d_beta_dim)
        b, T, _ = q.shape
        H = self.n_heads

        # Reshape into heads.
        q = q.reshape(b, T, H, self.d_state)
        k = k.reshape(b, T, H, self.d_state)
        v = v.reshape(b, T, H, self.head_dim)
        beta = torch.sigmoid(beta)  # gate in (0, 1)

        # L2-normalize q, k (DeltaNet convention).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Pad to block_len.
        q_pad, T_orig = _pad_to_block_len(q, self.block_len, time_dim=1)
        k_pad, _ = _pad_to_block_len(k, self.block_len, time_dim=1)
        v_pad, _ = _pad_to_block_len(v, self.block_len, time_dim=1)
        beta_pad, _ = _pad_to_block_len(beta, self.block_len, time_dim=1)

        # Scan.
        y_pad = _delta_rule_chunked(q_pad, k_pad, v_pad, beta_pad, block_len=self.block_len)
        y_t = y_pad[:, :T_orig]                       # [B, T, H, head_dim]
        y_t = y_t.reshape(b, T, H * self.head_dim)    # [B, T, d_v]
        y = _from_BTF(y_t, x, spatial_dim, self.d_v_dim, name="dn_y")
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Encoder layer wrapper
# ---------------------------------------------------------------------------


class DeltaNetChunkedLayerV2(rf.Module):
    """Pure-DeltaNet chunked-encoder layer (LayerNorm + DeltaNet + residual).

    Drop-in for ``ChunkedConformerEncoderV2.encoder_layer_per_layer``. Runs the
    DeltaNet chunked scan over the un-chunked time axis.
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        d_state: int = 128,
        expand: int = 1,
        head_dim: int = 64,
        n_heads: Optional[int] = None,
        block_len: int = 32,
        dropout: float = 0.0,
        **_ignored_kwargs: Any,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.norm = rf.LayerNorm(out_dim)
        self.dn = DeltaNetBlock(
            out_dim,
            d_state=d_state,
            expand=expand,
            head_dim=head_dim,
            n_heads=n_heads,
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
            return x + self.dn(x_norm, spatial_dim=spatial_dim)

        x_stride, _ = rf.slice(x_norm, axis=spatial_dim, size=chunking.end_chunk_size_dim)
        x_flat, full_time_dim = rf.merge_dims(
            x_stride, dims=(chunking.chunked_time_dim, chunking.end_chunk_size_dim)
        )
        # TODO: streaming -- thread DeltaNet state via streaming_cache_v2 across segments.
        y_flat = self.dn(x_flat, spatial_dim=full_time_dim)
        y_windowed, new_chunked_time_dim = rf.window(
            y_flat,
            spatial_dim=full_time_dim,
            window_dim=spatial_dim,
            window_left=0,
            stride=chunking.end_chunk_size_dim.dimension,
            pad_value=0.0,
        )
        y_chunked = rf.replace_dim_v2(
            y_windowed, in_dim=new_chunked_time_dim, out_dim=chunking.chunked_time_dim
        )
        return x + y_chunked
