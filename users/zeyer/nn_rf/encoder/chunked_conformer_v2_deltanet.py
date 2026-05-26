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
    Q: torch.Tensor,  # [B, L, H, Dk]
    K: torch.Tensor,  # [B, L, H, Dk]
    V: torch.Tensor,  # [B, L, H, Dv]
    beta: torch.Tensor,  # [B, L, H]   gate in (0, 1)
    block_len: int,
    initial_state: Optional[torch.Tensor] = None,  # [B, H, Dv, Dk] or None
) -> torch.Tensor:
    """Chunked parallel DeltaNet scan (closed-form WY representation).

    Returns Y of shape [B, L, H, Dv]. L must be a multiple of block_len.

    Recurrence (S of shape [Dv, Dk]):
        S_l = S_{l-1} (I - beta_l k_l k_l^T) + beta_l v_l k_l^T
        y_l = S_l q_l
    With u_l = beta_l (v_l - S_{l-1} k_l) the recurrence becomes a rank-1
    update S_l = S_{l-1} + u_l k_l^T, and within one block u solves the
    unit-lower-triangular system
        M u = beta V - beta (S_0 K^T)^T,   M = I + tril(beta * K K^T, -1)
    Splitting the rhs gives u = U - W S_0^T where U = M^{-1} (beta V) and
    W = M^{-1} (beta K). The block output and end-state are then
        y_l    = sum_{j<=l} <k_j, q_l> U_j + (q_l - sum_{j<=l} <k_j, q_l> W_j) S_0^T
        S_new  = S_0 - S_0 (W^T K) + U^T K
    See Yang et al. 2024 (https://arxiv.org/abs/2406.06484) eqns (8)-(9).

    The triangular solves run in fp32 even when Q/K/V are bf16/fp16: a bf16
    unit-triangular solve with block_len=32 and near-correlated L2-normed
    keys blows up to NaN on the first chunks.
    """
    L_ = Q.size(1)
    assert L_ % block_len == 0
    B, H = Q.size(0), Q.size(2)
    Dk, Dv = Q.size(3), V.size(3)
    dtype = Q.dtype

    Q = rearrange(Q, "b (c l) h d -> b c h l d", l=block_len)  # [B, C, H, L, Dk]
    K = rearrange(K, "b (c l) h d -> b c h l d", l=block_len)
    V = rearrange(V, "b (c l) h d -> b c h l d", l=block_len)
    beta = rearrange(beta, "b (c l) h -> b c h l", l=block_len)  # [B, C, H, L]

    # M = I + tril(beta * K K^T, -1), unit lower triangular.
    KK = torch.einsum("bchld,bchsd->bchls", K, K).float() * beta.unsqueeze(-1).float()
    eye = torch.eye(block_len, device=Q.device, dtype=torch.float32).expand_as(KK)
    M = eye + torch.tril(KK, diagonal=-1)  # [B, C, H, L, L], fp32
    # U = M^{-1} (beta V), W = M^{-1} (beta K). Same M, two solves.
    tilde_V = (V * beta.unsqueeze(-1)).float()
    tilde_K = (K * beta.unsqueeze(-1)).float()
    U = torch.linalg.solve_triangular(M, tilde_V, upper=False, unitriangular=True).to(dtype)
    W = torch.linalg.solve_triangular(M, tilde_K, upper=False, unitriangular=True).to(dtype)

    # Causal-masked QK (lower triangular incl. diagonal).
    QK = torch.einsum("bchld,bchsd->bchls", Q, K)
    causal_mask = torch.tril(torch.ones(block_len, block_len, device=Q.device, dtype=torch.bool))
    QK = QK.masked_fill(~causal_mask, 0.0)
    Y_diag = torch.einsum("bchls,bchsd->bchld", QK, U)  # [B, C, H, L, Dv]
    QW = torch.einsum("bchls,bchsd->bchld", QK, W)  # [B, C, H, L, Dk]

    # Cross-chunk: closed-form block update (no per-step loop).
    if initial_state is None:
        S = Q.new_zeros((B, H, Dv, Dk))
    else:
        S = initial_state
    n_chunks = Q.size(1)
    Y_list = []
    for c in range(n_chunks):
        q_eff_c = Q[:, c] - QW[:, c]  # [B, H, L, Dk]
        Y_off_c = torch.einsum("bhld,bhvd->bhlv", q_eff_c, S)  # [B, H, L, Dv]
        Y_list.append(Y_diag[:, c] + Y_off_c)
        WK = torch.einsum("bhld,bhle->bhde", W[:, c], K[:, c])  # [B, H, Dk, Dk]
        UK = torch.einsum("bhlv,bhld->bhvd", U[:, c], K[:, c])  # [B, H, Dv, Dk]
        S = S - torch.einsum("bhvd,bhde->bhve", S, WK) + UK

    Y = torch.stack(Y_list, dim=1)  # [B, C, H, L, Dv]
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
        normalize_eps: float = 1e-5,
        beta_init: float = -4.0,
        version: int = 1,
    ):
        """
        :param in_dim: input/output model dim.
        :param d_state: head key/value dim (Dk = Dv = d_state).
        :param expand: inner expansion factor (d_inner = in_dim * expand).
        :param head_dim: per-head feature dim. Determines n_heads = d_inner / head_dim
            (if ``n_heads`` not given).
        :param n_heads: number of heads (overrides head_dim if given).
        :param block_len: chunk length for the chunkwise scan.
        :param normalize_eps: eps for L2-normalizing q, k. Much larger than
            F.normalize's default 1e-12 so near-zero untrained keys don't get
            inflated to huge unit-vectors during warmup.
        :param beta_init: initial constant for the beta-projection bias. With
            sigmoid on top, default -4.0 gives beta ~ 0.018 at init, keeping
            the within-block state-decay near identity until training has
            stabilized.
        :param version: implementation version. v1 (broken) had a sign error
            in the within-block triangular factor and used bf16 for the solve;
            v>=2 uses the corrected sign, fp32 solves, the WY closed-form for
            cross-chunk state propagation, and the normalize_eps / beta_init
            knobs above.
        """
        super().__init__()
        assert version >= 2, f"DeltaNetBlock: version must be >= 2 (v1 broken); got {version}"
        self.in_dim = in_dim
        self.normalize_eps = normalize_eps
        self.version = version
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
        # Init beta-projection bias to beta_init so sigmoid(beta) ~ 0.018 at start.
        self.beta_proj.bias.initial = beta_init

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
        q = F.normalize(q, dim=-1, eps=self.normalize_eps)
        k = F.normalize(k, dim=-1, eps=self.normalize_eps)

        # Pad to block_len.
        q_pad, T_orig = _pad_to_block_len(q, self.block_len, time_dim=1)
        k_pad, _ = _pad_to_block_len(k, self.block_len, time_dim=1)
        v_pad, _ = _pad_to_block_len(v, self.block_len, time_dim=1)
        beta_pad, _ = _pad_to_block_len(beta, self.block_len, time_dim=1)

        # Scan.
        y_pad = _delta_rule_chunked(q_pad, k_pad, v_pad, beta_pad, block_len=self.block_len)
        y_t = y_pad[:, :T_orig]  # [B, T, H, head_dim]
        y_t = y_t.reshape(b, T, H * self.head_dim)  # [B, T, d_v]
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
        normalize_eps: float = 1e-5,
        beta_init: float = -4.0,
        version: int = 1,
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
            normalize_eps=normalize_eps,
            beta_init=beta_init,
            version=version,
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
        x_flat, full_time_dim = rf.merge_dims(x_stride, dims=(chunking.chunked_time_dim, chunking.end_chunk_size_dim))
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
        y_chunked = rf.replace_dim_v2(y_windowed, in_dim=new_chunked_time_dim, out_dim=chunking.chunked_time_dim)
        return x + y_chunked
