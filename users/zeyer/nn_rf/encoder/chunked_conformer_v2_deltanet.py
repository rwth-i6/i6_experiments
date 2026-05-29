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

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from returnn.tensor import Dim, Tensor
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
    return_final_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    states_list = []
    for c in range(n_chunks):
        q_eff_c = Q[:, c] - QW[:, c]  # [B, H, L, Dk]
        Y_off_c = torch.einsum("bhld,bhvd->bhlv", q_eff_c, S)  # [B, H, L, Dv]
        Y_list.append(Y_diag[:, c] + Y_off_c)
        WK = torch.einsum("bhld,bhle->bhde", W[:, c], K[:, c])  # [B, H, Dk, Dk]
        UK = torch.einsum("bhlv,bhld->bhvd", U[:, c], K[:, c])  # [B, H, Dv, Dk]
        S = S - torch.einsum("bhvd,bhde->bhve", S, WK) + UK
        if return_final_states:
            states_list.append(S)

    Y = torch.stack(Y_list, dim=1)  # [B, C, H, L, Dv]
    Y = rearrange(Y, "b c h l d -> b (c l) h d")
    if return_final_states:
        return Y, torch.stack(states_list, dim=1)  # [B, n_chunks, H, Dv, Dk]
    return Y


# ---------------------------------------------------------------------------
# Helpers (shared shape ops; same as mamba2 module).
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
    e.g. ``model`` while ``feat_dim`` is the projection's output ``dn-v``).
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
    """
    Pure-DeltaNet chunked-encoder layer.

    Default structure: ``x + DeltaNet(LN(x))``.
    Drop-in for ``ChunkedConformerEncoderV2.encoder_layer_per_layer``.
    Runs the DeltaNet chunked scan over the un-chunked time axis.

    Optional Conformer-flavored wrappers around the DeltaNet block,
    each gated by its own flag (default off).
    When all flags are off,
    the submodule signature is byte-identical to the bare layer,
    so the Sis hash stays stable for existing experiments.
    When on, the structure follows the standard Conformer template:
        x + 1/2 FFN1(LN(x))  ->  x + DeltaNet(LN(x))  ->  x + Conv(LN(x))  ->  x + 1/2 FFN2(LN(x))  ->  LN(x)
    with DeltaNet in place of multi-head self-attention.
    See ConMamba / Speech Slytherin (Jiang & Kim, Interspeech 2024)
    for the equivalent recipe wrapped around Mamba.
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
        # Optional Conformer-flavored wrappers (additive; defaults preserve current behavior).
        with_macaron_ff: bool = False,
        macaron_ff_dim_factor: int = 4,
        with_local_conv: bool = False,
        local_conv_kernel: int = 31,
        with_post_ln: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        assert version == 3, f"DeltaNetChunkedLayerV2: only version=3 (lookahead-correct) supported, got {version}"
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
        # Optional Conformer-flavored wrappers.
        # Always declare the attribute (default ``None``) so introspection / type checkers
        # see a consistent module shape regardless of the flags;
        # see README "RETURNN-frontend (rf) gotchas" -- "Optional submodules".
        self.ffn1_layer_norm = None
        self.ffn1 = None
        self.ffn2_layer_norm = None
        self.ffn2 = None
        self.conv_layer_norm = None
        self.conv_block = None
        self.final_layer_norm = None
        if with_macaron_ff:
            from returnn.frontend.encoder.conformer import ConformerPositionwiseFeedForward

            ff_dim = Dim(macaron_ff_dim_factor * out_dim.dimension, name="dn-ff")
            self.ffn1_layer_norm = rf.LayerNorm(out_dim)
            self.ffn1 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
            self.ffn2_layer_norm = rf.LayerNorm(out_dim)
            self.ffn2 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
        if with_local_conv:
            from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import ChunkedConformerConvBlockV2

            self.conv_layer_norm = rf.LayerNorm(out_dim)
            self.conv_block = ChunkedConformerConvBlockV2(
                out_dim=out_dim,
                kernel_size=local_conv_kernel,
                norm=rf.BatchNorm(out_dim, use_mask=False),
            )
        if with_post_ln:
            self.final_layer_norm = rf.LayerNorm(out_dim)

    def _dn_block(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        chunking: Optional[_BatchChunkingSettings],
    ) -> Tensor:
        """Run the bare DeltaNet block (chunked-aware), returning the per-position output.

        Splits the chunked vs unchunked path.
        Same math as the original inline implementation,
        extracted so the optional Conformer wrappers can call into it cleanly.
        """
        x_norm = self.norm(x)
        if self.dropout:
            x_norm = rf.dropout(x_norm, self.dropout, axis=self.out_dim)
        if chunking is None:
            return self.dn(x_norm, spatial_dim=spatial_dim)
        # Lookahead-correct chunked path: unbounded-past centers + strict-R per-chunk lookahead.
        # The old path re-windowed the global center scan (leaked/compounded the lookahead) -- removed.
        from .chunked_recurrent_lookahead import deltanet_chunked_forward

        return deltanet_chunked_forward(
            self.dn,
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
        # DeltaNet block.
        out = out + self._dn_block(out, spatial_dim=spatial_dim, chunking=chunking)
        # Local depthwise conv module (Conformer-style); reuses ChunkedConformerConvBlockV2 for chunked masking.
        if self.conv_block is not None:
            conv_in = self.conv_layer_norm(out)
            conv_out = self.conv_block(conv_in, spatial_dim=spatial_dim, chunking=chunking)
            out = out + rf.dropout(conv_out, self.dropout, axis=self.out_dim)
        # 1/2 FFN after (macaron half-step).
        if self.ffn2 is not None:
            ffn2_in = self.ffn2_layer_norm(out)
            ffn2_out = self.ffn2(ffn2_in)
            out = out + 0.5 * rf.dropout(ffn2_out, self.dropout, axis=self.out_dim)
        # Post-LN.
        if self.final_layer_norm is not None:
            out = self.final_layer_norm(out)
        return out


class BidirDeltaNetChunkedLayer(rf.Module):
    """
    Bidirectional-within-chunk DeltaNet chunked-encoder layer.

    Forward direction:
    per-chunk causal scan over the center frames
    (cross-chunk continuity via the flat-time merging trick,
    same as the causal :class:`DeltaNetChunkedLayerV2`).

    Backward direction:
    per-chunk reverse scan over the ``[center + lookahead]`` window;
    state resets per chunk -- no cross-chunk carry-over
    (lookahead provides bounded right context, history is on the wrong side).

    Outputs are averaged at center positions;
    lookahead positions keep the forward output
    (which contains the next chunk's first ``lookahead`` center frames
    via the ``rf.window`` overlap).

    Pre-norm Conformer style: ``y = x + 1/2 * (fwd(LN(x)) + bwd(LN(x)))``,
    with one shared LayerNorm on the input (matches the ConMamba / Speech
    Slytherin bidir wiring at :class:`xi-j/Mamba-ASR/modules/Conmamba.py`).

    The ``bidir_share_weights`` flag picks between
    a single shared DeltaNet block called twice on (x, flip(x))
    and two fully independent forward / backward DeltaNet blocks.
    DeltaNet has no scan-internal weights to share separately
    (unlike Mamba-2's A / D / conv1d),
    so the binary choice is the full one.

    Same optional Conformer wrappers as :class:`DeltaNetChunkedLayerV2`:
    macaron FFN, local depthwise conv, post-LN.
    All gated by their own flags; defaults preserve the bare bidir structure.
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
        bidir_share_weights: bool = False,
        # Optional Conformer-flavored wrappers (additive; defaults = bare bidir).
        with_macaron_ff: bool = False,
        macaron_ff_dim_factor: int = 4,
        with_local_conv: bool = False,
        local_conv_kernel: int = 31,
        with_post_ln: bool = False,
    ):
        """
        :param bidir_share_weights: if True,
            one DeltaNet block is shared between forward and backward directions
            (same q/k/v/beta projections, same output projection);
            the two scans differ only in input sequence order.
            If False (default),
            two independent DeltaNet blocks with separate weights are used,
            doubling the parameter count of these layers
            but giving each direction full capacity to specialize.
        :param with_macaron_ff: add a macaron 1/2-FFN before and after the bidir block.
        :param macaron_ff_dim_factor: FFN inner dim = ``out_dim * macaron_ff_dim_factor``.
        :param with_local_conv: add a Conformer-style depthwise causal conv module
            after the bidir block (uses :class:`ChunkedConformerConvBlockV2`).
        :param local_conv_kernel: kernel size of the depthwise conv when enabled.
        :param with_post_ln: add a final LayerNorm after the residual sum.
        Other parameters are forwarded verbatim to :class:`DeltaNetBlock`.
        """
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        assert version == 3, f"BidirDeltaNetChunkedLayer: only version=3 (lookahead-correct) supported, got {version}"
        self.bidir_share_weights = bidir_share_weights
        # One shared pre-LN (ConMamba-style: a single LN wraps the whole bidir block).
        self.norm = rf.LayerNorm(out_dim)
        # Forward DeltaNet block (always present).
        self.dn_fwd = DeltaNetBlock(
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
        # Backward block: present iff weights are not shared.
        if not bidir_share_weights:
            self.dn_bwd = DeltaNetBlock(
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
        # Optional Conformer-flavored wrappers (same options as the causal layer).
        # Always declare attributes (default ``None``); see README "Optional submodules".
        self.ffn1_layer_norm = None
        self.ffn1 = None
        self.ffn2_layer_norm = None
        self.ffn2 = None
        self.conv_layer_norm = None
        self.conv_block = None
        self.final_layer_norm = None
        if with_macaron_ff:
            from returnn.frontend.encoder.conformer import ConformerPositionwiseFeedForward

            ff_dim = Dim(macaron_ff_dim_factor * out_dim.dimension, name="bidir-dn-ff")
            self.ffn1_layer_norm = rf.LayerNorm(out_dim)
            self.ffn1 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
            self.ffn2_layer_norm = rf.LayerNorm(out_dim)
            self.ffn2 = ConformerPositionwiseFeedForward(
                out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=rf.swish
            )
        if with_local_conv:
            from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import ChunkedConformerConvBlockV2

            self.conv_layer_norm = rf.LayerNorm(out_dim)
            self.conv_block = ChunkedConformerConvBlockV2(
                out_dim=out_dim,
                kernel_size=local_conv_kernel,
                norm=rf.BatchNorm(out_dim, use_mask=False),
            )
        if with_post_ln:
            self.final_layer_norm = rf.LayerNorm(out_dim)

    def _bwd_block_call(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        """Dispatch to the backward block. Shares fwd weights when configured."""
        if self.bidir_share_weights:
            return self.dn_fwd(x, spatial_dim=spatial_dim)
        return self.dn_bwd(x, spatial_dim=spatial_dim)

    def _fwd_chunked(self, x_ln: Tensor, *, spatial_dim: Dim, chunking: _BatchChunkingSettings) -> Tensor:
        """Forward chunked scan -- identical to the causal layer's chunked path."""
        # Forward: unbounded-past centers + strict-R per-chunk lookahead (no rf.window overlap leak).
        from .chunked_recurrent_lookahead import deltanet_chunked_forward

        return deltanet_chunked_forward(
            self.dn_fwd,
            x_ln,
            spatial_dim=spatial_dim,
            center_dim=chunking.end_chunk_size_dim,
            chunked_time_dim=chunking.chunked_time_dim,
        )

    def _bwd_chunked(self, x_ln: Tensor, *, spatial_dim: Dim, chunking: _BatchChunkingSettings) -> Tensor:
        """Per-chunk reverse scan over ``[center + lookahead]``.

        Each chunk is processed independently (no cross-chunk state).
        Sequence of pure-rf operations:
        ``reverse_sequence`` along the within-chunk spatial axis,
        ``merge_dims`` to fold all batch-like dims (batch + chunked_time)
        into a super-batch so each chunk becomes an independent batch element,
        run the backward DeltaNet block,
        then the inverse: ``split_dims`` + ``reverse_sequence`` again.
        Output dims match the input layout.
        Values at the lookahead positions are computed but discarded by the caller
        (the combine step keeps the forward output there for cross-chunk continuity).
        """
        x_rev = rf.reverse_sequence(x_ln, axis=spatial_dim, handle_dynamic_dims=False)
        batch_dims = x_rev.remaining_dims((spatial_dim, x_rev.feature_dim))
        x_merged, super_batch_dim = rf.merge_dims(x_rev, dims=batch_dims)
        y_merged = self._bwd_block_call(x_merged, spatial_dim=spatial_dim)
        y_split = rf.split_dims(y_merged, axis=super_batch_dim, dims=batch_dims)
        return rf.reverse_sequence(y_split, axis=spatial_dim, handle_dynamic_dims=False)

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

        # Bidirectional DeltaNet (one shared pre-LN feeds both directions).
        x_ln = self.norm(out)
        if self.dropout:
            x_ln = rf.dropout(x_ln, self.dropout, axis=self.out_dim)

        if chunking is None:
            # Non-chunked: bidir over the whole sequence.
            y_fwd = self.dn_fwd(x_ln, spatial_dim=spatial_dim)
            x_ln_rev = rf.reverse_sequence(x_ln, axis=spatial_dim, handle_dynamic_dims=False)
            y_bwd_rev = self._bwd_block_call(x_ln_rev, spatial_dim=spatial_dim)
            y_bwd = rf.reverse_sequence(y_bwd_rev, axis=spatial_dim, handle_dynamic_dims=False)
            out = out + 0.5 * (y_fwd + y_bwd)
        else:
            y_fwd = self._fwd_chunked(x_ln, spatial_dim=spatial_dim, chunking=chunking)
            y_bwd = self._bwd_chunked(x_ln, spatial_dim=spatial_dim, chunking=chunking)
            # Combine: at center positions average fwd + bwd;
            # at lookahead positions keep fwd unchanged
            # (it carries the next chunk's content via rf.window overlap).
            center_size = chunking.end_chunk_size_dim.dimension
            positions = rf.range_over_dim(spatial_dim)
            mask = positions < center_size
            y_combined = rf.where(mask, 0.5 * (y_fwd + y_bwd), y_fwd)
            out = out + y_combined

        # Optional Conformer wrappers, same order as DeltaNetChunkedLayerV2.
        if self.conv_block is not None:
            conv_in = self.conv_layer_norm(out)
            conv_out = self.conv_block(conv_in, spatial_dim=spatial_dim, chunking=chunking)
            out = out + rf.dropout(conv_out, self.dropout, axis=self.out_dim)
        if self.ffn2 is not None:
            ffn2_in = self.ffn2_layer_norm(out)
            ffn2_out = self.ffn2(ffn2_in)
            out = out + 0.5 * rf.dropout(ffn2_out, self.dropout, axis=self.out_dim)
        if self.final_layer_norm is not None:
            out = self.final_layer_norm(out)
        return out
