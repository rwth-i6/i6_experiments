"""
Lookahead-correct chunked forward for the recurrent (Mamba-2 / DeltaNet) chunked layers.

The original chunked path ran the recurrent scan over the merged centers (unbounded past, good),
but reconstructed the lookahead by re-windowing that global output (rf.window) -- which pulls the
*next* chunk's centers into the current chunk's lookahead, so the effective lookahead compounds by
~R per recurrent layer (very bad latency).

This computes the lookahead correctly: unbounded-past centers via the merged-center scan (then a
clean un-merge back to chunks, no rf.window overlap), plus a strict-R per-chunk lookahead via a
state-seeded continuation over each chunk's own R lookahead frames (conv context taken from the
contiguous window, so no separate conv-state is needed). The lookahead then stays a strict R and
does not compound across layers; the center outputs are unchanged.

Used by the causal layers (Mamba2ChunkedLayerV2 / DeltaNetChunkedLayerV2) and by the forward
direction of the bidir layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from returnn.tensor import Tensor


def _batch_dim_of(x, spatial_dim, chunked_time_dim):
    batch_dims = [d for d in x.remaining_dims((spatial_dim, x.feature_dim)) if d != chunked_time_dim]
    assert len(batch_dims) == 1, f"expected one batch dim, got {batch_dims} for {x}"
    return batch_dims[0]


# --------------------------------------------------------------------------- Mamba-2


def _mamba2_center_lookahead(
    xBC_t, dt_t, *, conv_w, conv_b, a_vec, dt_bias, d_view,
    center, n_heads, head_dim, n_groups, d_state, d_inner, d_xBC, d_conv,
):
    """One-direction center+lookahead Mamba-2 SSD (pure torch).

    xBC_t: [B, Ct, window, d_xBC] (post in_proj, pre conv).  dt_t: [B, Ct, window, n_heads].
    Returns y_inner: [B, Ct, window, d_inner] (after the D skip, before gate / out_proj).
    """
    from .chunked_conformer_v2_mamba2 import _ssd_chunked

    Bb, Ct, window = xBC_t.shape[0], xBC_t.shape[1], xBC_t.shape[2]
    lookahead = window - center

    def _conv(seq):
        out = F.conv1d(F.pad(seq.transpose(1, 2), (d_conv - 1, 0)), conv_w, bias=conv_b, groups=d_xBC)
        return F.silu(out).transpose(1, 2)

    def _heads(xc, T, Bx):
        x_in = xc[..., :d_inner].reshape(Bx, T, n_heads, head_dim)
        b_s = xc[..., d_inner : d_inner + n_groups * d_state].reshape(Bx, T, n_groups, d_state)
        c_s = xc[..., d_inner + n_groups * d_state :].reshape(Bx, T, n_groups, d_state)
        if n_groups != n_heads:
            b_s = b_s.repeat_interleave(n_heads // n_groups, dim=2)
            c_s = c_s.repeat_interleave(n_heads // n_groups, dim=2)
        return x_in, b_s, c_s

    # center: merged contiguous conv + unbounded scan; block_len=center => per-chunk boundary states
    xBC_c = _conv(xBC_t[:, :, :center, :].reshape(Bb, Ct * center, d_xBC))
    x_in_c, b_c, c_c = _heads(xBC_c, Ct * center, Bb)
    dt_c = dt_t[:, :, :center, :].reshape(Bb, Ct * center, n_heads)
    a_c = F.softplus(dt_c + dt_bias) * a_vec
    y_c, fstates = _ssd_chunked(x_in_c, a_c, b_c, c_c, block_len=center, return_final_states=True)
    y_c = y_c.reshape(Bb, Ct, center, n_heads, head_dim)
    x_in_c = x_in_c.reshape(Bb, Ct, center, n_heads, head_dim)

    if lookahead > 0:
        # windowed conv gives correct conv context for the lookahead frames (contiguous window)
        xBC_w = _conv(xBC_t.reshape(Bb * Ct, window, d_xBC)).reshape(Bb, Ct, window, d_xBC)
        xBC_l = xBC_w[:, :, center:, :].reshape(Bb * Ct, lookahead, d_xBC)
        x_in_l, b_l, c_l = _heads(xBC_l, lookahead, Bb * Ct)
        dt_l = dt_t[:, :, center:, :].reshape(Bb * Ct, lookahead, n_heads)
        a_l = F.softplus(dt_l + dt_bias) * a_vec
        init = fstates.reshape(Bb * Ct, 1, n_heads, head_dim, d_state)
        y_l = _ssd_chunked(x_in_l, a_l, b_l, c_l, block_len=lookahead, initial_states=init)
        y_l = y_l.reshape(Bb, Ct, lookahead, n_heads, head_dim)
        x_in_l = x_in_l.reshape(Bb, Ct, lookahead, n_heads, head_dim)
        y_w = torch.cat([y_c, y_l], dim=2)
        x_in_w = torch.cat([x_in_c, x_in_l], dim=2)
    else:
        y_w, x_in_w = y_c, x_in_c

    y_w = y_w + x_in_w * d_view
    return y_w.reshape(Bb, Ct, window, d_inner)


def mamba2_chunked_forward(block, x, *, spatial_dim, center_dim, chunked_time_dim):
    """Full Mamba2Block chunked forward (in_proj -> center+lookahead -> gate -> norm -> out_proj)."""
    batch_dim = _batch_dim_of(x, spatial_dim, chunked_time_dim)
    proj_t = (
        block.in_proj(x)
        .copy_compatible_to_dims_raw([batch_dim, chunked_time_dim, spatial_dim, block.d_in_proj_dim])
        .contiguous()
    )
    d_inner = block.d_inner
    d_xBC = d_inner + 2 * block.n_groups * block.d_state
    z_t = proj_t[..., :d_inner]
    xBC_t = proj_t[..., d_inner : d_inner + d_xBC]
    dt_t = proj_t[..., d_inner + d_xBC :]
    y_w = _mamba2_center_lookahead(
        xBC_t, dt_t,
        conv_w=block.conv1d.filter.raw_tensor,
        conv_b=block.conv1d.bias.raw_tensor if block.conv1d.with_bias else None,
        a_vec=-torch.exp(block.A_log.raw_tensor),
        dt_bias=block.dt_bias.raw_tensor,
        d_view=block.D.raw_tensor.view(1, 1, 1, block.n_heads, 1),
        center=center_dim.dimension,
        n_heads=block.n_heads, head_dim=block.head_dim, n_groups=block.n_groups,
        d_state=block.d_state, d_inner=d_inner, d_xBC=d_xBC, d_conv=block.d_conv,
    )
    y_w = y_w * F.silu(z_t)
    y = Tensor(
        "m2_y_chunked", dims=[batch_dim, chunked_time_dim, spatial_dim, block.d_inner_dim],
        dtype=str(y_w.dtype).split(".")[-1], raw_tensor=y_w.contiguous(), feature_dim=block.d_inner_dim,
    )
    return block.out_proj(block.norm_pre_out(y))


def mamba2_bidir_fwd_chunked(layer, xBC, dt, *, spatial_dim, center_dim, chunked_time_dim):
    """Forward-direction center+lookahead Mamba-2 for the bidir layer (shared in_proj already applied).

    Returns rf [batch, chunked_time, window, d_inner_dim] (before gate / shared norm+out_proj),
    matching the output role of BidirMamba2ChunkedLayer._ssd_one_direction.
    """
    batch_dim = _batch_dim_of(xBC, spatial_dim, chunked_time_dim)
    xBC_t = xBC.copy_compatible_to_dims_raw([batch_dim, chunked_time_dim, spatial_dim, layer.d_xBC_dim]).contiguous()
    dt_t = dt.copy_compatible_to_dims_raw([batch_dim, chunked_time_dim, spatial_dim, layer.n_heads_dim]).contiguous()
    y_w = _mamba2_center_lookahead(
        xBC_t, dt_t,
        conv_w=layer.conv1d_fwd.filter.raw_tensor,
        conv_b=layer.conv1d_fwd.bias.raw_tensor if layer.conv1d_fwd.with_bias else None,
        a_vec=-torch.exp(layer.A_log_fwd.raw_tensor),
        dt_bias=layer.dt_bias_fwd.raw_tensor,
        d_view=layer.D_fwd.raw_tensor.view(1, 1, 1, layer.n_heads, 1),
        center=center_dim.dimension,
        n_heads=layer.n_heads, head_dim=layer.head_dim, n_groups=layer.n_groups,
        d_state=layer.d_state, d_inner=layer.d_inner, d_xBC=layer.d_xBC, d_conv=layer.d_conv,
    )
    return Tensor(
        "m2_bidir_fwd_y", dims=[batch_dim, chunked_time_dim, spatial_dim, layer.d_inner_dim],
        dtype=str(y_w.dtype).split(".")[-1], raw_tensor=y_w.contiguous(), feature_dim=layer.d_inner_dim,
    )


# --------------------------------------------------------------------------- DeltaNet


def deltanet_chunked_forward(block, x, *, spatial_dim, center_dim, chunked_time_dim):
    """Full DeltaNetBlock chunked forward (works for the causal layer and the bidir layer's dn_fwd)."""
    from .chunked_conformer_v2_deltanet import _delta_rule_chunked

    H = block.n_heads
    d_state = block.d_state  # Dk
    head_dim = block.head_dim  # Dv
    center = center_dim.dimension
    window = spatial_dim.dimension
    lookahead = window - center

    batch_dim = _batch_dim_of(x, spatial_dim, chunked_time_dim)

    def _raw(p, feat_dim):
        return p.copy_compatible_to_dims_raw([batch_dim, chunked_time_dim, spatial_dim, feat_dim]).contiguous()

    q = _raw(block.q_proj(x), block.d_qk_dim)
    k = _raw(block.k_proj(x), block.d_qk_dim)
    v = _raw(block.v_proj(x), block.d_v_dim)
    beta = _raw(block.beta_proj(x), block.d_beta_dim)  # [B, Ct, W, H]
    Bb, Ct = q.shape[0], q.shape[1]
    q = F.normalize(q.reshape(Bb, Ct, window, H, d_state), dim=-1, eps=block.normalize_eps)
    k = F.normalize(k.reshape(Bb, Ct, window, H, d_state), dim=-1, eps=block.normalize_eps)
    v = v.reshape(Bb, Ct, window, H, head_dim)
    beta = torch.sigmoid(beta)

    q_c = q[:, :, :center].reshape(Bb, Ct * center, H, d_state)
    k_c = k[:, :, :center].reshape(Bb, Ct * center, H, d_state)
    v_c = v[:, :, :center].reshape(Bb, Ct * center, H, head_dim)
    beta_c = beta[:, :, :center].reshape(Bb, Ct * center, H)
    y_c, fstates = _delta_rule_chunked(q_c, k_c, v_c, beta_c, block_len=center, return_final_states=True)
    y_c = y_c.reshape(Bb, Ct, center, H, head_dim)

    if lookahead > 0:
        q_l = q[:, :, center:].reshape(Bb * Ct, lookahead, H, d_state)
        k_l = k[:, :, center:].reshape(Bb * Ct, lookahead, H, d_state)
        v_l = v[:, :, center:].reshape(Bb * Ct, lookahead, H, head_dim)
        beta_l = beta[:, :, center:].reshape(Bb * Ct, lookahead, H)
        init = fstates.reshape(Bb * Ct, H, head_dim, d_state)
        y_l = _delta_rule_chunked(q_l, k_l, v_l, beta_l, block_len=lookahead, initial_state=init)
        y_l = y_l.reshape(Bb, Ct, lookahead, H, head_dim)
        y_w = torch.cat([y_c, y_l], dim=2)
    else:
        y_w = y_c

    y_w = y_w.reshape(Bb, Ct, window, H * head_dim)
    y = Tensor(
        "dn_y_chunked", dims=[batch_dim, chunked_time_dim, spatial_dim, block.d_v_dim],
        dtype=str(y_w.dtype).split(".")[-1], raw_tensor=y_w.contiguous(), feature_dim=block.d_v_dim,
    )
    return block.out_proj(y)
