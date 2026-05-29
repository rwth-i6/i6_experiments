"""
Lookahead-correct chunked forward for the recurrent (Mamba-2 / DeltaNet) chunked layers.

The original chunked path ran the recurrent scan over the merged centers (unbounded past, good),
but reconstructed the lookahead by re-windowing that global output -- which pulls the *next*
chunk's centers into the current chunk's lookahead, so the effective lookahead compounds by ~R
per recurrent layer (very bad latency).

This computes the lookahead correctly: continue the scan from each chunk's boundary state over
that chunk's own R lookahead frames (conv context taken from the contiguous window, so no extra
conv-state is needed). The lookahead then stays a strict R and does not compound across layers.
The center outputs are unchanged (still the unbounded-past causal scan over the merged centers).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from returnn.tensor import Tensor


def mamba2_chunked_forward(block, x, *, spatial_dim, center_dim, chunked_time_dim):
    """
    :param block: a Mamba2Block (holds in_proj / conv1d / A_log / dt_bias / D / norm_pre_out / out_proj).
    :param x: rf [batch, chunked_time, window(=center+lookahead), in_dim]  (already layer-normed).
    :param spatial_dim: the window dim (center + lookahead).
    :param center_dim: the center (stride) dim.
    :param chunked_time_dim: the chunks dim.
    :return: rf [batch, chunked_time, window, in_dim].
    """
    from .chunked_conformer_v2_mamba2 import _ssd_chunked

    d_inner = block.d_inner
    n_groups = block.n_groups
    d_state = block.d_state
    n_heads = block.n_heads
    head_dim = block.head_dim
    d_xBC = d_inner + 2 * n_groups * d_state
    d_conv = block.d_conv

    center = center_dim.dimension
    window = spatial_dim.dimension
    lookahead = window - center

    batch_dims = [d for d in x.remaining_dims((spatial_dim, x.feature_dim)) if d != chunked_time_dim]
    assert len(batch_dims) == 1, f"expected one batch dim, got {batch_dims} for {x}"
    batch_dim = batch_dims[0]

    proj = block.in_proj(x)  # [batch, chunked_time, window, d_in_proj]
    proj_t = proj.copy_compatible_to_dims_raw([batch_dim, chunked_time_dim, spatial_dim, block.d_in_proj_dim]).contiguous()
    Bb, Ct = proj_t.shape[0], proj_t.shape[1]

    z_t = proj_t[..., :d_inner]
    xBC_t = proj_t[..., d_inner : d_inner + d_xBC]
    dt_t = proj_t[..., d_inner + d_xBC :]

    conv_w = block.conv1d.filter.raw_tensor
    conv_b = block.conv1d.bias.raw_tensor if block.conv1d.with_bias else None

    def _conv(seq):  # [Bx, T, d_xBC] -> [Bx, T, d_xBC]
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

    a_vec = -torch.exp(block.A_log.raw_tensor)  # [n_heads]
    dt_bias = block.dt_bias.raw_tensor

    # center path: merged contiguous conv + unbounded scan; block_len=center => per-chunk boundary states
    xBC_c = _conv(xBC_t[:, :, :center, :].reshape(Bb, Ct * center, d_xBC))
    x_in_c, b_c, c_c = _heads(xBC_c, Ct * center, Bb)
    dt_c = dt_t[:, :, :center, :].reshape(Bb, Ct * center, n_heads)
    a_c = F.softplus(dt_c + dt_bias) * a_vec
    y_c, fstates = _ssd_chunked(x_in_c, a_c, b_c, c_c, block_len=center, return_final_states=True)
    y_c = y_c.reshape(Bb, Ct, center, n_heads, head_dim)
    x_in_c = x_in_c.reshape(Bb, Ct, center, n_heads, head_dim)

    if lookahead > 0:
        # lookahead path: windowed conv (correct context within the contiguous window) + state-seeded scan
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

    y_w = y_w + x_in_w * block.D.raw_tensor.view(1, 1, 1, n_heads, 1)
    y_w = y_w.reshape(Bb, Ct, window, d_inner)
    y_w = y_w * F.silu(z_t)
    y = Tensor(
        "m2_y_chunked",
        dims=[batch_dim, chunked_time_dim, spatial_dim, block.d_inner_dim],
        dtype=str(y_w.dtype).split(".")[-1],
        raw_tensor=y_w.contiguous(),
        feature_dim=block.d_inner_dim,
    )
    y = block.norm_pre_out(y)
    return block.out_proj(y)


def deltanet_chunked_forward(block, x, *, spatial_dim, center_dim, chunked_time_dim):
    """
    :param block: a DeltaNetBlock (q_proj / k_proj / v_proj / beta_proj / out_proj; no conv).
    :param x: rf [batch, chunked_time, window(=center+lookahead), in_dim]  (already layer-normed).
    :return: rf [batch, chunked_time, window, in_dim].
    """
    from .chunked_conformer_v2_deltanet import _delta_rule_chunked

    H = block.n_heads
    d_state = block.d_state  # Dk
    head_dim = block.head_dim  # Dv
    center = center_dim.dimension
    window = spatial_dim.dimension
    lookahead = window - center

    batch_dims = [d for d in x.remaining_dims((spatial_dim, x.feature_dim)) if d != chunked_time_dim]
    assert len(batch_dims) == 1, f"expected one batch dim, got {batch_dims} for {x}"
    batch_dim = batch_dims[0]

    def _raw(p, feat_dim):
        return p.copy_compatible_to_dims_raw([batch_dim, chunked_time_dim, spatial_dim, feat_dim]).contiguous()

    q = _raw(block.q_proj(x), block.d_qk_dim)
    k = _raw(block.k_proj(x), block.d_qk_dim)
    v = _raw(block.v_proj(x), block.d_v_dim)
    beta = _raw(block.beta_proj(x), block.d_beta_dim)  # [B, Ct, W, H]
    Bb, Ct = q.shape[0], q.shape[1]
    q = q.reshape(Bb, Ct, window, H, d_state)
    k = k.reshape(Bb, Ct, window, H, d_state)
    v = v.reshape(Bb, Ct, window, H, head_dim)
    beta = torch.sigmoid(beta)
    q = F.normalize(q, dim=-1, eps=block.normalize_eps)
    k = F.normalize(k, dim=-1, eps=block.normalize_eps)

    # center: merged contiguous centers, unbounded scan; block_len=center => per-chunk boundary states
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
        "dn_y_chunked",
        dims=[batch_dim, chunked_time_dim, spatial_dim, block.d_v_dim],
        dtype=str(y_w.dtype).split(".")[-1],
        raw_tensor=y_w.contiguous(),
        feature_dim=block.d_v_dim,
    )
    return block.out_proj(y)
