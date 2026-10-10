"""Batched CTC prefix scoring for frame-synchronous DSM decoding.

Adapted from ESPnet's ``CTCPrefixScoreTH`` for per-beam prefix metadata and structural DSM tokens.
Blank occupies the final vocabulary column.
"""

from typing import Tuple

import torch

# finite log(0) avoids undefined ``-inf - -inf`` increments for infeasible prefixes
LOG0 = -1e10


def prepare_ctc_log_probs(
    ctc_log_probs: torch.Tensor, enc_lens: torch.Tensor, *, blank_id: int
) -> torch.Tensor:
    """Mask padded frames so they can emit only blank.

    At ``t >= enc_lens[b]``, blank receives log probability zero and all labels
    receive :data:`LOG0`.
    """
    x = ctc_log_probs.clone()
    B, T, _ = x.shape
    idx = torch.arange(T, device=x.device).unsqueeze(0)  # shape: (1, T)
    pad = idx >= enc_lens.to(x.device).unsqueeze(1)  # shape: (B, T)
    x[pad] = LOG0
    x[..., blank_id][pad] = 0.0
    return x


def initial_r(x: torch.Tensor, *, blank_id: int) -> torch.Tensor:
    """Initialize the forward table for an empty prefix.

    The result has shape ``(B, T, 2)``. Column 0 remains :data:`LOG0`, while
    column 1 contains cumulative blank log probabilities.
    """
    B, T, _ = x.shape
    r = x.new_full((B, T, 2), LOG0)
    r[:, :, 1] = torch.cumsum(x[:, :, blank_id], dim=1)
    return r


def extend(
    x: torch.Tensor,
    r_parent: torch.Tensor,
    s_parent: torch.Tensor,
    last_parent: torch.Tensor,
    length_parent: torch.Tensor,
    min_frames_parent: torch.Tensor,
    cand: torch.Tensor,
    *,
    blank_id: int,
    pad_id: int,
    word_id: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Extend each parent prefix by one candidate.

    Forward tables have shape ``(B, P, T, 2)``. PAD and WORD preserve the
    parent CTC state. ``min_frames_child`` tracks the minimum CTC path length.
    ``psi_cum[..., f]`` is the prefix probability truncated to frames ``0..f``,
    meaningful only for real-label candidates.
    """
    B, P = cand.shape
    T = x.shape[1]
    is_real = (cand != pad_id) & (cand != word_id)  # shape: (B, P)

    # frame scores for each proposed label and for blank
    x_c = torch.gather(x, 2, cand.unsqueeze(1).expand(B, T, P)).transpose(1, 2)
    x_bl = x[:, :, blank_id].unsqueeze(1).expand(B, P, T)

    r_sum = torch.logaddexp(r_parent[..., 0], r_parent[..., 1])  # shape: (B, P, T)
    # adjacent equal labels require a blank separator
    repeat = (cand == last_parent) & (length_parent > 0)  # shape: (B, P)
    log_phi = torch.where(repeat.unsqueeze(-1), r_parent[..., 1], r_sum)  # shape: (B, P, T)

    r_child = x.new_full((B, P, T, 2), LOG0)
    psi_cum = x.new_full((B, P, T), LOG0)
    empty = length_parent == 0  # shape: (B, P)
    # only the first label can complete at frame zero
    r_child[:, :, 0, 0] = torch.where(empty, x_c[:, :, 0], x.new_full((B, P), LOG0))
    log_psi = torch.where(empty, x_c[:, :, 0], x.new_full((B, P), LOG0))  # shape: (B, P)
    psi_cum[:, :, 0] = log_psi

    for t in range(1, T):
        prev_n = r_child[:, :, t - 1, 0]
        prev_b = r_child[:, :, t - 1, 1]
        phi_tm1 = log_phi[:, :, t - 1]
        r_child[:, :, t, 0] = torch.logaddexp(prev_n, phi_tm1) + x_c[:, :, t]
        r_child[:, :, t, 1] = torch.logaddexp(prev_n, prev_b) + x_bl[:, :, t]
        log_psi = torch.logaddexp(log_psi, phi_tm1 + x_c[:, :, t])
        psi_cum[:, :, t] = log_psi

    # clamp preserves prefix-probability monotonicity under roundoff and finite-LOG0 arithmetic
    incr = torch.where(is_real, (log_psi - s_parent).clamp(max=0.0), x.new_zeros((B, P)))
    r_child = torch.where(is_real.view(B, P, 1, 1), r_child, r_parent)
    s_child = torch.where(is_real, log_psi, s_parent)
    last_child = torch.where(is_real, cand, last_parent)
    length_child = torch.where(is_real, length_parent + 1, length_parent)
    min_frames_child = torch.where(
        is_real, min_frames_parent + 1 + repeat.long(), min_frames_parent
    )
    return incr, r_child, s_child, last_child, length_child, min_frames_child, psi_cum


def close_sequence(
    r_final: torch.Tensor, s_final: torch.Tensor, end_frames: torch.Tensor
) -> torch.Tensor:
    """Return the EOS delta from prefix probability to exact-sequence probability."""
    B, K, T, _ = r_final.shape
    r_sum = torch.logaddexp(r_final[..., 0], r_final[..., 1])  # shape: (B, K, T)
    # use long indices for torch builds that reject int32 in gather
    end = end_frames.to(r_final.device).clamp(min=0, max=T - 1).long()  # shape: (B,)
    r_sum_end = torch.gather(
        r_sum, 2, end.view(B, 1, 1).expand(B, K, 1)
    ).squeeze(2)  # shape: (B, K)
    # exact-sequence paths are a subset of prefix paths
    return (r_sum_end - s_final).clamp(max=0.0)
