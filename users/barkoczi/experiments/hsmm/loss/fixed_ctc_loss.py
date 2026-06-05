"""
copied from Albert
CTC loss with fixed gradient (in case of PyTorch).
See https://github.com/pytorch/pytorch/issues/52241.

This here is PyTorch specific.
We provide a pure PyTorch function and also a RF function.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
#from returnn.tensor import Tensor, Dim


import torch



def torch_ctc_fixed_grad(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Calculates the CTC loss, using :func:`torch.nn.functional.ctc_loss`.

    Only the gradient is different, specifically, it is fixed.
    The Torch CTC loss implementation has a bug in the gradient calculation.
    Specifically, for grad ctc_loss w.r.t. log_probs,
    it calculates exp(log_probs) - y, where y are the soft targets,
    but it should be -y.
    We correct for that here.

    https://github.com/pytorch/pytorch/issues/52241

    Note: Why does the original ctc_loss still usually works fine then?
    Usually it is with log_softmax before.
    grad_{z_j} log_softmax(z)_i = 1_{i=j} - softmax(z)_j.
    Thus (with incorrect grad of torch.ctc_loss w.r.t. log_softmax(z)):
    grad_{z_tj} torch.ctc_loss(log_softmax(z)) = sum_i (softmax(z)_ti - y_ti) * (1_{i=j} - softmax(z)_tj)
      = softmax(z)_tj - y_tj - (sum_i (softmax(z)_ti) - sum_i (y_ti)) * softmax(z)_tj
      = softmax(z)_tj - y_tj.
    I.e. the grad of torch.ctc_loss w.r.t. z is correct.
    The crucial property is that sum_i (softmax(z)_ti - y_ti) = 0.

    :param log_probs: shape [T, N, C]
    :param targets: shape [N, S]
    :param input_lengths: shape [N]
    :param target_lengths: shape [N]
    :param args: passed to :func:`torch.nn.functional.ctc_loss`
    :param kwargs: passed to :func:`torch.nn.functional.ctc_loss`
    :return: loss (either scalar or [N], depending on reduction)
    """
    import torch

    # We avoid the global torch import in this module, thus we lazily define these classes here.
    global _FixCTCGradFunc, _StoreGradScaleFunc
    if not _FixCTCGradFunc or not _StoreGradScaleFunc:

        class _FixCTCGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, log_probs, input_lengths):
                loss_scale_buffer = {}
                ctx.loss_scale_buffer = loss_scale_buffer
                ctx.save_for_backward(log_probs, input_lengths)
                return log_probs, loss_scale_buffer

            @staticmethod
            def backward(ctx, grad_output, _grad_scale):
                loss_scale_buffer = ctx.loss_scale_buffer
                (log_probs, input_lengths) = ctx.saved_tensors
                assert isinstance(loss_scale_buffer, dict) and set(loss_scale_buffer.keys()) == {"scale"}
                # Pop so that we avoid any potential memory leaks.
                loss_scale_buffer: torch.Tensor = loss_scale_buffer.pop("scale")

                # The ctc_loss (incorrectly) calculates (exp(log_probs) - y) * scale,
                # where y are the soft targets,
                # and where we control scale=1 via _StoreGradScaleFunc.
                global _FixedCTCGradStep
                if False and _FixedCTCGradStep % 1000 == 0:  # do sanity check from time to time
                    # if input is not normalized, this will fail
                    sum_res = grad_output[0, 0].sum().detach().cpu()
                    assert -1e-2 <= sum_res <= 1e-2, (
                        f"Unexpected sum of grad_output {sum_res} at step {_FixedCTCGradStep},"
                        f" grad_output {grad_output}, grad_output[0,0] {grad_output[0, 0]}."
                    )
                _FixedCTCGradStep += 1
                # We want to return -y * loss_scale_buffer instead.
                # Thus, subtract the exp(log_probs) from the grad_output.
                grad_input = grad_output - log_probs.exp()  # [T, N, C]
                if loss_scale_buffer.ndim == 1:
                    grad_input.multiply_(loss_scale_buffer[None, :, None])
                else:
                    grad_input.multiply_(loss_scale_buffer)
                input_lengths = input_lengths.to(grad_input.device)
                max_time = grad_input.shape[0]
                mask = torch.arange(max_time, device=input_lengths.device)[:, None] < input_lengths[None, :]  # [T, N]
                grad_input = torch.where(mask[:, :, None], grad_input, torch.zeros_like(grad_input))

                return grad_input, None

        class _StoreGradScaleFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, loss, loss_scale_buffer):
                ctx.loss_scale_buffer = loss_scale_buffer
                return loss.clone()

            @staticmethod
            def backward(ctx, grad_output):
                loss_scale_buffer = ctx.loss_scale_buffer
                assert not loss_scale_buffer
                loss_scale_buffer["scale"] = grad_output
                return torch.ones_like(grad_output), None

    log_probs, loss_scale_buffer = _FixCTCGradFunc.apply(log_probs, input_lengths)
    loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, *args, **kwargs)
    loss = _StoreGradScaleFunc.apply(loss, loss_scale_buffer)
    return loss


_FixCTCGradFunc = None
_StoreGradScaleFunc = None
_FixedCTCGradStep = 0




def ctc_forward_logprob_batch(
    log_probs: torch.Tensor,         # (T_max, B, C)  log-softmax over classes (incl. blank)
    targets: torch.Tensor,           # (B, U_max)     label IDs (no blanks), right-padded
    input_lengths: torch.Tensor,     # (B,)           lengths in time steps
    target_lengths: torch.Tensor,    # (B,)           lengths in labels (no blanks)
    blank: int = 0,
) -> torch.Tensor:
    """
    Vectorized CTC forward: returns (B,) log p(target | x).
    - log_probs must already be log-softmaxed over classes (dim=-1).
    - Handles variable T and U via masks (no Python loops over batch).
    Used to verify the gradients of unnormalized outputs
    """
    assert log_probs.dim() == 3, "log_probs must be (T, B, C)"
    T_max, B, C = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype
    neg_inf = -500000000.0 # this has to be very very large, otherwise compuation is incorrect for long sequences, especially unnormalized probs,

    # ----- Build extended target with blanks: l' of length S_b = 2*U_b + 1 -----
    U_max = targets.size(1)
    S_max = 2 * U_max + 1
    targets = targets.to(torch.long)
    ext = torch.full((B, S_max), blank, dtype=torch.long, device=device)  # start with blanks

    # Fill labels at odd positions up to each U_b
    u_idx = torch.arange(U_max, device=device)             # (U_max,)
    u_mask = u_idx.unsqueeze(0) < target_lengths.unsqueeze(1)  # (B, U_max)
    b_idx, uu = torch.where(u_mask)                        # valid (b, u) pairs
    s_pos = 2 * uu + 1                                     # odd positions in l'
    ext[b_idx, s_pos] = targets[b_idx, uu]

    # State-lengths and masks over S dimension
    S_len = 2 * target_lengths + 1                         # (B,)
    s_idx = torch.arange(S_max, device=device).unsqueeze(0)  # (1, S_max)
    S_mask = s_idx < S_len.unsqueeze(1)                    # (B, S_max), True for valid states

    # ----- Alpha table: (T_max, B, S_max) in log-space -----
    alpha = torch.full((T_max, B, S_max), neg_inf, dtype=dtype, device=device)

    # t=0 initialization (only for sequences with T_b>0)
    active0 = input_lengths > 0                            # (B,)
    if active0.any():
        # alpha[0, :, 0] = log P(blank at t=0)
        a0 = log_probs[0, active0, blank]                  # (B_active,)
        alpha[0, active0, 0] = a0

        # alpha[0, :, 1] = log P(l'_1 at t=0) iff U_b>0 (i.e., S_len>1)
        has_label0 = (target_lengths > 0) & active0        # (B,)
        if has_label0.any():
            # ext[has_label0, 1] are the first label IDs
            cls = ext[has_label0, 1]                       # (B_has,)
            a1 = log_probs[0, has_label0, cls]             # (B_has,)
            alpha[0, has_label0, 1] = a1

    # Ensure invalid states at t=0 are -inf
    alpha[0] = torch.where(S_mask, alpha[0], torch.full_like(alpha[0], neg_inf))

    # Precompute helpers for skip condition: ext != blank and, for s>=2, ext[s] != ext[s-2]
    ext_is_nonblank = (ext != blank)                       # (B, S_max)
    ext_shift2 = torch.full_like(ext, -1)
    ext_shift2[:, 2:] = ext[:, :-2]
    neq_s2 = torch.zeros_like(ext, dtype=torch.bool)
    neq_s2[:, 2:] = ext[:, 2:] != ext[:, :-2]              # True where ext[s] != ext[s-2]
    skip_allowed = ext_is_nonblank & neq_s2                # (B, S_max)

    # ----- DP over time (vectorized over batch and states) -----
    for t in range(1, T_max):
        active_t = t < input_lengths                       # (B,)
        if not active_t.any():
            break

        emit_t = log_probs[t].gather(1, ext)               # (B, S_max) -> log P(z_s at t)

        prev = alpha[t - 1]                                # (B, S_max)

        stay = prev                                        # from s
        move = torch.full_like(prev, neg_inf);  move[:, 1:] = prev[:, :-1]   # from s-1
        skip = torch.full_like(prev, neg_inf);  skip[:, 2:] = prev[:, :-2]   # from s-2
        # mask out disallowed skips
        skip = torch.where(skip_allowed, skip, torch.full_like(skip, neg_inf))

        summed = torch.logsumexp(torch.stack([stay, move, skip], dim=0), dim=0)  # (B, S_max)
        alpha_t = emit_t + summed

        # Mask invalid states and inactive sequences at this t
        alpha_t = torch.where(S_mask, alpha_t, torch.full_like(alpha_t, neg_inf))
        alpha[t] = torch.full_like(alpha_t, neg_inf)
        alpha[t, active_t] = alpha_t[active_t]

    # ----- Termination: logsumexp of last two valid states at t = T_b - 1 -----
    # Gather alpha at per-batch end times
    t_end = (input_lengths - 1).clamp_min(0)               # (B,)
    batch_indices = torch.arange(B, device=device)
    alpha_T = alpha[t_end, batch_indices, :]               # (B, S_max)

    s_last = S_len - 1                                     # (B,)
    s_prev = S_len - 2                                     # (B,) (only valid if target_lengths>0)

    g1 = alpha_T[batch_indices, s_last]                    # (B,)
    g2 = torch.full_like(g1, neg_inf)
    mask_u = target_lengths > 0
    if mask_u.any():
        g2[mask_u] = alpha_T[mask_u, s_prev[mask_u]]

    return torch.logsumexp(torch.stack([g1, g2], dim=0), dim=0)  # (B,)

def ctc_loss_forward_batch(
    log_probs: torch.Tensor,   # (T_max, B, C)  log-softmax over classes (incl. blank)
    targets: torch.Tensor,     # (B, U_max)     int labels (no blanks), right-padded
    input_lengths: torch.Tensor,# (B,)
    target_lengths: torch.Tensor,# (B,)
    blank: int = 0,
    zero_infinity: bool = False,
) -> torch.Tensor:
    # Compute per-sample loss = -log p(l|x)
    logp = ctc_forward_logprob_batch(log_probs, targets, input_lengths, target_lengths, blank=blank)
    loss = -logp
    if zero_infinity:
        # Match PyTorch: set inf losses to 0 and zero their gradients
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
    return loss  # (B,)



def compare_grads_log_probs(
    logits: torch.Tensor,           # (T, B, C) raw scores
    targets: torch.Tensor,          # (B, U_max) int64, padded
    input_lengths: torch.Tensor,    # (B,)
    target_lengths: torch.Tensor,   # (B,)
    *,
    blank: int = 0,
    zero_infinity: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):

    targets = targets.to(torch.long)
    assert targets.dtype == torch.long, "targets must be int64"
    device = logits.device

    # Clone two independent log_probs tensors to get separate gradients
    log_probs1 = logits.detach().clone().requires_grad_(True)
    log_probs2 = logits.detach().clone().requires_grad_(True)

    # ===== Path A: our DP forward + autograd on log_probs =====
    lossA = ctc_loss_forward_batch(
        log_probs1, targets, input_lengths, target_lengths, blank=blank, zero_infinity=zero_infinity
    ).sum()  # scalar for backward
    lossA.backward()
    gradA = log_probs1.grad.detach()

    # ===== Path B: your fixed-grad wrapper on log_probs =====
    lossB = torch_ctc_fixed_grad(
        log_probs2, targets, input_lengths, target_lengths,
        blank=blank, reduction="sum", zero_infinity=zero_infinity
    )
    lossB.backward()
    gradB = log_probs2.grad.detach()
    print("gradA", gradA[230:233,0])
    print("gradB", gradB[230:233,0])

    # Compare
    diff = (gradA - gradB).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (gradA.abs() + 1e-12)).max().item()
    ok = torch.allclose(gradA, gradB, atol=atol, rtol=rtol)

    max_value_diff = (lossA.detach()-lossB.detach()).abs().max().item()

    # Optional: mask out inactive time steps (t >= input_length)
    # (Both paths should already zero those, but you can double-check)
    # build mask [T,B,1] and check torch.max(diff[~mask]) as needed.

    return {
        "ok": ok, "max_abs_diff": max_abs, "max_rel_diff": max_rel,
        "lossA": lossA.detach(), "lossB": lossB.detach(), "max_value_abs_diff": max_value_diff,
    }