"""
CTC loss with fixed gradient (in case of PyTorch).
See https://github.com/pytorch/pytorch/issues/52241.

This here is PyTorch specific.
We provide a pure PyTorch function and also a RF function.
"""


from __future__ import annotations
from typing import TYPE_CHECKING
from returnn.tensor import Tensor, Dim

if TYPE_CHECKING:
    import torch


def ctc_loss_fixed_grad(
    *,
    logits: Tensor,
    logits_normalized: bool = False,
    targets: Tensor,
    input_spatial_dim: Dim,
    targets_spatial_dim: Dim,
    blank_index: int,
    max_approx: bool = False,
) -> Tensor:
    """
    Calculates the CTC loss, using :func:`torch_ctc_fixed_grad`.

    Output is of shape [B].

    :param logits: (before softmax). shape [B...,input_spatial,C]
    :param logits_normalized: whether the logits are already normalized (e.g. via log-softmax)
    :param targets: sparse. shape [B...,targets_spatial] -> C
    :param input_spatial_dim: spatial dim of input logits
    :param targets_spatial_dim: spatial dim of targets
    :param blank_index: vocab index of the blank symbol
    :param max_approx: if True, use max instead of sum over alignments (max approx, Viterbi)
    :return: loss shape [B...]
    """
    import torch
    from returnn.util.basic import prod

    assert isinstance(logits.raw_tensor, torch.Tensor)
    if max_approx:
        raise NotImplementedError("ctc_loss: max_approx not implemented for PyTorch")
    assert targets.sparse_dim and targets.sparse_dim.dimension <= logits.feature_dim.dimension
    # PyTorch expects the logits to be of shape (T, B, C) where T is the input spatial dim.
    batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))
    batch_shape = [d.get_dim_value() for d in batch_dims]
    batch_n_elems = prod(batch_shape)
    logits = logits.copy_transpose([input_spatial_dim] + batch_dims + [logits.feature_dim])
    logits_raw: torch.Tensor = logits.raw_tensor
    input_lengths: torch.Tensor = input_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw(batch_dims)
    if input_lengths.numel() != batch_n_elems:
        input_lengths = input_lengths.expand(batch_shape)
    logits_raw_shape = logits_raw.shape  # [T, B..., C]
    if len(batch_dims) != 1:
        logits_raw = torch.reshape(
            logits_raw, logits_raw.shape[:1] + (batch_n_elems,) + logits_raw.shape[-1:]
        )  # [T, B', C]
        input_lengths = torch.reshape(input_lengths, (batch_n_elems,))  # [B']
    if logits_normalized:
        log_probs = logits_raw
    else:
        log_probs = torch.nn.functional.log_softmax(logits_raw, dim=-1)
    # PyTorch expects the targets to be of shape (B, S) where S is the targets spatial dim.
    targets_raw = targets.copy_compatible_to_dims_raw(batch_dims + [targets_spatial_dim])  # [B..., S]
    targets_raw_shape = batch_shape + [targets_spatial_dim.get_dim_value()]
    if targets_raw.numel() != prod(targets_raw_shape):
        targets_raw = targets_raw.expand(targets_raw_shape)
    targets_lengths = targets_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw(batch_dims)
    if targets_lengths.numel() != batch_n_elems:
        targets_lengths = targets_lengths.expand(batch_shape)
    if len(batch_dims) != 1:
        targets_raw = torch.reshape(targets_raw, (batch_n_elems, targets_raw.shape[-1]))  # [B', S]
        targets_lengths = torch.reshape(targets_lengths, (batch_n_elems,))  # [B']
    if log_probs.dtype == torch.bfloat16:
        # Currently (PyTorch 2.5), ctc_loss does not support bfloat16.
        log_probs = log_probs.to(torch.float32)
    loss_raw = torch_ctc_fixed_grad(
        log_probs=log_probs,
        targets=targets_raw,
        input_lengths=input_lengths,
        target_lengths=targets_lengths,
        blank=blank_index,
        zero_infinity=True,
        reduction="none",
    )
    if len(batch_dims) != 1:
        loss_raw = torch.reshape(loss_raw, logits_raw_shape[1:-1])
    loss = Tensor(
        name="ctc_loss", dims=batch_dims, raw_tensor=loss_raw, dtype=str(loss_raw.dtype).replace("torch.", "")
    )
    return loss


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

    :param log_probs: shape [T, N, C]
    :param targets: shape [N, S]
    :param input_lengths: shape [N]
    :param target_lengths: shape [N]
    :param args: passed to :func:`torch.nn.functional.ctc_loss`
    :param kwargs: passed to :func:`torch.nn.functional.ctc_loss`
    :return: loss (either scalar or [N], depending on reduction)
    """
    import torch

    global _FixCTCGradFunc
    if not _FixCTCGradFunc:

        class _FixCTCGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, log_probs, input_lengths):
                ctx.save_for_backward(log_probs, input_lengths)
                return log_probs

            @staticmethod
            def backward(ctx, grad_output):
                (log_probs, input_lengths) = ctx.saved_tensors

                # The ctc_loss calculates exp(log_probs) - y, where y are the soft targets.
                # We want to return -y instead.
                # Thus, subtract the exp(log_probs) from the grad_output.
                grad_input = grad_output - log_probs.exp()
                input_lengths = input_lengths.to(grad_input.device)
                max_time = grad_input.shape[0]
                mask = torch.arange(max_time, device=input_lengths.device)[:, None] < input_lengths[None, :]  # [T, N]
                grad_input = torch.where(mask[:, :, None], grad_input, torch.zeros_like(grad_input))

                return grad_input, None

    log_probs = _FixCTCGradFunc.apply(log_probs, input_lengths)
    return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, *args, **kwargs)


_FixCTCGradFunc = None