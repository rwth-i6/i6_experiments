"""Device-only qparams for the min/max observers used by QAT."""

import torch


def fast_qparams(min_val, max_val, quant_min, quant_max, eps, symmetric, dtype):
    """Match UniformQuantizationObserverBase without scalar GPU-to-CPU reads.

    Quantization bounds must be Python ints. Scalar qparams retain the observer's
    one-element shape, but use reshape rather than constructing tensors from
    Python scalar values. Uninitialized observers retain the default (1, 0).
    """
    if min_val.numel() == 0 or max_val.numel() == 0:
        return eps.new_ones(1, dtype=torch.float32), eps.new_zeros(1, dtype=torch.int32)
    min_neg = torch.minimum(min_val, torch.zeros_like(min_val))
    max_pos = torch.maximum(max_val, torch.zeros_like(max_val))
    if symmetric:
        max_pos = torch.maximum(-min_neg, max_pos)
        scale = torch.maximum(max_pos / (float(quant_max - quant_min) / 2.0), eps)
        midpoint = (quant_min + quant_max) // 2 if dtype == torch.quint8 else 0
        zero_point = torch.full_like(scale, midpoint, dtype=torch.int32)
    else:
        scale = torch.maximum((max_pos - min_neg) / float(quant_max - quant_min), eps)
        # Match the observer's integer conversion BEFORE subtraction/clamping.
        zero_point = quant_min - torch.round(min_neg / scale).to(torch.int32)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    uninitialized = (min_val == float("inf")) & (max_val == float("-inf"))
    scale = torch.where(uninitialized, torch.ones_like(scale), scale).float()
    zero_point = torch.where(uninitialized, torch.zeros_like(zero_point), zero_point)
    return scale.reshape(-1), zero_point.reshape(-1)


def observer_qparams(observer, quant_min, quant_max):
    """Calculate qparams from buffers; updating the observer remains separate."""
    return fast_qparams(
        observer.min_val,
        observer.max_val,
        quant_min,
        quant_max,
        observer.eps,
        observer.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric),
        observer.dtype,
    )
