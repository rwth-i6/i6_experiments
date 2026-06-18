"""
Span masking for BEST-RQ, computed at the *subsampled* (encoder-output) frame rate.

Why at the subsampled rate: the targets are one-per-encoder-frame (the quantizer stacks
``stack_size`` input frames). Generating the mask at the subsampled rate and then expanding
it x``stack_size`` to corrupt the input features guarantees that "masked encoder positions"
and "masked targets" refer to exactly the same frames -- avoiding the target/mask
misalignment bug present in the in-repo implementations we reviewed (mask via max-pool vs.
targets via plain striding).

Masking strategy (paper-faithful, BEST-RQ Sec. 3): every frame is independently a span
*start* with probability ``mask_prob``; each start masks ``mask_length`` consecutive frames
(spans may overlap, so the realized masked fraction is ``~1-(1-p)^L`` and should be measured,
not assumed). At least ``min_masks`` span(s) are forced per sequence so the loss is never
empty. Masking never lands in padding.

Implementation: fully vectorized on-device with torch RNG (no numpy, no per-sequence Python
loop, no CPU<->GPU sync), so it is fast and reproducible under RETURNN's seeding. Each DDP
rank masks its own (different) data independently, which is correct.
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_span_mask(
    *,
    batch_size: int,
    time_size: int,
    lengths: torch.Tensor,
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    :param batch_size: B.
    :param time_size: T2 (subsampled time dim).
    :param lengths: [B] valid (subsampled) frame counts.
    :param mask_prob: per-frame span-start probability.
    :param mask_length: span length L (in subsampled frames).
    :param device: device for the mask.
    :param min_masks: minimum number of span starts forced per sequence.
    :param generator: optional torch.Generator for reproducibility (else global RNG).
    :return: [B, T2] bool mask, True = masked. Never True on padding frames.
    """
    b, t = batch_size, time_size
    lengths = lengths.to(device=device)
    arange_t = torch.arange(t, device=device)

    # A span can only *start* where the full L-frame span fits inside the valid region.
    max_start = (lengths - mask_length + 1).clamp(min=1)  # [B]
    valid_start = arange_t[None, :] < max_start[:, None]  # [B, T]

    rand = torch.rand(b, t, device=device, generator=generator)
    starts = (rand < mask_prob) & valid_start  # [B, T] bool

    # Guarantee >= min_masks starts per sequence: for rows with too few starts, add random
    # extra starts (sampled uniformly over the valid start range) until satisfied.
    if min_masks > 0:
        need = (min_masks - starts.sum(dim=1)).clamp(min=0)  # [B]
        if need.any():
            # Up to min_masks candidate positions per row, uniform in [0, max_start).
            cand = (torch.rand(b, min_masks, device=device, generator=generator) * max_start[:, None]).long()
            cand = cand.clamp(max=t - 1)
            add = torch.zeros_like(starts)
            add.scatter_(1, cand, True)
            # Only add for rows that need it; take only as many as needed by masking via cumsum.
            keep = (torch.cumsum(add.long(), dim=1) <= need[:, None]) & add
            starts = starts | (keep & valid_start)

    # Expand each start at position s to cover [s, s+L-1] via a +1/-1 cumulative-sum trick.
    delta = torch.zeros(b, t + mask_length + 1, device=device, dtype=torch.float32)
    starts_f = starts.to(torch.float32)
    delta[:, :t] += starts_f
    delta[:, mask_length : mask_length + t] -= starts_f
    covered = torch.cumsum(delta, dim=1)[:, :t] > 0.5  # [B, T] bool

    # Never mask padding frames.
    valid = arange_t[None, :] < lengths[:, None]
    return covered & valid
