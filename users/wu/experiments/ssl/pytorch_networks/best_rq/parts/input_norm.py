"""
Input feature normalization for BEST-RQ.

BEST-RQ (Chiu et al. 2022) states that normalizing the input features to ~0 mean /
unit std is *critical*: without it the random projection collapses to a small subset of
codes (poor codebook utilization). We use a **per-utterance, per-feature** mean/variance
normalization computed over the valid (non-padding) frames. This is:

* deterministic (a pure function of the utterance) -> the quantizer targets are stable
  across training, unlike a running/global statistic that keeps drifting;
* stateless and therefore trivially DDP-safe (no buffers to sync across ranks);
* padding-aware (statistics ignore padded frames; padded frames are zeroed out).

The *same* normalized features must feed both the frozen quantizer (which produces the
targets) and the (masked) encoder input -- never normalize one and not the other.
"""

from __future__ import annotations

import torch


def apply_global_norm(
    features: torch.Tensor, lengths: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Fixed corpus-level (global) normalization: ``(x - mean) / std`` with precomputed per-feature
    ``mean``/``std`` ([F]). This is the **preferred** BEST-RQ input normalization -- it is a fixed
    affine, so it is sequence-independent: a given acoustic frame maps to the same quantizer target
    regardless of the utterance it appears in (unlike per-utterance norm). Padded frames are zeroed.

    :param features: [B, T, F].
    :param lengths: [B] valid frame counts.
    :param mean: [F] global per-feature mean.
    :param std: [F] global per-feature std.
    :return: [B, T, F] normalized, padding zeroed.
    """
    t = features.shape[1]
    valid = (torch.arange(t, device=features.device)[None, :] < lengths.to(features.device)[:, None]).unsqueeze(-1)
    normed = (features - mean) / std
    return normed * valid.to(features.dtype)


def masked_mean_var_norm(features: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Per-utterance, per-feature normalization to zero mean / unit variance over valid frames.

    :param features: [B, T, F] input features (e.g. log-mel).
    :param lengths: [B] number of valid frames per sequence.
    :param eps: variance floor for numerical stability.
    :return: [B, T, F] normalized features, with padded frames set to 0.
    """
    assert features.dim() == 3, f"expected [B,T,F], got {tuple(features.shape)}"
    b, t, f = features.shape
    device = features.device
    lengths = lengths.to(device=device)

    valid = torch.arange(t, device=device)[None, :] < lengths[:, None]  # [B, T] bool, True = valid
    mask = valid.unsqueeze(-1).to(features.dtype)  # [B, T, 1]
    n = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]

    mean = (features * mask).sum(dim=1) / n  # [B, F]
    centered = features - mean.unsqueeze(1)  # [B, T, F]
    var = (centered * centered * mask).sum(dim=1) / n  # [B, F]
    std = torch.sqrt(var + eps)  # [B, F]

    normed = centered / std.unsqueeze(1)
    normed = normed * mask  # keep padded frames at exactly 0
    return normed
