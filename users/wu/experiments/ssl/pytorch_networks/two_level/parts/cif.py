"""
Continuous Integrate-and-Fire (CIF) segmentation + pooling -- the core novelty of the two-level
BEST-RQ + CIF model.

NEW (two-level/CIF logic). Given frozen lower-encoder (layer-9, 25 Hz) frame features and a per-frame
"fire weight" ``alpha_t in (0,1)``, this pools the frames into a *variable* number ``K`` of
higher-level tokens by the ORIGINAL CIF integrate-and-fire rule (Dong & Xu 2020, arXiv:1905.11235):
accumulate ``alpha`` over time, emit one token each time the accumulated mass crosses an integer, and
split the *firing* frame's weight continuously across the closing token and the next one.

v1 design (decided with the user):
* RAW integrate-and-fire -- NO global alpha scaling/normalization, NO alpha cap. Token length is only
  *softly* controlled by a quantity loss (``quantity_loss``).
* Each COMPLETED token's contributing weights sum to exactly 1.0 (a convex combination of frame
  features), so no extra per-token normalization is needed.
* The final partial token (residual mass < 1) is DROPPED. Exception: if an utterance produces zero
  completed tokens, the single partial token is kept (a min-1 floor) so the high encoder always sees
  >= 1 token and the loss is never empty.

Differentiability (the property the whole idea rests on): the token COUNT (``floor`` of the cumulative
mass) and the integer token routing are non-differentiable, but each frame's contribution WEIGHTS are
differentiable in ``alpha`` (they are affine in the cumulative sums of ``alpha``). So gradient flows
``CE -> pooled z -> alpha -> predictor`` -- this is what lets masked prediction shape the segmentation,
not just the quantity loss. ``test_cif.py`` asserts this with a gradient-flow check.

Because ``sigmoid`` keeps ``alpha < 1``, a single frame's mass crosses AT MOST one integer boundary, so
each frame contributes to at most two consecutive tokens. That is what makes the pooling a clean, fully
vectorized two-``scatter_add`` op (no per-frame Python loop, no per-element CPU sync). The only host
sync is one ``K.max()`` to size the padded token axis -- unavoidable and cheap (once per forward).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class CIFAlphaPredictor(nn.Module):
    """Small local temporal module predicting the per-frame CIF fire weight ``alpha_t = sigmoid(s_t)``.

    LayerNorm -> Conv1d(kernel, 'same') -> GELU -> Linear(->1) -> sigmoid, over the (frozen) layer-9
    features. The convolution gives ``alpha`` a small temporal receptive field so a fire can depend on
    a short neighbourhood, not just one frame.
    """

    def __init__(self, dim: int, kernel_size: int = 5, hidden_dim: Optional[int] = None):
        super().__init__()
        assert kernel_size % 2 == 1, "use an odd kernel_size for symmetric 'same' padding"
        hidden_dim = hidden_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: [B, T, D]  :return: alpha [B, T] in (0, 1)."""
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden]
        x = self.act(x)
        return torch.sigmoid(self.proj(x).squeeze(-1))  # [B, T]


def cif_pool(
    features: torch.Tensor,
    alpha: torch.Tensor,
    lengths: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Raw integrate-and-fire CIF pooling.

    :param features: [B, T, D] lower-encoder frame features (the caller passes the FROZEN, detached
        layer-9 output; gradient must NOT flow into the encoder through here).
    :param alpha: [B, T] per-frame fire weights in (0, 1) (trainable; gradient flows here).
    :param lengths: [B] valid frame counts (frames >= length are padding).
    :param eps: small constant to stabilize ``floor`` exactly on integer boundaries.
    :return: (z, z_mask, diag)
        z: [B, K_max, D] pooled higher-level tokens (zeros on padded token slots);
        z_mask: [B, K_max] bool, True = valid (fired) token;
        diag: dict of per-utterance diagnostics: ``fires`` [B] (K), ``sum_alpha`` [B], ``alpha`` [B, T]
              (padding-masked), ``frame_valid`` [B, T] bool.
    """
    # The integrate-and-fire bookkeeping MUST be fp32: under bf16 the cumulative sum saturates (the bf16
    # ULP exceeds a typical alpha once the running mass passes ~64) and the fire/weight split silently
    # decouples from the true CIF, breaking the convex-combination invariant. Cast here so the pooler is
    # correct regardless of the caller's autocast state (the model also disables autocast around this).
    features = features.float()
    alpha = alpha.float()
    b, t, d = features.shape
    device = features.device
    # Zero alpha on padding frames so they neither fire nor contribute mass.
    frame_valid = torch.arange(t, device=device)[None, :] < lengths.to(device)[:, None]  # [B, T]
    alpha = alpha * frame_valid.to(alpha.dtype)

    csum = torch.cumsum(alpha, dim=1)  # [B, T] cumulative mass AFTER frame t
    prev = csum - alpha                # [B, T] cumulative mass BEFORE frame t

    # 0-based index of the token that frame t begins contributing to, and whether t crosses an integer.
    j = torch.floor(prev + eps)                                  # [B, T] float
    fired = torch.floor(csum + eps) > j                          # [B, T] bool (crosses boundary j+1)
    boundary = j + 1.0                                           # the integer crossed when fired
    # Split frame t's mass: w_over -> next token (j+1) only if it fired; w_stay -> current token (j).
    w_over = torch.where(fired, csum - boundary, torch.zeros_like(csum))  # [B, T] in [0, alpha)
    w_stay = alpha - w_over                                                # [B, T]  (= alpha if not fired)
    j = j.long()

    # Per-utterance completed-token count K = floor(total mass), floored to >=1 (min-1: keep the single
    # partial token for utts whose mass never reaches 1, so the high encoder always sees a token).
    last = (lengths.to(device) - 1).clamp(min=0)[:, None]
    total = csum.gather(1, last).squeeze(1)            # [B] total accumulated mass
    K = torch.floor(total + eps).long().clamp(min=1)   # [B] valid token count
    k_max = int(K.max().item())                        # one host sync to size the padded token axis

    # Scatter-add weighted frame features into the token axis. Allocate one slop column [k_max] to
    # receive the dropped partial/overflow token, then trim it. Each frame writes to <=2 token slots.
    z = features.new_zeros(b, k_max + 1, d)
    j_stay = j.clamp(min=0, max=k_max)
    j_over = (j + 1).clamp(min=0, max=k_max)
    z.scatter_add_(1, j_stay.unsqueeze(-1).expand(-1, -1, d), w_stay.unsqueeze(-1) * features)
    z.scatter_add_(1, j_over.unsqueeze(-1).expand(-1, -1, d), w_over.unsqueeze(-1) * features)
    z = z[:, :k_max, :]  # drop the partial/residual token beyond K

    z_mask = torch.arange(k_max, device=device)[None, :] < K[:, None]  # [B, k_max] True = valid token
    z = z * z_mask.unsqueeze(-1).to(z.dtype)

    diag = {"fires": K, "sum_alpha": total, "alpha": alpha, "frame_valid": frame_valid}
    return z, z_mask, diag


def quantity_loss(
    sum_alpha: torch.Tensor,
    lengths: torch.Tensor,
    *,
    target_rate_hz: float,
    frame_rate_hz: float = 25.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Utterance-level CIF quantity loss (soft length control).

    For each utterance ``Q_b = sum_t alpha_{b,t}`` (already padding-masked) and a rate-derived target
    ``Q*_b = T_b * (target_rate_hz / frame_rate_hz)`` (note: ``Q*`` depends only on the input length
    ``T_b``, so it is available at train AND inference). Loss = mean_b ((Q_b - Q*_b)/(Q*_b + eps))^2.

    :param sum_alpha: [B] total accumulated mass per utterance (``diag['sum_alpha']``).
    :param lengths: [B] valid frame counts ``T_b``.
    :param target_rate_hz: desired CIF token rate (e.g. 12.5 for 80 ms, 8.333 for 120 ms at 25 Hz).
    :param frame_rate_hz: lower-encoder frame rate (25 Hz here -- DERIVED, see two_level_v1).
    """
    q_star = lengths.to(sum_alpha.dtype) * (target_rate_hz / frame_rate_hz)  # [B]
    rel = (sum_alpha - q_star) / (q_star + eps)
    return (rel * rel).mean()
