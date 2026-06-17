"""
Muon optimizer (MomentUm Orthogonalized by Newton-Schulz), Keller Jordan 2024.
Used in the modded-nanogpt speedruns and nanochat (there with AdamW for embeddings / head / scalars).

Single self-contained hybrid so it drops into RETURNN's one-optimizer slot:
- **Muon** on the 2-D "hidden" matmul weights
  (ndim == 2 with both dims < ``adam_dim_threshold``),
- **AdamW** on everything else
  (1-D norms / biases, the 4-D conv frontend, and the wide 2-D embedding / output projection,
  which are exactly the params nanogpt/nanochat keep on Adam).

The Adam-grouped params use ``lr * adam_lr_ratio`` (Muon's LR is much larger than Adam's),
so a single scheduled LR drives both groups with the right relative scale.

v1 note: the Muon-vs-Adam split is by tensor shape (no param names), so it's a heuristic;
refine to name-based grouping if Muon proves out.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal factor (U V^T) of ``g``."""
    assert g.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transpose = g.size(0) > g.size(1)
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transpose:
        x = x.T
    return x.to(g.dtype)


class Muon(Optimizer):
    """Hybrid Muon (2-D hidden weights) + AdamW (everything else). See module docstring."""

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adam_dim_threshold: int = 8192,
        adam_betas=(0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_lr_ratio: float = 0.025,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adam_dim_threshold=adam_dim_threshold,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            adam_lr_ratio=adam_lr_ratio,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _use_muon(p: torch.Tensor, threshold: int) -> bool:
        return p.ndim == 2 and max(p.shape) < threshold

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            thr = group["adam_dim_threshold"]
            b1, b2 = group["adam_betas"]
            eps = group["adam_eps"]
            adam_lr = lr * group["adam_lr_ratio"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if self._use_muon(p, thr):
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    g_eff = g.add(buf, alpha=mom) if nesterov else buf
                    o = _zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr * scale)
                else:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - b1**t
                    bc2 = 1 - b2**t
                    denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)
                    if wd != 0:
                        p.mul_(1 - adam_lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-adam_lr / bc1)
        return loss
