"""
AdEMAMix (Pagliardini, Ablin, Grangier, Apple 2024, "The AdEMAMix Optimizer: Better, Faster, Older").
AdamW plus a **second, slow gradient EMA** (beta3 ~ 0.9999) mixed into the update with weight alpha --
~2x token/update efficiency vs AdamW (better use of each gradient).

The paper ramps alpha and beta3 with a scheduler; here they are fixed (a fine first cut). Only the fast
EMA and the second moment are bias-corrected; the slow EMA is a (deliberately) slow accumulator.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class AdEMAMix(Optimizer):
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999, 0.9999), alpha=5.0, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2, b3 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # fast EMA m1
                    state["exp_avg_slow"] = torch.zeros_like(p)  # slow EMA m2
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m1, m2, v = state["exp_avg"], state["exp_avg_slow"], state["exp_avg_sq"]
                m1.mul_(b1).add_(g, alpha=1 - b1)
                m2.mul_(b3).add_(g, alpha=1 - b3)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m1_hat = m1 / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                denom = v_hat.sqrt().add_(eps)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m1_hat.add(m2, alpha=alpha), denom, value=-lr)
        return loss
