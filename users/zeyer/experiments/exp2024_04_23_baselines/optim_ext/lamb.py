"""
LAMB optimizer (You et al. 2019, "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes").
AdamW update with a per-layer **trust ratio** (||w|| / ||update||) -- the canonical large-batch optimizer.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class LAMB(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, trust_clip=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, trust_clip=trust_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            clip = group["trust_clip"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                update = m_hat / (v_hat.sqrt() + eps)
                if wd != 0:
                    update = update.add(p, alpha=wd)
                w_norm = p.norm()
                u_norm = update.norm()
                if float(w_norm) > 0.0 and float(u_norm) > 0.0:
                    trust = float((w_norm / u_norm).clamp(max=clip))
                else:
                    trust = 1.0
                p.add_(update, alpha=-lr * trust)
        return loss
