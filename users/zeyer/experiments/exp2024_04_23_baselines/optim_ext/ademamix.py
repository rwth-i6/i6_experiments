"""
AdEMAMix optimizer (Pagliardini, Ablin, Grangier 2024, "The AdEMAMix Optimizer: Better, Faster, Older", ICLR 2025).

Vendored from the official Apple implementation https://github.com/apple/ml-ademamix (pytorch/ademamix.py),
kept close to upstream on purpose.

AdamW plus a second, very slow gradient EMA (beta3, default 0.9999) mixed into the update with weight alpha.
Turning the slow EMA on at full strength from step 0 is unstable (stale old gradients amplified by alpha),
so alpha and beta3 are warmed up:
  - alpha: linear 0 -> alpha over ``alpha_warmup`` steps.
  - beta3: warmed up in EMA-half-life space (beta1 -> beta3) over ``beta3_warmup`` steps.
With both warmups None it degenerates to the fixed-alpha form (which diverged here at ep8).
"""

import math
import torch
from torch.optim import Optimizer

__all__ = ["AdEMAMixV2"]


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMixV2(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        beta3_warmup=None,
        alpha_warmup=None,
        eps=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
        )
        super(AdEMAMixV2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMixV2, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if beta1 != 0.0:
                        state["exp_avg_fast"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_fast, exp_avg_slow, exp_avg_sq = (
                    state["exp_avg_fast"],
                    state["exp_avg_slow"],
                    state["exp_avg_sq"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(
                        state["step"], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup
                    )
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(
                        state["step"], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup
                    )
                else:
                    beta3 = beta3_final

                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update = (exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow) / denom

                update.add_(p, alpha=lmbda)

                p.add_(-lr * update)

        return loss
