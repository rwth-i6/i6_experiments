"""Deterministic parameter-noise injection, for controlled model-degradation studies.

Adds Gaussian noise to every model parameter, scaled per-tensor by that tensor's
std (so the relative perturbation is uniform across layers of different scale):

    theta += std * std(theta) * eps,   eps ~ N(0, 1)

Deterministic given (std, seed) and a fixed parameter iteration order, so the SAME
perturbed model is reproduced across separate jobs (grad-align vs forced-align vs
cross-attn) at a given noise level -> WBE and WER are measured on the same model.
"""

from __future__ import annotations


def apply_param_noise(model, std: float, seed: int = 0) -> None:
    import torch

    if not std:
        return
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() == 0:
                continue
            noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
            scale = std * float(p.detach().float().std().clamp_min(1e-8).item())
            p.add_((noise * scale).to(dtype=p.dtype, device=p.device))
